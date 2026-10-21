import cv2
import numpy as np
import math
import os

# ==========================================
# HELPER FUNCTIONS & MATH
# ==========================================

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0

def calculate_containment(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    
    min_area = min(areaA, areaB)
    return float(inter_area) / min_area if min_area > 0 else 0.0

def custom_nms_with_containment(boxes, scores, iou_thresh=0.22, containment_thresh=0.75):
    if len(boxes) == 0:
        return []
        
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep_indices = []
    
    for i in sorted_indices:
        box_i = boxes[i]
        keep = True
        
        for j in keep_indices:
            box_j = boxes[j]
            iou = calculate_iou(box_i, box_j)
            containment = calculate_containment(box_i, box_j)
            
            if iou > iou_thresh or containment > containment_thresh:
                keep = False
                break
                
        if keep:
            keep_indices.append(i)
            
    return keep_indices

def _to_gray_float64(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr.astype(np.float64, copy=False))
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] == 3 else cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    return np.ascontiguousarray(gray.astype(np.float64, copy=False))

def lap4_variance_of_laplacian(image: np.ndarray, window_size: int = 9) -> np.ndarray:
    img = _to_gray_float64(image)
    lap_kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    lap = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=lap_kernel, borderType=cv2.BORDER_REFLECT101)
    area = float(window_size * window_size)
    sum_lap = cv2.boxFilter(lap, ddepth=cv2.CV_64F, ksize=(window_size, window_size), normalize=False, borderType=cv2.BORDER_REFLECT101)
    sum_lap2 = cv2.boxFilter(lap * lap, ddepth=cv2.CV_64F, ksize=(window_size, window_size), normalize=False, borderType=cv2.BORDER_REFLECT101)
    response_map = sum_lap2 - (sum_lap * sum_lap) / area
    return np.maximum(response_map, 0.0)

def compute_color_divergence(bgr_image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_color = np.mean(lab, axis=(0, 1))
    diff = lab - mean_color
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)

# ==========================================
# RELATIVE TRACKING CLASSES
# ==========================================

class TrackedSubject:
    def __init__(self, box, track_id, score=1.0):
        self.track_id = track_id
        self.box = list(box) 
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.score = score

    def update(self, new_box, score, deadband_px, alpha_pos=0.25, alpha_size=0.12):
        self.hits += 1
        self.misses = 0
        self.score = 0.7 * self.score + 0.3 * score

        curr_cx = self.box[0] + self.box[2] / 2.0
        curr_cy = self.box[1] + self.box[3] / 2.0
        new_cx = new_box[0] + new_box[2] / 2.0
        new_cy = new_box[1] + new_box[3] / 2.0

        dist = math.hypot(new_cx - curr_cx, new_cy - curr_cy)
        if dist > deadband_px:
            smooth_cx = alpha_pos * new_cx + (1.0 - alpha_pos) * curr_cx
            smooth_cy = alpha_pos * new_cy + (1.0 - alpha_pos) * curr_cy
        else:
            smooth_cx, smooth_cy = curr_cx, curr_cy

        smooth_w = alpha_size * new_box[2] + (1.0 - alpha_size) * self.box[2]
        smooth_h = alpha_size * new_box[3] + (1.0 - alpha_size) * self.box[3]

        self.box = [
            int(smooth_cx - smooth_w / 2.0),
            int(smooth_cy - smooth_h / 2.0),
            int(smooth_w),
            int(smooth_h)
        ]

class MultiSubjectPersistenceTracker:
    def __init__(self, max_tracks=3, min_hits=4, max_age=5, iou_thresh=0.20):
        self.max_tracks = max_tracks
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracks = []
        self.next_id = 0

    def update(self, detected_boxes, detected_scores, w, h):
        matched_tracks = set()
        matched_detections = set()

        dist_thresh = max(w, h) * 0.10
        deadband_px = max(w, h) * 0.005 

        for t_idx, track in enumerate(self.tracks):
            best_score = 0.0
            best_d_idx = -1

            t_cx = track.box[0] + track.box[2] / 2.0
            t_cy = track.box[1] + track.box[3] / 2.0

            for d_idx, det in enumerate(detected_boxes):
                if d_idx in matched_detections: continue

                iou = calculate_iou(track.box, det)
                d_cx = det[0] + det[2] / 2.0
                d_cy = det[1] + det[3] / 2.0
                dist = math.hypot(t_cx - d_cx, t_cy - d_cy)

                if iou > self.iou_thresh and iou > best_score:
                    best_score = iou
                    best_d_idx = d_idx
                elif best_score == 0.0 and dist < dist_thresh:
                    best_d_idx = d_idx

            if best_d_idx != -1:
                track.update(detected_boxes[best_d_idx], detected_scores[best_d_idx], deadband_px)
                if track.hits >= self.min_hits:
                    track.confirmed = True
                matched_tracks.add(t_idx)
                matched_detections.add(best_d_idx)
            else:
                track.misses += 1

        for d_idx, det in enumerate(detected_boxes):
            if d_idx not in matched_detections and len(self.tracks) < (self.max_tracks + 2):
                self.tracks.append(TrackedSubject(det, self.next_id, detected_scores[d_idx]))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        confirmed_tracks = [t for t in self.tracks if t.confirmed and t.misses == 0]
        confirmed_tracks.sort(key=lambda t: t.score, reverse=True)
        return [t.box for t in confirmed_tracks[:self.max_tracks]]

# ==========================================
# MAIN ANALYZER CLASS
# ==========================================

class CompositionAnalyzer:
    """
    Stateful Python engine designed to be instantiated once by Chaquopy.
    Maintains tracking history across frames.
    """
    def __init__(self, cache_dir=None):
        current_dir = os.path.dirname(__file__)
        self.prototxt = os.path.join(current_dir, "deploy.prototxt")
        self.caffemodel = os.path.join(current_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.edge_model = os.path.join(current_dir, "model.yml.gz")
        
        # Initialize Networks
        self.face_net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection(self.edge_model)
        self.edge_boxes = cv2.ximgproc.createEdgeBoxes()
        self.edge_boxes.setMaxBoxes(75) 
        
        # Initialize Tracker
        self.tracker = MultiSubjectPersistenceTracker(max_tracks=3, min_hits=4, max_age=5)
        self.max_proc_dim = 360  # Anchor length for dynamic tensor resizing

    def _generate_feedback(self, faces, objects, w, h):
        """Semantic Decision Engine integrating geometric and aspect ratio rules."""
        is_portrait = h > w
        third_x1, third_x2 = w // 3, 2 * (w // 3)
        third_y1, third_y2 = h // 3, 2 * (h // 3)
        
        tol_x = w * (0.06 if is_portrait else 0.08)
        tol_y = h * 0.08 

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[0])
            
            if len(faces) == 1:
                fx, fy, fw, fh = faces[0]
                face_cx = fx + fw // 2
                eye_y = fy + fh * 0.35
                
                target_x = (w // 2) if is_portrait else (third_x1 if face_cx < w // 2 else third_x2)
                target_y = third_y1
                
                if eye_y - target_y > tol_y: return "Move Up (Fix Headroom)"
                if target_y - eye_y > tol_y: return "Move Down (Fix Headroom)"
                if face_cx - target_x > tol_x: return "Move Left (Align Face)"
                if target_x - face_cx > tol_x: return "Move Right (Align Face)"
                
                # Mixed Frame Balancing
                if len(objects) >= 1:
                    ox, oy, ow, oh = objects[0]
                    obj_cx, obj_cy = ox + ow // 2, oy + oh // 2
                    
                    width_pct, height_pct = ow / float(w), oh / float(h)
                    ratio = ow / float(max(1, oh))
                    is_tall = height_pct > 0.33 or ratio < 0.75
                    
                    diag_x = (w // 2) if is_portrait else (third_x2 if target_x == third_x1 else third_x1)
                    diag_y = third_y2
                    
                    if obj_cx - diag_x > tol_x: return "Move Object Left (Balance)"
                    if diag_x - obj_cx > tol_x: return "Move Object Right (Balance)"
                    
                    if not is_tall:
                        if obj_cy - diag_y > tol_y: return "Move Object Up (Balance)"
                        if diag_y - obj_cy > tol_y: return "Move Object Down (Balance)"
                    
                return "ALIGNED"

            elif len(faces) == 2:
                f1x, f1y, f1w, f1h = faces[0]
                f1_cx, f1_eye_y = f1x + f1w // 2, f1y + f1h * 0.35
                
                if f1_eye_y - third_y1 > tol_y: return "Move Up (Left Person Headroom)"
                if third_y1 - f1_eye_y > tol_y: return "Move Down (Left Person Headroom)"
                if f1_cx - third_x1 > tol_x: return "Move Left (Align Left Person)"
                if third_x1 - f1_cx > tol_x: return "Move Right (Align Left Person)"
                
                f2x, f2y, f2w, f2h = faces[1]
                f2_cx = f2x + f2w // 2
                
                if f2_cx - third_x2 > tol_x: return "Move Left (Align Right Person)"
                if third_x2 - f2_cx > tol_x: return "Move Right (Align Right Person)"
                
                return "ALIGNED"
            else:
                return "ALIGNED (Wide Shot)"

        else:
            # Objects Only
            if len(objects) > 3: return "Cluttered Frame: Move Closer"
            
            if len(objects) == 1:
                ox, oy, ow, oh = objects[0]
                obj_cx, obj_cy = ox + ow // 2, oy + oh // 2
                
                width_pct, height_pct = ow / float(w), oh / float(h)
                ratio = ow / float(max(1, oh))
                is_wide = width_pct > 0.33 or ratio > 1.3
                is_tall = height_pct > 0.33 or ratio < 0.75
                
                if is_wide and is_tall:
                    target_x, target_y = w // 2, h // 2
                    if obj_cx - target_x > tol_x: return "Move Left (Center Object)"
                    if target_x - obj_cx > tol_x: return "Move Right (Center Object)"
                    if obj_cy - target_y > tol_y: return "Move Up (Center Object)"
                    if target_y - obj_cy > tol_y: return "Move Down (Center Object)"
                elif is_wide:
                    target_y = third_y1 if obj_cy < h // 2 else third_y2
                    if obj_cy - target_y > tol_y: return "Move Up (Align Object)"
                    if target_y - obj_cy > tol_y: return "Move Down (Align Object)"
                elif is_tall:
                    target_x = (w // 2) if is_portrait else (third_x1 if obj_cx < w // 2 else third_x2)
                    if obj_cx - target_x > tol_x: return "Move Left (Align Object)"
                    if target_x - obj_cx > tol_x: return "Move Right (Align Object)"
                else:
                    target_x = (w // 2) if is_portrait else (third_x1 if obj_cx < w // 2 else third_x2)
                    target_y = third_y1 if obj_cy < h // 2 else third_y2
                    if obj_cx - target_x > tol_x: return "Move Left (Align Object)"
                    if target_x - obj_cx > tol_x: return "Move Right (Align Object)"
                    if obj_cy - target_y > tol_y: return "Move Up (Align Object)"
                    if target_y - obj_cy > tol_y: return "Move Down (Align Object)"
                
                return "ALIGNED"
                
            elif len(objects) == 2:
                objects = sorted(objects, key=lambda o: o[0])
                o1_cx = objects[0][0] + objects[0][2] // 2
                if o1_cx - third_x1 > tol_x: return "Move Left (Balance Objects)"
                if third_x1 - o1_cx > tol_x: return "Move Right (Balance Objects)"
                
                o2_cx = objects[1][0] + objects[1][2] // 2
                if o2_cx - third_x2 > tol_x: return "Move Left (Balance Objects)"
                if third_x2 - o2_cx > tol_x: return "Move Right (Balance Objects)"
                
                return "ALIGNED"
            
        return ""

    def process_frame(self, frame_bgr_array, rotation_degrees=0):
        """
        Accepts a BGR numpy array from Kotlin/CameraX and applies native rotation.
        Returns a dictionary containing feedback and bounding boxes.
        """
        # Apply CameraX hardware rotation
        if rotation_degrees == 90:
            frame_bgr_array = cv2.rotate(frame_bgr_array, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_degrees == 180:
            frame_bgr_array = cv2.rotate(frame_bgr_array, cv2.ROTATE_180)
        elif rotation_degrees == 270:
            frame_bgr_array = cv2.rotate(frame_bgr_array, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = frame_bgr_array.shape[:2]

        # Dynamic Aspect-Preserving Tensor
        if w >= h:
            d_w = self.max_proc_dim
            d_h = int(self.max_proc_dim * (h / float(w)))
        else:
            d_h = self.max_proc_dim
            d_w = int(self.max_proc_dim * (w / float(h)))

        frame_total_area = d_w * d_h
        min_box_w, min_box_h = int(d_w * 0.10), int(d_h * 0.12)
        min_box_area = frame_total_area * 0.035
        max_box_area = frame_total_area * 0.50
        border_x, border_y = int(d_w * 0.08), int(d_h * 0.10)

        # ==========================================
        # PHASE 1: SEMANTIC DETECTION (FACE)
        # ==========================================
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr_array, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        face_detections = self.face_net.forward()
        
        valid_faces = []
        for i in range(face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence > 0.6:
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                fw, fh = endX - startX, endY - startY
                valid_faces.append([startX, startY, fw, fh])

        # ==========================================
        # PHASE 2: STRUCTURAL DETECTION (SALIENCY)
        # ==========================================
        small_frame = cv2.resize(frame_bgr_array, (d_w, d_h), interpolation=cv2.INTER_AREA)

        smoothed_frame = cv2.bilateralFilter(small_frame, 9, 75, 75)
        lab = cv2.cvtColor(smoothed_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_frame = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

        color_div_map = compute_color_divergence(smoothed_frame)

        focus_map = lap4_variance_of_laplacian(equalized_frame, window_size=9)
        focus_norm = cv2.normalize(focus_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        peak_sharpness = np.percentile(focus_norm, 98)
        _, focus_mask = cv2.threshold(focus_norm, int(peak_sharpness * 0.38), 255, cv2.THRESH_BINARY)
        
        kernel_focus = np.ones((7, 7), np.uint8)
        focus_mask = cv2.morphologyEx(cv2.dilate(focus_mask, kernel_focus, iterations=1), cv2.MORPH_CLOSE, kernel_focus, iterations=2)

        masked_frame = cv2.bitwise_and(equalized_frame, equalized_frame, mask=focus_mask)
        float_frame = np.float32(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)) / 255.0

        edges = self.edge_detector.detectEdges(float_frame)
        orimap = self.edge_detector.computeOrientation(edges)
        boxes, raw_scores = self.edge_boxes.getBoundingBoxes(self.edge_detector.edgesNms(edges, orimap), orimap)

        # ==========================================
        # PHASE 3: PRUNING & SCALING
        # ==========================================
        scaled_boxes = []
        scaled_scores = []

        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                bx, by, bw, bh = box
                box_area = bw * bh
                aspect_ratio = bw / float(max(1, bh))

                if bw < min_box_w or bh < min_box_h or box_area < min_box_area or box_area > max_box_area or aspect_ratio < 0.28 or aspect_ratio > 3.2:
                    continue

                cx, cy = min(max(bx + bw // 2, 0), d_w - 1), min(max(by + bh // 2, 0), d_h - 1)
                
                if cx < border_x or cx > (d_w - border_x) or cy < border_y or cy > (d_h - border_y) or focus_mask[cy, cx] == 0:
                    continue

                roi_color = color_div_map[by:by+bh, bx:bx+bw]
                mean_color_score = float(np.mean(roi_color)) if roi_color.size > 0 else 0.0
                area_factor = (box_area / frame_total_area) ** 0.25
                
                sx, sy = w / float(d_w), h / float(d_h)
                scaled_boxes.append([int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy)])
                scaled_scores.append(raw_scores[i].item() * (1.0 + 0.5 * mean_color_score) * area_factor)

        # Apply Semantic Override (Torso Drop-Zone Omission)
        semantic_saliency_boxes = []
        semantic_saliency_scores = []
        for i, box in enumerate(scaled_boxes):
            interference = False
            for fbox in valid_faces:
                if calculate_iou(box, fbox) > 0.05:
                    interference = True
                    break
                
                f_left, f_right = fbox[0], fbox[0] + fbox[2]
                b_left, b_right = box[0], box[0] + box[2]
                overlap = min(f_right, b_right) - max(f_left, b_left)
                
                if overlap > 0 and box[1] > (fbox[1] + fbox[3] * 0.5):
                    interference = True
                    break
                    
            if not interference:
                semantic_saliency_boxes.append(box)
                semantic_saliency_scores.append(scaled_scores[i])

        current_detections = []
        current_scores = []
        
        # Custom NMS to annihilate nested boxes
        if len(semantic_saliency_boxes) > 0:
            indices = custom_nms_with_containment(
                semantic_saliency_boxes, 
                semantic_saliency_scores, 
                iou_thresh=0.22, 
                containment_thresh=0.75 
            )
            for i in indices[:3]:
                current_detections.append(semantic_saliency_boxes[i])
                current_scores.append(semantic_saliency_scores[i])

        # ==========================================
        # PHASE 4: STATEFUL TRACKING & OUTPUT
        # ==========================================
        stable_objects = self.tracker.update(current_detections, current_scores, w, h)
        feedback_text = self._generate_feedback(valid_faces, stable_objects, w, h)
        is_aligned = feedback_text.startswith("ALIGNED")

        return {
            "feedback_text": feedback_text,
            "is_aligned": is_aligned,
            "faces": valid_faces,
            "objects": stable_objects
        }

    def process_frame_nv21(self, nv21_bytes: bytes, width: int, height: int, rotation_degrees: int) -> dict:
        """
        Fast JNI bridge. Converts the flat NV21 byte array from Android CameraX 
        into a BGR OpenCV numpy array, then applies hardware rotation.
        """
        yuv = np.frombuffer(nv21_bytes, dtype=np.uint8).reshape((height + height // 2, width))
        frame_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return self.process_frame(frame_bgr, rotation_degrees)