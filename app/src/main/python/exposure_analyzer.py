import cv2
import numpy as np
import collections
import math

class ExposureAnalyzer:
    def __init__(self):
        # --- THRESHOLDS ---
        self.TARGET_LUM = 115.0
        self.EV_TOLERANCE = 25.0
        self.HARD_CLIP_RATIO = 0.10
        self.MOTION_THRESH = 5.0      
        self.SNR_THRESH = 15.0        
        self.DEPTH_FLAT_THRESH = 2.0  
        
        self.WB_SLOPE = 1.173384
        self.WB_INTERCEPT = -0.215403

        # --- TEMPORAL SMOOTHING BUFFERS ---
        self.lum_history = collections.deque(maxlen=15)
        self.speed_history = collections.deque(maxlen=15)
        self.snr_history = collections.deque(maxlen=3)    
        self.wb_history = collections.deque(maxlen=3)
        self.depth_history = collections.deque(maxlen=3)

        # --- LUCAS-KANADE STATE ---
        self.prev_gray = None
        self.prev_pts = None
        self.feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.frame_count = 0

        # --- CACHED NATIVE METRICS ---
        self.current_metrics = {
            "luminance": 115.0, "is_clipped_blacks": False, "is_clipped_whites": False,
            "motion_speed": 0.0, "snr": 20.0, "wb_ratio": 1.0, "depth_ratio": 5.0
        }

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        self.frame_count += 1
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # TIER 1: Continuous Evaluation
        self._evaluate_exposure(gray)
        self._evaluate_motion(gray)

        # TIER 2: 1-Second Cycle (~30fps)
        if self.frame_count % 30 == 0:
            self._evaluate_noise(gray)

        # TIER 3: 3-Second Cycle
        if self.frame_count % 90 == 0:
            self._evaluate_white_balance(frame_bgr)
            self._evaluate_depth(gray)
            self.frame_count = 0

        return self._resolve_state()

    def _evaluate_exposure(self, gray: np.ndarray):
        h, w = gray.shape
        cy, cx = h // 4, w // 4
        center_roi = gray[cy:h-cy, cx:w-cx]
        
        lum = float((cv2.mean(center_roi)[0] * 0.70) + (cv2.mean(gray)[0] * 0.30))
        self.lum_history.append(lum)
        self.current_metrics["luminance"] = float(sum(self.lum_history) / len(self.lum_history))

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total_pixels = float(h * w)
        black_ratio = float(np.sum(hist[:5])) / total_pixels
        white_ratio = float(np.sum(hist[251:])) / total_pixels

        self.current_metrics["is_clipped_blacks"] = bool(black_ratio > self.HARD_CLIP_RATIO and black_ratio > white_ratio)
        self.current_metrics["is_clipped_whites"] = bool(white_ratio > self.HARD_CLIP_RATIO and white_ratio > black_ratio)

    def _evaluate_motion(self, gray: np.ndarray):
        median_speed = 0.0

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 15 or self.frame_count == 1:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray.copy()
            return

        new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        if new_pts is not None and status is not None:
            good_new = new_pts[status == 1]
            good_old = self.prev_pts[status == 1]

            if len(good_new) > 0:
                dx = good_new[:, 0] - good_old[:, 0]
                dy = good_new[:, 1] - good_old[:, 1]
                distances = np.sqrt(dx**2 + dy**2)
                median_speed = float(np.percentile(distances, 90))

        self.speed_history.append(median_speed)
        self.current_metrics["motion_speed"] = float(sum(self.speed_history) / len(self.speed_history))

        self.prev_gray = gray.copy()
        self.prev_pts = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

    def _evaluate_noise(self, gray: np.ndarray):
        gray_f32 = gray.astype(np.float32)
        Gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
        G_inner = (np.abs(Gx) + np.abs(Gy))[1:-1, 1:-1]

        N_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        abs_lap_inner = np.abs(cv2.filter2D(gray_f32, cv2.CV_32F, N_kernel))[1:-1, 1:-1]

        g_max = float(G_inner.max())
        if g_max < 1e-5: 
            return

        hist = cv2.calcHist([G_inner], [0], None, [2048], [0, g_max])
        cumsum = np.cumsum(hist)
        bin_idx = np.searchsorted(cumsum, (10 / 100.0) * G_inner.size)
        G_th = (bin_idx / 2047) * g_max

        valid_noise_pixels = abs_lap_inner[(G_inner <= G_th)]
        if valid_noise_pixels.size == 0: 
            return

        sigma = float(math.sqrt(math.pi / 2.0) * (1.0 / 6.0) * np.mean(valid_noise_pixels))
        sigma = max(sigma, 0.001)

        raw_snr = float(self.current_metrics["luminance"] / sigma)
        self.snr_history.append(raw_snr)
        self.current_metrics["snr"] = float(sum(self.snr_history) / len(self.snr_history))

    def _evaluate_white_balance(self, frame_bgr: np.ndarray):
        small = cv2.resize(frame_bgr, (320, 240)).astype(np.float32)
        grad_x = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        mag_sq = grad_x**2 + grad_y**2

        edge_mask = mag_sq > (20.0**2)
        valid_grad = np.sqrt(mag_sq) * edge_mask

        b_edge = float(np.sum(valid_grad[:, :, 0]))
        r_edge = float(np.sum(valid_grad[:, :, 2]))
        raw_ratio = r_edge / (b_edge + 1e-6)

        ratio = float(max(0.1, (self.WB_SLOPE * raw_ratio) + self.WB_INTERCEPT))
        self.wb_history.append(ratio)
        self.current_metrics["wb_ratio"] = float(sum(self.wb_history) / len(self.wb_history))

    def _evaluate_depth(self, gray: np.ndarray):
        gray_f64 = np.ascontiguousarray(gray.astype(np.float64))
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        
        lap = cv2.filter2D(gray_f64, cv2.CV_64F, lap_kernel, borderType=cv2.BORDER_REFLECT101)
        area = 81.0 
        
        sum_lap = cv2.boxFilter(lap, cv2.CV_64F, ksize=(9, 9), normalize=False, borderType=cv2.BORDER_REFLECT101)
        sum_lap2 = cv2.boxFilter(lap*lap, cv2.CV_64F, ksize=(9, 9), normalize=False, borderType=cv2.BORDER_REFLECT101)
        
        response_map = np.maximum(sum_lap2 - (sum_lap * sum_lap) / area, 0.0)
        
        p90 = float(np.percentile(response_map, 90))
        p10 = float(np.percentile(response_map, 10))
        
        ratio = float(p90 / (p10 + 1e-6))
        self.depth_history.append(ratio)
        self.current_metrics["depth_ratio"] = float(sum(self.depth_history) / len(self.depth_history))

    def _resolve_state(self) -> dict:
        m = self.current_metrics
        
        response = {
            "primary_action": "", 
            "primary_color": "",
            "secondary_action": "", 
            "secondary_color": ""
        }

        # 1. Catastrophic Clipping Override
        if m["is_clipped_whites"]:
            response["primary_action"] = "SEVERE CLIPPING. Reduce Light Immediately."
            response["primary_color"] = "RED"
        elif m["is_clipped_blacks"]:
            response["primary_action"] = "SEVERE CLIPPING. Add Light Immediately."
            response["primary_color"] = "RED"
        else:
            is_high_motion = m["motion_speed"] > self.MOTION_THRESH
            is_high_noise = m["snr"] < self.SNR_THRESH
            is_underexposed = m["luminance"] < (self.TARGET_LUM - self.EV_TOLERANCE)
            is_overexposed = m["luminance"] > (self.TARGET_LUM + self.EV_TOLERANCE)

            # 2. Primary Exposure Resolution
            if is_underexposed:
                if is_high_motion:
                    response["primary_action"] = "Too dark & moving. Raise ISO."
                elif is_high_noise:
                    response["primary_action"] = "Grainy. Hold steady & Lower Shutter."
                else:
                    response["primary_action"] = "Too dark. Raise ISO or Lower Shutter."
                response["primary_color"] = "AMBER"
            elif is_overexposed:
                response["primary_action"] = "Too bright. Lower ISO or Raise Shutter."
                response["primary_color"] = "AMBER"
            else:
                if is_high_noise and not is_high_motion:
                    response["primary_action"] = "Hidden grain. Lower ISO & Lower Shutter."
                    response["primary_color"] = "AMBER"
                else:
                    response["primary_action"] = "Exposure Locked."
                    response["primary_color"] = "GREEN"

        # 3. Secondary Aesthetic Resolution
        if m["depth_ratio"] < self.DEPTH_FLAT_THRESH:
            response["secondary_action"] = "Flat depth. Move closer to subject."
            response["secondary_color"] = "CYAN"
        elif m["wb_ratio"] > 1.08:
            response["secondary_action"] = "Too Warm. Add Cool/Blue."
            response["secondary_color"] = "CYAN"
        elif m["wb_ratio"] < 0.92:
            response["secondary_action"] = "Too Cold. Add Warm/Orange."
            response["secondary_color"] = "CYAN"

        # --- TEMPORARY CALIBRATION INJECTION ---
        # Inject the raw values into the secondary_action string so they appear on screen
        stats_str = f"[SPD: {m['motion_speed']:.1f} | SNR: {m['snr']:.1f} | WB: {m['wb_ratio']:.2f} | DP: {m['depth_ratio']:.1f}]"
        
        if response["secondary_action"]:
            response["secondary_action"] = f"{stats_str}\n{response['secondary_action']}"
        else:
            response["secondary_action"] = stats_str
            # Default to white text if there is no active secondary warning
            response["secondary_color"] = "WHITE" 

        return response

    def analyze_frame_nv21(self, nv21_bytes: bytes, width: int, height: int) -> dict:
        yuv = np.frombuffer(nv21_bytes, dtype=np.uint8).reshape((height + height // 2, width))
        frame_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return self.analyze_frame(frame_bgr)