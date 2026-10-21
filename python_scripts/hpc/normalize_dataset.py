import os
import json
import numpy as np
import multiprocessing
import traceback
import glob

def normalize_array(arr, invert=True):
    """
    Applies purely relative robust normalization.
    Uses Unique Counts, Median Absolute Deviation (MAD), and Standard Z-Scores 
    to dynamically identify and clamp outliers without absolute thresholds.
    """
    valid_idx = [i for i, v in enumerate(arr) if v is not None]
    if not valid_idx:
        return [None] * len(arr)

    valid_vals = np.array([arr[i] for i in valid_idx], dtype=float)
    unique_vals = np.unique(valid_vals)

    # 1. Zero Variance
    if len(unique_vals) == 1:
        return [1.0 if i in valid_idx else None for i in range(len(arr))]

    # 2. Binary States (e.g., D_N microscopic shifts)
    if len(unique_vals) == 2:
        clipped_vals = valid_vals # No outliers possible in binary states
        
    # 3. Continuous Distributions (Statistical Outlier Handling)
    else:
        median = np.median(valid_vals)
        # Median Absolute Deviation (MAD)
        mad = np.median(np.abs(valid_vals - median))

        if mad > 1e-12:
            # Iglewicz and Hoaglin (1993) Modified Z-Score
            # 1.4826 approximates standard deviation for normal distributions
            # 3.5 is the statistically optimal cutoff for outlier detection
            pseudo_std = mad * 1.4826
            lower_bound = median - 3.5 * pseudo_std
            upper_bound = median + 3.5 * pseudo_std
        else:
            # Zero-Inflated Fallback (e.g. P_ceiling where >50% of data is exactly 0.0)
            # MAD is 0, so we use 4-Sigma rule to capture the positive tail relative to its own variance
            mean = np.mean(valid_vals)
            std = np.std(valid_vals)
            lower_bound = mean - 4.0 * std
            upper_bound = mean + 4.0 * std

        clipped_vals = np.clip(valid_vals, lower_bound, upper_bound)

    # 4. Final Min-Max Scaling
    e_min = np.min(clipped_vals)
    e_max = np.max(clipped_vals)
    delta = e_max - e_min

    # Floating point math safeguard
    if delta < 1e-12:
        return [1.0 if i in valid_idx else None for i in range(len(arr))]

    norm_vals = (clipped_vals - e_min) / delta
    
    # Clean rounding artifacts just in case
    norm_vals = np.clip(norm_vals, 0.0, 1.0)
    
    if invert:
        norm_vals = 1.0 - norm_vals

    # 5. Reconstruct array
    out_arr = [None] * len(arr)
    for i, val in zip(valid_idx, norm_vals):
        out_arr[i] = float(val)

    return out_arr

def safe_mean(*args):
    """Calculates the arithmetic mean while dynamically ignoring None values."""
    valid = [a for a in args if a is not None]
    return sum(valid) / len(valid) if valid else 0.0

def process_single_file(filepaths):
    in_path, out_path = filepaths
    base_name = os.path.basename(in_path)

    if os.path.exists(out_path):
        return f"Skipped (already exists): {base_name}"

    try:
        rows = []
        with open(in_path, 'r') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        if not rows:
            return f"Empty file skipped: {base_name}"

        metrics = {
            "D_N": [], "D_env": [], "A_quiet": [],
            "D_low": [], "D_high": [], "D_CF": [], "D_LRA": [], "P_pump": [],
            "D_SB": [],
            "D_sib": [], "D_HF": [], "S_select": [],
            "D_R": [],
            "D_L": [], "D_TP": [], "P_ceiling": [], "GR_mean": []
        }

        for row in rows:
            rm = row["raw_metrics"]
            metrics["D_N"].append(rm["denoise"].get("D_N"))
            metrics["D_env"].append(rm["denoise"].get("D_env"))
            metrics["A_quiet"].append(rm["denoise"].get("A_quiet"))
            
            metrics["D_low"].append(rm["compressor"].get("D_low"))
            metrics["D_high"].append(rm["compressor"].get("D_high"))
            metrics["D_CF"].append(rm["compressor"].get("D_CF"))
            metrics["D_LRA"].append(rm["compressor"].get("D_LRA"))
            metrics["P_pump"].append(rm["compressor"].get("P_pump"))
            
            metrics["D_SB"].append(rm["eq"].get("D_SB"))
            
            metrics["D_sib"].append(rm["deesser"].get("D_sib"))
            metrics["D_HF"].append(rm["deesser"].get("D_HF"))
            metrics["S_select"].append(rm["deesser"].get("S_select"))
            
            metrics["D_R"].append(rm["reverb"].get("D_R"))
            
            metrics["D_L"].append(rm["limiter"].get("D_L"))
            metrics["D_TP"].append(rm["limiter"].get("D_TP"))
            metrics["P_ceiling"].append(rm["limiter"].get("P_ceiling"))
            
            gr = rm["limiter"].get("GR_mean")
            metrics["GR_mean"].append(abs(gr) if gr is not None else None)

        normed = {}
        
        # Inverted Metrics (Errors, penalties, distances)
        for key in ["D_N", "D_env", "A_quiet", "D_low", "D_high", "D_CF", "D_LRA", "P_pump", "D_SB", "D_sib", "D_HF", "D_R", "D_L", "D_TP", "P_ceiling", "GR_mean"]:
            normed[key] = normalize_array(metrics[key], invert=True)
            
        # Direct Metrics (Ratios where higher is better)
        for key in ["S_select"]:
            normed[key] = normalize_array(metrics[key], invert=False)

        with open(out_path, 'w') as f:
            for i, row in enumerate(rows):
                row["normalized_metrics"] = {
                    "denoise": { "D_N": normed["D_N"][i], "D_env": normed["D_env"][i], "A_quiet": normed["A_quiet"][i] },
                    "compressor": { "D_low": normed["D_low"][i], "D_high": normed["D_high"][i], "D_CF": normed["D_CF"][i], "D_LRA": normed["D_LRA"][i], "P_pump": normed["P_pump"][i] },
                    "eq": { "D_SB": normed["D_SB"][i] },
                    "deesser": { "D_sib": normed["D_sib"][i], "D_HF": normed["D_HF"][i], "S_select": normed["S_select"][i] },
                    "reverb": { "D_R": normed["D_R"][i] },
                    "limiter": { "D_L": normed["D_L"][i], "D_TP": normed["D_TP"][i], "P_ceiling": normed["P_ceiling"][i], "GR_mean": normed["GR_mean"][i] }
                }
                
                row["effect_scores"] = {
                    "denoise": safe_mean(normed["D_N"][i], normed["D_env"][i], normed["A_quiet"][i]),
                    "compressor": safe_mean(normed["D_low"][i], normed["D_high"][i], normed["D_CF"][i], normed["D_LRA"][i], normed["P_pump"][i]),
                    "eq": safe_mean(normed["D_SB"][i]),
                    "deesser": safe_mean(normed["D_sib"][i], normed["D_HF"][i], normed["S_select"][i]),
                    "reverb": safe_mean(normed["D_R"][i]),
                    "limiter": safe_mean(normed["D_L"][i], normed["D_TP"][i], normed["P_ceiling"][i], normed["GR_mean"][i])
                }
                
                f.write(json.dumps(row) + '\n')

        return f"SUCCESS: {base_name}"
        
    except Exception as e:
        return f"ERROR on {base_name}:\n{traceback.format_exc()}"

if __name__ == "__main__":
    HOME_DIR = os.environ.get("HOME")
    input_dir = os.path.join(HOME_DIR, "VoxEffects", "results")
    output_dir = os.path.join(HOME_DIR, "VoxEffects", "results_normalized")

    os.makedirs(output_dir, exist_ok=True)

    input_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(input_files)} files to normalize.")

    file_pairs = [(f, os.path.join(output_dir, os.path.basename(f))) for f in input_files]

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting relative statistical normalization on {num_cores} cores...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(process_single_file, file_pairs):
            print(result)

    print("\n--- NORMALIZATION COMPLETE ---")