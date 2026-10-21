import os
import json
import glob

def verify_normalization_bounds():
    home_dir = os.environ.get("HOME")
    target_dir = os.path.join(home_dir, "VoxEffects", "results_normalized")
    
    if not os.path.exists(target_dir):
        print(f"ERROR: Directory not found at {target_dir}")
        return

    files = glob.glob(os.path.join(target_dir, "*.jsonl"))
    print(f"Scanning {len(files)} files for bounds violations...\n")
    
    violations = []
    global_ranges = {}

    for file_path in files:
        filename = os.path.basename(file_path)
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    norm_metrics = data.get("normalized_metrics", {})
                    
                    for effect, metrics in norm_metrics.items():
                        for metric_name, val in metrics.items():
                            if val is not None:
                                # Track global min/max
                                if metric_name not in global_ranges:
                                    global_ranges[metric_name] = {"min": val, "max": val}
                                else:
                                    global_ranges[metric_name]["min"] = min(global_ranges[metric_name]["min"], val)
                                    global_ranges[metric_name]["max"] = max(global_ranges[metric_name]["max"], val)
                                
                                # Check bounds with microscopic floating-point tolerance
                                if val < -1e-7 or val > 1.0000001:
                                    violations.append(f"{filename} (Row {line_num}) | {metric_name} = {val}")
                                    
                except json.JSONDecodeError:
                    print(f"Corrupt JSON encountered in {filename} at line {line_num}")

    print("=" * 50)
    print(" NORMALIZATION BOUNDS REPORT")
    print("=" * 50)
    
    if violations:
        print(f"[!] FAILED: Found {len(violations)} values outside the [0, 1] range.\n")
        for v in violations[:20]:
            print(f"  - {v}")
        if len(violations) > 20:
            print(f"  ... plus {len(violations) - 20} additional violations.")
    else:
        print("[+] SUCCESS: 100% of values are strictly bounded within [0, 1].\n")

    print("OBSERVED METRIC RANGES:")
    for metric in sorted(global_ranges.keys()):
        m_min = global_ranges[metric]['min']
        m_max = global_ranges[metric]['max']
        print(f"  {metric:<10}: [{m_min:.6f}, {m_max:.6f}]")

if __name__ == "__main__":
    verify_normalization_bounds()