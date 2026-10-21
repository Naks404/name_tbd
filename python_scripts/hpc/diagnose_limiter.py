import os
import glob
import json
from collections import defaultdict

# ================= CONFIGURATION =================
RESULTS_DIR = "/home/l/lialiosk/VoxEffects/results_normalized"
# =================================================

def diagnose_limiter():
    jsonl_files = glob.glob(os.path.join(RESULTS_DIR, "*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {RESULTS_DIR}")
        return

    sample_files = jsonl_files[:100]

    metrics_sum_opt0 = defaultdict(float)
    metrics_sum_opt1 = defaultdict(float)
    count_opt0 = 0
    count_opt1 = 0

    print("Parsing JSONL files to isolate Limiter metrics...")

    for filepath in sample_files:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                limiter_opt = data["indices"]["limiter"]
                limiter_metrics = data["normalized_metrics"]["limiter"]

                for metric_name, val in limiter_metrics.items():
                    if limiter_opt == 0:
                        metrics_sum_opt0[metric_name] += val
                    else:
                        metrics_sum_opt1[metric_name] += val

                # Fixed: Unconditionally tally the option once per line
                if limiter_opt == 0:
                    count_opt0 += 1
                else:
                    count_opt1 += 1

    if count_opt0 == 0:
        print("Failed to parse metrics.")
        return

    print(f"\n--- Limiter Bias Diagnostic ---")
    print(f"Analyzed {len(sample_files)} files ({count_opt0 + count_opt1} total preset evaluations).")
    print(f"{'Metric Name':<15} | {'Opt 0 Avg':<12} | {'Opt 1 Avg':<12} | {'Difference'}")
    print("-" * 65)
    
    for metric_name in metrics_sum_opt0.keys():
        avg0 = metrics_sum_opt0[metric_name] / count_opt0
        avg1 = metrics_sum_opt1[metric_name] / count_opt1
        diff = avg0 - avg1
        
        flag = " <--- SEVERE BIAS" if abs(diff) > 0.25 else ""
        print(f"{metric_name:<15} | {avg0:<12.4f} | {avg1:<12.4f} | {diff:+.4f}{flag}")

if __name__ == "__main__":
    diagnose_limiter()
