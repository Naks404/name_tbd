import os
import json
import glob
import numpy as np
import multiprocessing
from scipy.stats import spearmanr, entropy
from collections import Counter

def process_file(filepath):
    """Processes a single normalized JSONL file and computes all ranking statistics."""
    # Matrix to hold scores: 2520 rows, 6 columns (DN, DRC, EQ, DS, RVB, LIM)
    S = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            scores = row.get("effect_scores", {})
            # Extract in exact order of cardinality n = (3, 15, 105, 315, 1260, 2520)
            S.append([
                scores.get("denoise", 0.0),
                scores.get("compressor", 0.0),
                scores.get("eq", 0.0),
                scores.get("deesser", 0.0),
                scores.get("reverb", 0.0),
                scores.get("limiter", 0.0)
            ])
            
    S = np.array(S, dtype=float)
    if len(S) == 0: return None

    # 1. Equal-weight L1
    Q_L1 = np.mean(S, axis=1)
    
    # 2. Equal-weight L2
    D_L2 = np.sqrt(np.mean((1.0 - S)**2, axis=1))
    Q_L2 = 1.0 - D_L2
    
    # 3. Log/cardinality-weighted score
    n_vals = np.array([3, 15, 105, 315, 1260, 2520])
    log_n = np.log2(n_vals)
    w = log_n / np.sum(log_n)
    Q_LOG = np.sum(S * w, axis=1)

    # Helper function to extract winner, second-best, margin, and full rank indices
    def get_stats(Q):
        ranks = np.argsort(Q)[::-1] # Descending order
        p_star = ranks[0]
        best_score = Q[p_star]
        second_best = Q[ranks[1]]
        margin = best_score - second_best
        return p_star, best_score, second_best, margin, ranks, Q

    p1, b1, sb1, m1, r1, q1_full = get_stats(Q_L1)
    p2, b2, sb2, m2, r2, q2_full = get_stats(Q_L2)
    p3, b3, sb3, m3, r3, q3_full = get_stats(Q_LOG)

    # Rank correlation (Spearman)
    corr_12, _ = spearmanr(Q_L1, Q_L2)
    corr_13, _ = spearmanr(Q_L1, Q_LOG)
    corr_23, _ = spearmanr(Q_L2, Q_LOG)

    # Top-k overlaps (using set intersections)
    overlaps = {}
    for k in [5, 10, 25, 50]:
        overlaps[f"12_{k}"] = len(set(r1[:k]).intersection(set(r2[:k]))) / k
        overlaps[f"13_{k}"] = len(set(r1[:k]).intersection(set(r3[:k]))) / k
        overlaps[f"23_{k}"] = len(set(r2[:k]).intersection(set(r3[:k]))) / k

    # Winner cross-evaluation matrix (How does each method's winner score across all methods?)
    cross_eval = {
        "p_L1": [q1_full[p1], q2_full[p1], q3_full[p1]],
        "p_L2": [q1_full[p2], q2_full[p2], q3_full[p2]],
        "p_LOG": [q1_full[p3], q2_full[p3], q3_full[p3]]
    }

    return {
        "p_star": (p1, p2, p3),
        "margins": (m1, m2, m3),
        "correlations": (corr_12, corr_13, corr_23),
        "overlaps": overlaps,
        "cross_eval": cross_eval
    }

if __name__ == "__main__":
    HOME_DIR = os.environ.get("HOME")
    input_dir = os.path.join(HOME_DIR, "VoxEffects", "results_normalized")
    files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Analyzing {len(files)} files using {num_cores} cores...\n")

    results = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        for res in pool.imap_unordered(process_file, files):
            if res: results.append(res)

    N = len(results)
    
    # ---------------- AGGREGATION ----------------
    
    # Winner Agreements
    agree_12 = sum(1 for r in results if r["p_star"][0] == r["p_star"][1]) / N
    agree_13 = sum(1 for r in results if r["p_star"][0] == r["p_star"][2]) / N
    agree_23 = sum(1 for r in results if r["p_star"][1] == r["p_star"][2]) / N
    agree_all = sum(1 for r in results if r["p_star"][0] == r["p_star"][1] == r["p_star"][2]) / N

    # Correlations
    c_12 = [r["correlations"][0] for r in results]
    c_13 = [r["correlations"][1] for r in results]
    c_23 = [r["correlations"][2] for r in results]

    # Margins
    m_1 = [r["margins"][0] for r in results]
    m_2 = [r["margins"][1] for r in results]
    m_3 = [r["margins"][2] for r in results]

    # Overlaps
    def get_overlap(key): return np.mean([r["overlaps"][key] for r in results])

    # Distributions
    p1_counts = Counter([r["p_star"][0] for r in results])
    p2_counts = Counter([r["p_star"][1] for r in results])
    p3_counts = Counter([r["p_star"][2] for r in results])

    def calc_entropy(counts):
        probs = np.array(list(counts.values())) / sum(counts.values())
        return entropy(probs, base=2)

    # Cross Eval (Average across all recordings)
    avg_cross_eval = np.mean([[r["cross_eval"]["p_L1"], r["cross_eval"]["p_L2"], r["cross_eval"]["p_LOG"]] for r in results], axis=0)

    # ---------------- REPORTING ----------------
    print(f"=== WINNER AGREEMENT ===")
    print(f"L1 vs L2  : {agree_12:.2%}")
    print(f"L1 vs LOG : {agree_13:.2%}")
    print(f"L2 vs LOG : {agree_23:.2%}")
    print(f"All Three : {agree_all:.2%}\n")

    print(f"=== RANK CORRELATION (SPEARMAN) ===")
    print(f"L1-L2  | Mean: {np.mean(c_12):.4f} | Median: {np.median(c_12):.4f}")
    print(f"L1-LOG | Mean: {np.mean(c_13):.4f} | Median: {np.median(c_13):.4f}")
    print(f"L2-LOG | Mean: {np.mean(c_23):.4f} | Median: {np.median(c_23):.4f}\n")

    print(f"=== TOP-K OVERLAP (MEAN PERCENTAGE) ===")
    print(f"k=5  | L1-L2: {get_overlap('12_5'):.1%} | L1-LOG: {get_overlap('13_5'):.1%} | L2-LOG: {get_overlap('23_5'):.1%}")
    print(f"k=10 | L1-L2: {get_overlap('12_10'):.1%} | L1-LOG: {get_overlap('13_10'):.1%} | L2-LOG: {get_overlap('23_10'):.1%}")
    print(f"k=25 | L1-L2: {get_overlap('12_25'):.1%} | L1-LOG: {get_overlap('13_25'):.1%} | L2-LOG: {get_overlap('23_25'):.1%}")
    print(f"k=50 | L1-L2: {get_overlap('12_50'):.1%} | L1-LOG: {get_overlap('13_50'):.1%} | L2-LOG: {get_overlap('23_50'):.1%}\n")

    print(f"=== MARGIN STATISTICS ===")
    def print_margin_stats(name, m):
        print(f"{name:^5} | {np.mean(m):.4f} | {np.median(m):.4f} | {np.std(m):.4f} | {np.min(m):.4f} | {np.percentile(m, 25):.4f} | {np.percentile(m, 75):.4f} | {np.max(m):.4f}")
    
    print(f"{'Method':^5} | {'Mean':^6} | {'Median':^6} | {'Std':^6} | {'Min':^6} | {'25th':^6} | {'75th':^6} | {'Max':^6}")
    print("-" * 75)
    print_margin_stats("L1", m_1)
    print_margin_stats("L2", m_2)
    print_margin_stats("LOG", m_3)
    print("\n")

    print(f"=== SELECTED-LABEL DISTRIBUTION ===")
    def print_dist(name, counts):
        print(f"-- {name} --")
        print(f"Unique Presets: {len(counts)} / 2520")
        print(f"Label Entropy : {calc_entropy(counts):.4f} bits")
        top10 = [(k, v) for k, v in counts.most_common(10)]
        print(f"Top 10 labels : {top10}\n")
        
    print_dist("L1", p1_counts)
    print_dist("L2", p2_counts)
    print_dist("LOG", p3_counts)

    print(f"=== WINNER CROSS-EVALUATION MATRIX ===")
    print(f"{'Evaluating Winner from':<25} | {'L1 Score':<10} | {'L2 Score':<10} | {'LOG Score':<10}")
    print("-" * 65)
    print(f"{'L1 Winner ($p_{L1}^*$)':<25} | {avg_cross_eval[0][0]:<10.4f} | {avg_cross_eval[0][1]:<10.4f} | {avg_cross_eval[0][2]:<10.4f}")
    print(f"{'L2 Winner ($p_{L2}^*$)':<25} | {avg_cross_eval[1][0]:<10.4f} | {avg_cross_eval[1][1]:<10.4f} | {avg_cross_eval[1][2]:<10.4f}")
    print(f"{'LOG Winner ($p_{LOG}^*$)':<25} | {avg_cross_eval[2][0]:<10.4f} | {avg_cross_eval[2][1]:<10.4f} | {avg_cross_eval[2][2]:<10.4f}")