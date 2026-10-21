import os
import glob
import json
import numpy as np
import torch
from pathlib import Path

# Define the 6 modules and their respective number of options
MODULES = {
    'denoise': 3,
    'compressor': 5,
    'eq': 7,
    'deesser': 3,
    'reverb': 4,
    'limiter': 2
}

def compute_temperature_softmax(means, temp=0.1):
    """
    Converts averaged scores into probabilities that sum to 1.0.
    Temperature T=0.1 sharpens the contrast so bad options drop near 0%.
    """
    t = torch.tensor(means, dtype=torch.float32)
    # Subtract the maximum value for numerical stability before dividing by temperature
    scaled = (t - torch.max(t)) / temp
    return torch.nn.functional.softmax(scaled, dim=0)

def extract_targets():
    # Set up exact cluster paths
    input_dir = "/home/l/lialiosk/VoxEffects/results_normalized"
    output_dir = "/home/l/lialiosk/VoxEffects/training_targets"
    os.makedirs(output_dir, exist_ok=True)

    # Find all normalized JSONL files
    jsonl_files = glob.glob(os.path.join(input_dir, "*_metrics.jsonl"))
    
    if not jsonl_files:
        print(f"No files found in {input_dir}. Please check the path.")
        return

    print(f"Found {len(jsonl_files)} files. Starting extraction...")

    for filepath in jsonl_files:
        # Extract the base recording ID (e.g., 'f1_script1_clean')
        recording_id = Path(filepath).stem.replace("_metrics", "")
        
        # Temporary storage for the 2520 presets inside this specific file
        preset_indices = {mod: [] for mod in MODULES.keys()}
        l2_scores = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # 1. Extract the 6 normalized effect scores
                scores = data['effect_scores']
                s_array = np.array([
                    scores['denoise'], scores['compressor'], scores['eq'],
                    scores['deesser'], scores['reverb'], scores['limiter']
                ], dtype=np.float32)
                
                # 2. Calculate the L2 Quality Score for this specific combination
                # Formula: D_L2 = sqrt( mean( (1.0 - s)^2 ) )
                d_l2 = np.sqrt(np.mean((1.0 - s_array)**2))
                q_l2 = 1.0 - d_l2
                l2_scores.append(q_l2)
                
                # 3. Store which hardware indices (0, 1, 2, etc.) were active
                idx_data = data['indices']
                for mod in MODULES.keys():
                    preset_indices[mod].append(idx_data[mod])
        
        l2_scores = np.array(l2_scores)
        
        # 4. Marginalize (Group and Average)
        soft_targets = {}
        for mod, num_options in MODULES.items():
            mod_indices = np.array(preset_indices[mod])
            mod_means = np.zeros(num_options, dtype=np.float32)
            
            for opt_idx in range(num_options):
                # Find all rows where this specific option was used
                mask = (mod_indices == opt_idx)
                # Calculate the average L2 score of all combinations containing this option
                mod_means[opt_idx] = np.mean(l2_scores[mask])
            
            # 5. Convert the averaged scores to probabilities using Softmax
            soft_targets[mod] = compute_temperature_softmax(mod_means, temp=0.1)

        # 6. Save the PyTorch dictionary bundle (.pt) for this recording
        out_path = os.path.join(output_dir, f"{recording_id}_targets.pt")
        torch.save(soft_targets, out_path)
        
    print(f"Extraction complete! Soft targets saved to {output_dir}")

if __name__ == "__main__":
    extract_targets()