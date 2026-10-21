import os
import random
import glob
import json
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
import librosa
import numpy as np

# ================= Configuration =================
RAW_AUDIO_DIR = "/scratch/l/lialiosk/daps"
RESULTS_DIR = "/home/l/lialiosk/VoxEffects/results_normalized"

# CRITICAL: Now correctly routed to $SCRATCH to avoid quota limits
OUTPUT_DIR = "/scratch/l/lialiosk/VoxEffects_ml_dataset"

# Strict directory exclusions
EXCLUDED_DIRS = {"cleanraw", "produced"}

SLICE_LENGTH_SEC = 20
SAMPLE_RATE = 44100
N_FFT = 1024
HOP_LENGTH = 512
TEMPERATURE = 0.1
# =================================================

def extract_hd_linear_spectrogram(y, sr):
    """
    Extracts a 513-bin Linear Spectrogram, keeping high-frequency noise intact.
    Input shape: Audio array
    Output shape: [1, 513, Time_Frames]
    """
    # Compute Short-Time Fourier Transform (1024 FFT gives 513 frequency bins)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Get the raw magnitude (amplitude)
    magnitude = np.abs(D)
    
    # Convert to Decibel scale so quiet background static visually "pops"
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Convert to PyTorch tensor and add the [Channel] dimension
    tensor_spec = torch.tensor(log_spectrogram, dtype=torch.float32).unsqueeze(0)
    
    return tensor_spec

def generate_soft_targets(jsonl_filepath):
    """
    Parses the evaluation JSONL, corrects the Limiter bias, 
    and generates temperature-scaled soft targets.
    """
    module_scores = {
        'denoise': defaultdict(list),
        'compressor': defaultdict(list),
        'eq': defaultdict(list),
        'deesser': defaultdict(list),
        'reverb': defaultdict(list),
        'limiter': defaultdict(list)
    }

    with open(jsonl_filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            indices = data['indices']
            
            # --- STRATEGY 2: Limiter Bias Intervention ---
            # Drop GR_mean and P_ceiling. Calculate based strictly on Loudness and True Peak.
            lim_metrics = data['normalized_metrics']['limiter']
            fair_limiter_score = (lim_metrics['D_L'] + lim_metrics['D_TP']) / 2.0
            
            # Override the biased score with our fair score
            data['effect_scores']['limiter'] = fair_limiter_score

            # Append to our marginalization tallies
            for mod in module_scores.keys():
                preset_idx = indices[mod]
                module_scores[mod][preset_idx].append(data['effect_scores'][mod])

    # Convert to Soft-Label Target Tensors (Temperature Scaled)
    soft_targets = {}
    for mod, idx_scores in module_scores.items():
        num_options = max(idx_scores.keys()) + 1
        avg_scores = torch.zeros(num_options, dtype=torch.float32)
        
        for idx, scores in idx_scores.items():
            avg_scores[idx] = sum(scores) / len(scores)
        
        # Apply Softmax with temperature to create smooth probability curves
        scaled_scores = avg_scores / TEMPERATURE
        soft_targets[mod] = F.softmax(scaled_scores, dim=0)

    return soft_targets

def split_speakers(audio_files, train_ratio=0.8):
    """Extracts speaker ID and performs a gender-stratified 80/20 split."""
    speakers = set()
    file_map = {}
    
    for filepath in audio_files:
        filename = Path(filepath).stem
        speaker_id = filename.split('_')[0]
        speakers.add(speaker_id)
        
        if speaker_id not in file_map:
            file_map[speaker_id] = []
        file_map[speaker_id].append(filepath)
        
    female_speakers = sorted([s for s in speakers if s.startswith('f')])
    male_speakers = sorted([s for s in speakers if s.startswith('m')])
    
    random.shuffle(female_speakers)
    random.shuffle(male_speakers)
    
    f_split = int(len(female_speakers) * train_ratio)
    m_split = int(len(male_speakers) * train_ratio)
    
    train_speakers = set(female_speakers[:f_split] + male_speakers[:m_split])
    val_speakers = set(female_speakers[f_split:] + male_speakers[m_split:])
    
    print(f"Train Speakers ({len(train_speakers)}): {sorted(list(train_speakers))}")
    print(f"Val Speakers ({len(val_speakers)}): {sorted(list(val_speakers))}")
    
    return train_speakers, val_speakers, file_map

def get_valid_audio_files():
    """Recursively fetches valid .wav files while applying strict exclusion rules."""
    valid_files = []
    
    for path in Path(RAW_AUDIO_DIR).rglob("*.wav"):
        if path.name.startswith("._"):
            continue
            
        if any(excluded_dir in path.parts for excluded_dir in EXCLUDED_DIRS):
            continue
            
        valid_files.append(str(path))
        
    return sorted(valid_files)

def process_and_slice():
    os.makedirs(os.path.join(OUTPUT_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'val'), exist_ok=True)

    audio_files = get_valid_audio_files()
    if not audio_files:
        print(f"No valid .wav files found in {RAW_AUDIO_DIR} after applying filters.")
        return

    print(f"Found {len(audio_files)} valid audio files to process.")

    train_speakers, _, file_map = split_speakers(audio_files)
    
    samples_per_slice = SLICE_LENGTH_SEC * SAMPLE_RATE
    half_slice = samples_per_slice // 2

    for speaker_id, files in file_map.items():
        split_folder = 'train' if speaker_id in train_speakers else 'val'
        
        for filepath in files:
            base_name = Path(filepath).stem
            
            # --- STRATEGY 2: Dynamic JSONL Processing ---
            jsonl_file = os.path.join(RESULTS_DIR, f"{base_name}_metrics.jsonl")
            
            if not os.path.exists(jsonl_file):
                print(f"Skipping {base_name}, no matching JSONL found in {RESULTS_DIR}")
                continue
                
            # Generate the corrected targets on the fly
            soft_targets = generate_soft_targets(jsonl_file)
            
            y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            total_samples = len(y)
            
            if total_samples < samples_per_slice:
                print(f"Warning: {base_name} is shorter than 20 seconds total. Skipping.")
                continue

            num_full_slices = total_samples // samples_per_slice
            residual_samples = total_samples % samples_per_slice

            for i in range(num_full_slices):
                start_sample = i * samples_per_slice
                end_sample = start_sample + samples_per_slice
                y_slice = y[start_sample:end_sample]
                
                # Use HD Linear Spectrograms
                spectrogram_tensor = extract_hd_linear_spectrogram(y_slice, sr)
                
                data_bundle = {
                    'spectrogram': spectrogram_tensor,
                    'targets': soft_targets
                }
                
                out_name = f"{base_name}_slice{i:03d}.pt"
                torch.save(data_bundle, os.path.join(OUTPUT_DIR, split_folder, out_name))
                
            if residual_samples >= half_slice:
                y_slice = y[-samples_per_slice:]
                
                spectrogram_tensor = extract_hd_linear_spectrogram(y_slice, sr)
                data_bundle = {
                    'spectrogram': spectrogram_tensor,
                    'targets': soft_targets
                }
                
                out_name = f"{base_name}_slice_residual.pt"
                torch.save(data_bundle, os.path.join(OUTPUT_DIR, split_folder, out_name))
                
    print(f"\nProcessing complete! HD Tensor dataset successfully prepared and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    random.seed(42)
    process_and_slice()