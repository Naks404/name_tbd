# Run this cell first in Colab to ensure dependencies are installed:
# !pip install pedalboard librosa soundfile pyloudnorm srmrpy numpy scipy

import pedalboard
import os
import json
import glob
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
import pyloudnorm as pyln
from srmrpy import srmr
import math
import multiprocessing
import traceback

EPS = 1e-12

# ==============================================================================
# 1. OPTIMIZED FEATURE EXTRACTION (LAZY EVALUATION)
# ==============================================================================

class AudioFeatures:
    def __init__(self, audio, sr, meter, speech_mask, frame_ms=20, hop_ms=10):
        self.audio = audio
        self.sr = sr
        self.meter = meter

        frame_len = int(sr * frame_ms / 1000)
        hop_len = int(sr * hop_ms / 1000)

        shape = (1 + (len(audio) - frame_len) // hop_len, frame_len)
        strides = (audio.strides[0] * hop_len, audio.strides[0])
        self.frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

        rms = np.sqrt(np.mean(self.frames**2, axis=1))
        # Clip silence to -100 dBFS to prevent epsilon math explosions from closed noise gates
        self.energies = np.clip(20 * np.log10(rms + EPS), -100, None)

        # Safely align the provided fixed mask length
        if len(speech_mask) > len(self.frames):
            self.speech_mask = speech_mask[:len(self.frames)]
        elif len(speech_mask) < len(self.frames):
            self.speech_mask = np.pad(speech_mask, (0, len(self.frames) - len(speech_mask)), constant_values=False)
        else:
            self.speech_mask = speech_mask

        # Fallback to pure statistical split only if the track is completely broken/silent
        if not np.any(self.speech_mask) or not np.any(~self.speech_mask):
            self.speech_mask = self.energies > np.percentile(self.energies, 50)

        self.speech_energies = self.energies[self.speech_mask]

        self._fft_speech = None
        self._freqs = None
        self._eq_bands = None
        self._deesser_bands = None
        self._macroscopic = {}

    @property
    def fft_data(self):
        if self._fft_speech is None:
            # Generate FFT for ALL frames to maintain identical array sizes
            self._fft_speech = np.abs(np.fft.rfft(self.frames, axis=1))**2
            self._freqs = np.fft.rfftfreq(self.frames.shape[1], 1/self.sr)
        return self._fft_speech, self._freqs

    @property
    def eq_bands(self):
        if self._eq_bands is None:
            power_spectrum, freqs = self.fft_data
            bands = [(20,60), (60,250), (250,500), (500,2000), (2000,4000),
                     (4000,6000), (6000,10000), (10000,16000), (16000,20000)]

            E_k = np.zeros(9)
            # Filter to speech frames here so EQ only evaluates active voice
            speech_power = power_spectrum[self.speech_mask]
            if len(speech_power) == 0:
                speech_power = power_spectrum
            mean_power = np.mean(speech_power, axis=0)

            for i, (f_min, f_max) in enumerate(bands):
                idx = (freqs >= f_min) & (freqs < f_max)
                E_k[i] = np.sum(mean_power[idx])

            self._eq_bands = E_k
        return self._eq_bands

    @property
    def deesser_bands(self):
        if self._deesser_bands is None:
            power_spectrum, freqs = self.fft_data
            mid_idx = (freqs >= 1000) & (freqs < 4000)
            hf_idx = (freqs >= 5000) & (freqs < 10000)

            E_MID = np.sum(power_spectrum[:, mid_idx], axis=1)
            E_HF = np.sum(power_spectrum[:, hf_idx], axis=1)
            self._deesser_bands = (E_MID, E_HF)
        return self._deesser_bands

    def get_macroscopic(self):
        if not self._macroscopic:
            rms = np.sqrt(np.mean(self.audio**2))
            rms_db = 20 * np.log10(rms + EPS)
            peak_db = 20 * np.log10(np.max(np.abs(self.audio)) + EPS)
            self._macroscopic['CF'] = peak_db - rms_db

            self._macroscopic['LUFS'] = self.meter.integrated_loudness(self.audio)

            # HIGHLY OPTIMIZED TRUE PEAK: Oversample only the 10 loudest microscopic windows
            abs_audio = np.abs(self.audio)
            top_indices = np.argpartition(abs_audio, -10)[-10:]
            max_tp = -np.inf

            for idx in top_indices:
                start = max(0, idx - 50)
                end = min(len(self.audio), idx + 50)
                window_os = signal.resample_poly(self.audio[start:end], up=4, down=1)
                tp_val = np.max(np.abs(window_os))
                if tp_val > max_tp:
                    max_tp = tp_val

            self._macroscopic['TP'] = 20 * np.log10(max_tp + EPS)

            if len(self.audio) / self.sr > 3.0:
                try:
                    lra_val = self.meter.loudness_range(self.audio)
                    self._macroscopic['LRA'] = None if math.isnan(lra_val) or math.isinf(lra_val) else float(lra_val)
                except Exception:
                    self._macroscopic['LRA'] = None
            else:
                self._macroscopic['LRA'] = None

        return self._macroscopic

# ==============================================================================
# 2. RAW UNNORMALIZED METRIC CALCULATORS
# ==============================================================================

def get_denoise_raw(feat_X0, feat_X1, feat_P):
    N_Y = np.median(feat_X1.energies[~feat_X0.speech_mask])
    N_P = np.median(feat_P.energies[~feat_P.speech_mask])
    D_N = abs(N_Y - N_P)

    E_X = feat_X0.speech_energies
    E_Y = feat_X1.energies[feat_X0.speech_mask]

    g = np.median(E_X - E_Y)
    E_Y_prime = E_Y + g
    D_env = np.sqrt(np.mean((E_Y_prime - E_X)**2))

    thresh_20 = np.percentile(E_X, 20)
    Q_mask = E_X <= thresh_20

    A_quiet_val = -np.median(E_Y_prime[Q_mask] - E_X[Q_mask])
    A_quiet = max(0, A_quiet_val)

    return D_N, D_env, A_quiet

def get_compressor_raw(feat_X1, feat_X2, feat_P):
    L_Y = feat_X2.speech_energies
    L_P = feat_P.speech_energies

    Q10_Y, Q50_Y, Q90_Y = np.percentile(L_Y, [10, 50, 90])
    Q10_P, Q50_P, Q90_P = np.percentile(L_P, [10, 50, 90])

    D_low = abs((Q50_Y - Q10_Y) - (Q50_P - Q10_P))
    D_high = abs((Q90_Y - Q50_Y) - (Q90_P - Q50_P))

    mac_Y = feat_X2.get_macroscopic()
    mac_P = feat_P.get_macroscopic()

    D_CF = abs(mac_Y['CF'] - mac_P['CF'])

    D_LRA = None
    if mac_Y['LRA'] is not None and mac_P['LRA'] is not None:
        D_LRA = abs(mac_Y['LRA'] - mac_P['LRA'])

    GR = feat_X2.energies - feat_X1.energies
    delta_GR = np.diff(GR)
    P_pump = np.mean(np.abs(delta_GR)) / (np.mean(np.abs(GR)) + EPS)

    return D_low, D_high, D_CF, D_LRA, P_pump

def get_eq_raw(feat_X3, feat_P):
    SB_Y = feat_X3.eq_bands / (np.sum(feat_X3.eq_bands) + EPS)
    SB_P = feat_P.eq_bands / (np.sum(feat_P.eq_bands) + EPS)

    D_SB = 0.5 * np.sum(np.abs(SB_Y - SB_P))
    return D_SB

def get_deesser_raw(feat_X3, feat_X4, feat_P):
    # Anchor the evaluation to X3's mask so before/after arrays perfectly align
    mask = feat_X3.speech_mask

    E_MID_X3_full, E_HF_X3_full = feat_X3.deesser_bands
    E_MID_Y_full, E_HF_Y_full = feat_X4.deesser_bands

    E_MID_X3 = E_MID_X3_full[mask]
    E_HF_X3 = E_HF_X3_full[mask]
    E_MID_Y = E_MID_Y_full[mask]
    E_HF_Y = E_HF_Y_full[mask]

    # Produced track evaluates its own mask independently
    E_MID_P = feat_P.deesser_bands[0][feat_P.speech_mask]
    E_HF_P = feat_P.deesser_bands[1][feat_P.speech_mask]

    SI_Y_frames = 10 * np.log10(E_HF_Y / (E_MID_Y + EPS) + EPS)
    SI_P_frames = 10 * np.log10(E_HF_P / (E_MID_P + EPS) + EPS)

    thresh_Y = np.percentile(SI_Y_frames, 90) if len(SI_Y_frames) > 0 else 0
    thresh_P = np.percentile(SI_P_frames, 90) if len(SI_P_frames) > 0 else 0

    sib_mask_Y = SI_Y_frames >= thresh_Y
    sib_mask_P = SI_P_frames >= thresh_P

    SI_Y = np.median(SI_Y_frames[sib_mask_Y]) if np.any(sib_mask_Y) else 0.0
    SI_P = np.median(SI_P_frames[sib_mask_P]) if np.any(sib_mask_P) else 0.0
    D_sib = abs(SI_Y - SI_P)

    HFfrac_Y = np.median(E_HF_Y[~sib_mask_Y] / (E_HF_Y[~sib_mask_Y] + E_MID_Y[~sib_mask_Y] + EPS)) if np.any(~sib_mask_Y) else 0.0
    HFfrac_P = np.median(E_HF_P[~sib_mask_P] / (E_HF_P[~sib_mask_P] + E_MID_P[~sib_mask_P] + EPS)) if np.any(~sib_mask_P) else 0.0
    D_HF = abs(HFfrac_Y - HFfrac_P)

    HF_before = E_HF_X3 / (E_HF_X3 + E_MID_X3 + EPS)
    HF_after = E_HF_Y / (E_HF_Y + E_MID_Y + EPS)
    A_i = HF_before - HF_after

    SI_before = 10 * np.log10(E_HF_X3 / (E_MID_X3 + EPS) + EPS)
    sib_mask_select = SI_before >= (np.percentile(SI_before, 90) if len(SI_before) > 0 else 0)

    A_sib = np.median(A_i[sib_mask_select]) if np.any(sib_mask_select) else 0.0
    A_normal = np.median(A_i[~sib_mask_select]) if np.any(~sib_mask_select) else 0.0
    S_select = A_sib / (A_sib + A_normal + EPS)

    return D_sib, D_HF, S_select

def get_reverb_raw(audio_X5, R_P, sr):
    # R_P is now a pre-calculated float, saving thousands of compute cycles
    R_Y, _ = srmr(audio_X5, sr, fast=True, norm=False)
    D_R = abs(R_Y - R_P) / (abs(R_Y) + abs(R_P) + EPS)
    return D_R

def get_limiter_raw(feat_X5, feat_X6, feat_P, ceiling=None):
    mac_Y = feat_X6.get_macroscopic()
    mac_P = feat_P.get_macroscopic()

    D_L = abs(mac_Y['LUFS'] - mac_P['LUFS'])
    D_TP = abs(mac_Y['TP'] - mac_P['TP'])

    P_ceiling = 0.0
    if ceiling is not None:
        P_ceiling = max(0, mac_Y['TP'] - ceiling)

    GR_mean = np.mean(feat_X6.energies - feat_X5.energies)
    return D_L, D_TP, P_ceiling, GR_mean


# ==============================================================================
# 3. DYNAMIC JSON CONFIGURATION PARSER
# ==============================================================================

def build_pedalboard_chain(preset_data):
    """
    Dynamically maps JSON dictionary specifications to Pedalboard Python objects.
    """
    if preset_data.get("bypass", False):
        return pedalboard.Pedalboard([])

    pb_effects = []
    for effect in preset_data.get("effects", []):
        etype = effect["type"]
        params = effect.get("params", {})

        # Handle JSON to Pedalboard class name mismatches
        if etype == "LowShelfFilter":
            plugin_class = pedalboard.LowShelfFilter
        elif etype == "HighShelfFilter":
            plugin_class = pedalboard.HighShelfFilter
        else:
            # Fetch directly from pedalboard library (NoiseGate, Compressor, PeakFilter, Reverb, Limiter)
            plugin_class = getattr(pedalboard, etype)

        pb_effects.append(plugin_class(**params))

    return pedalboard.Pedalboard(pb_effects)

def load_voxeffects_space(config_path):
    with open(config_path, 'r') as f:
        vox_config = json.load(f)

    space = {
        "denoise": [build_pedalboard_chain(p) for p in vox_config["de_noise_presets"]],
        "compressor": [build_pedalboard_chain(p) for p in vox_config["dynamic_presets"]],
        "eq": [build_pedalboard_chain(p) for p in vox_config["eq_presets"]],
        "deesser": [build_pedalboard_chain(p) for p in vox_config["de_esser_presets"]],
        "reverb": [build_pedalboard_chain(p) for p in vox_config["reverb_presets"]],
        "limiter": [build_pedalboard_chain(p) for p in vox_config["limiter_presets"]]
    }

    print(f"Loaded dynamically from JSON: Denoise({len(space['denoise'])}), "
          f"Compressor({len(space['compressor'])}), EQ({len(space['eq'])}), "
          f"Deesser({len(space['deesser'])}), Reverb({len(space['reverb'])}), "
          f"Limiter({len(space['limiter'])})")

    return space

def apply_plugin(plugin, audio, sr):
    # CRITICAL FIX: Flush the plugin's internal DSP buffers so previous loops don't bleed in
    if hasattr(plugin, 'reset'):
        plugin.reset()
    """Forces 1D audio to (Channels, Samples), processes it, flattens back, and perfectly aligns length."""
    audio_2d = np.expand_dims(audio, axis=0) if audio.ndim == 1 else audio
    out_2d = plugin(audio_2d, sr)
    out_1d = out_2d.squeeze()

    # Bulletproof length alignment
    if len(out_1d) < len(audio):
        out_1d = np.pad(out_1d, (0, len(audio) - len(out_1d)))
    return out_1d[:len(audio)]

def get_limiter_threshold(plugin_chain):
    """Safely extracts the threshold_db from inside a Pedalboard container."""
    if not hasattr(plugin_chain, '__iter__'):
        return getattr(plugin_chain, 'threshold_db', 0.0)

    for effect in plugin_chain:
        if isinstance(effect, pedalboard.Limiter):
            return effect.threshold_db
    return 0.0

def create_vad_mask(audio, sr, top_db=35, frame_ms=20, hop_ms=10):
    """Uses standard Librosa VAD to find non-silent intervals and creates a boolean frame mask."""
    # Librosa standard VAD: noise is anything 35dB below the absolute peak
    intervals = librosa.effects.split(audio, top_db=top_db)

    hop_len = int(sr * hop_ms / 1000)
    frame_len = int(sr * frame_ms / 1000)
    num_frames = 1 + (len(audio) - frame_len) // hop_len

    mask = np.zeros(num_frames, dtype=bool)
    for start_sample, end_sample in intervals:
        start_frame = start_sample // hop_len
        end_frame = end_sample // hop_len
        mask[start_frame:end_frame+1] = True

    return mask


# ==============================================================================
# 4. OPTIMIZED NESTED PROCESSING PIPELINE (WITH CRITICAL FIXES)
# ==============================================================================

def process_recording(raw_path, produced_path, output_jsonl_path, space):
    audio_X0, sr = sf.read(raw_path)
    audio_P, sr_p = sf.read(produced_path)

    if audio_X0.ndim > 1: audio_X0 = np.mean(audio_X0, axis=1)
    if audio_P.ndim > 1: audio_P = np.mean(audio_P, axis=1)

    if sr != sr_p:
        audio_P = signal.resample_poly(audio_P, up=sr, down=sr_p)




    # 1. Initialize shared tools ONCE to save memory
    shared_meter = pyln.Meter(sr)

    # 2. Compute librosa VAD precisely ONCE per file
    mask_X0 = create_vad_mask(audio_X0, sr)
    mask_P = create_vad_mask(audio_P, sr) # sr matches due to earlier resampling

    # 3. Pre-compute heavy metrics for the Produced track ONCE
    feat_P = AudioFeatures(audio_P, sr, shared_meter, mask_P)
    R_P_target, _ = srmr(audio_P, sr, fast=True, norm=False)

    feat_X0 = AudioFeatures(audio_X0, sr, shared_meter, mask_X0)

    with open(output_jsonl_path, 'a') as f_out:
        for idx_dn, plugin_dn in enumerate(space["denoise"]):
            audio_X1 = apply_plugin(plugin_dn, audio_X0, sr)[:len(audio_X0)]
            feat_X1 = AudioFeatures(audio_X1, sr, shared_meter, mask_X0)
            raw_dn = get_denoise_raw(feat_X0, feat_X1, feat_P)

            for idx_c, plugin_c in enumerate(space["compressor"]):
                audio_X2 = apply_plugin(plugin_c, audio_X1, sr)[:len(audio_X1)]
                feat_X2 = AudioFeatures(audio_X2, sr, shared_meter, mask_X0)
                raw_comp = get_compressor_raw(feat_X1, feat_X2, feat_P)

                for idx_eq, plugin_eq in enumerate(space["eq"]):
                    audio_X3 = apply_plugin(plugin_eq, audio_X2, sr)[:len(audio_X2)]
                    feat_X3 = AudioFeatures(audio_X3, sr, shared_meter, mask_X0)
                    raw_eq = get_eq_raw(feat_X3, feat_P)

                    for idx_ds, plugin_ds in enumerate(space["deesser"]):
                        audio_X4 = apply_plugin(plugin_ds, audio_X3, sr)[:len(audio_X3)]
                        feat_X4 = AudioFeatures(audio_X4, sr, shared_meter, mask_X0)
                        raw_ds = get_deesser_raw(feat_X3, feat_X4, feat_P)

                        for idx_rvb, plugin_rvb in enumerate(space["reverb"]):
                            audio_X5 = apply_plugin(plugin_rvb, audio_X4, sr)[:len(audio_X4)]
                            feat_X5 = AudioFeatures(audio_X5, sr, shared_meter, mask_X0)

                            raw_rvb = get_reverb_raw(audio_X5, R_P_target, sr)

                            for idx_lim, plugin_lim in enumerate(space["limiter"]):
                                audio_X6 = apply_plugin(plugin_lim, audio_X5, sr)[:len(audio_X5)]
                                feat_X6 = AudioFeatures(audio_X6, sr, shared_meter, mask_X0)

                                ceil_val = get_limiter_threshold(plugin_lim) if hasattr(plugin_lim, 'process') else 0.0
                                raw_lim = get_limiter_raw(feat_X5, feat_X6, feat_P, ceiling=ceil_val)

                                record = {
                                    "recording_id": os.path.basename(raw_path),
                                    "preset_id": f"{idx_dn}-{idx_c}-{idx_eq}-{idx_ds}-{idx_rvb}-{idx_lim}",
                                    "indices": {
                                        "denoise": idx_dn, "compressor": idx_c, "eq": idx_eq,
                                        "deesser": idx_ds, "reverb": idx_rvb, "limiter": idx_lim
                                    },
                                    "raw_metrics": {
                                        "denoise": {
                                            "D_N": float(raw_dn[0]), "D_env": float(raw_dn[1]), "A_quiet": float(raw_dn[2])
                                        },
                                        "compressor": {
                                            "D_low": float(raw_comp[0]), "D_high": float(raw_comp[1]),
                                            "D_CF": float(raw_comp[2]), "D_LRA": float(raw_comp[3]) if raw_comp[3] is not None else None,
                                            "P_pump": float(raw_comp[4])
                                        },
                                        "eq": {
                                            "D_SB": float(raw_eq)
                                        },
                                        "deesser": {
                                            "D_sib": float(raw_ds[0]), "D_HF": float(raw_ds[1]), "S_select": float(raw_ds[2])
                                        },
                                        "reverb": {
                                            "D_R": float(raw_rvb)
                                        },
                                        "limiter": {
                                            "D_L": float(raw_lim[0]), "D_TP": float(raw_lim[1]),
                                            "P_ceiling": float(raw_lim[2]), "GR_mean": float(raw_lim[3])
                                        }
                                    }
                                }

                                f_out.write(json.dumps(record) + '\n')


# ==============================================================================
# 5. BATCH EXECUTION RUNNER & DAPS PARSER
# ==============================================================================

def get_daps_pairs(daps_dir):
    pairs = []
    produced_dir = os.path.join(daps_dir, "produced")
    produced_files = glob.glob(os.path.join(produced_dir, "*_produced.wav"))

    for prod_path in produced_files:
        filename = os.path.basename(prod_path)
        
        # 1. Skip macOS hidden metadata files
        if filename.startswith("._"):
            continue
            
        prefix = filename.replace("_produced.wav", "")

        # Find all matching raw files inside the device/environment subfolders
        raw_candidates = glob.glob(os.path.join(daps_dir, "*", f"{prefix}_*.wav"))
        for raw_path in raw_candidates:
            raw_filename = os.path.basename(raw_path)
            
            # 2. Skip macOS hidden metadata files in the raw directories
            if raw_filename.startswith("._"):
                continue
                
            folder_name = os.path.basename(os.path.dirname(raw_path))
            
            # 3. Exclude 'produced' (itself) and 'cleanraw' (unfiltered raw takes)
            if folder_name not in ["produced", "cleanraw"]:
                pairs.append((raw_path, prod_path))

    return pairs

def process_single_pair(args):
    """Wrapper function for multiprocessing that handles individual files."""
    raw_path, produced_path, output_dir, config_path = args
    base_name = os.path.basename(raw_path).replace('.wav', '')
    output_jsonl_path = os.path.join(output_dir, f"{base_name}_metrics.jsonl")

    # RESUME CAPABILITY: If this file already exists, skip it entirely.
    if os.path.exists(output_jsonl_path):
        return f"Skipped (already exists): {base_name}"

    try:
        # Build the Pedalboard objects locally on this specific CPU core
        space = load_voxeffects_space(config_path)
        
        open(output_jsonl_path, 'w').close()
        process_recording(raw_path, produced_path, output_jsonl_path, space)
        return f"SUCCESS: {base_name}"
    except Exception as e:
        error_trace = traceback.format_exc()
        return f"ERROR on {base_name}:\n{error_trace}"

if __name__ == "__main__":
    import sys
    import math
    import traceback
    
    # Parse SLURM array arguments (default to 0 and 1 for local testing)
    chunk_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    HOME_DIR = os.environ.get("HOME")
    SCRATCH_DIR = os.environ.get("SCRATCH", HOME_DIR) 

    config_path = os.path.join(SCRATCH_DIR, "VoxEffects", "config", "speech_effect_chain_v2.json")
    daps_dir = os.path.join(SCRATCH_DIR, "daps")
    output_dir = os.path.join(HOME_DIR, "VoxEffects", "results")

    os.makedirs(output_dir, exist_ok=True)

    if chunk_idx == 0:
        print(f"Scanning DAPS directory: {daps_dir}")
        
    daps_pairs = get_daps_pairs(daps_dir)
    
    # CRITICAL: Sort the list so every node parses the exact same file order
    daps_pairs.sort()
    
    # --- CHUNKING LOGIC ---
    chunk_size = math.ceil(len(daps_pairs) / total_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(daps_pairs))
    chunk_pairs = daps_pairs[start_idx:end_idx]

    print(f"--- NODE {chunk_idx+1}/{total_chunks} --- Processing {len(chunk_pairs)} pairs (Index {start_idx} to {end_idx-1}) out of {len(daps_pairs)} total pairs.")

    pool_args = [(raw, prod, output_dir, config_path) for raw, prod in chunk_pairs]

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting parallel processing pool with {num_cores} cores...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(process_single_pair, pool_args):
            print(result)

    print(f"\n--- NODE {chunk_idx+1} FINISHED ---")
