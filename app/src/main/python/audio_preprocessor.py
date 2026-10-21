import wave
import numpy as np

class AudioPreprocessor:
    def __init__(self):
        self.target_frames = 1723
        self.n_fft = 1024
        self.hop_length = 512
        
        # Exact mathematical replica of Librosa's default periodic Hann window
        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)

    def process_wav_to_tensor(self, file_path: str) -> bytes:
        # 1. Read WAV file using Python's native wave module
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            raw_data = wav_file.readframes(n_frames)

        # 2. Convert raw bytes to normalized float32 (-1.0 to 1.0)
        if sampwidth == 2: # 16-bit PCM
            data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(raw_data, dtype=np.float32)

        # Convert to mono if stereo
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        # 3. 1:1 LIBROSA STFT REPLICA
        # Librosa center=True default (Reflection padding)
        pad_width = self.n_fft // 2
        y_padded = np.pad(data, pad_width, mode='reflect')

        # Frame the signal (Librosa framing logic via numpy stride tricks)
        n_frames_stft = 1 + (len(y_padded) - self.n_fft) // self.hop_length
        strides = (y_padded.itemsize * self.hop_length, y_padded.itemsize)
        frames = np.lib.stride_tricks.as_strided(
            y_padded, 
            shape=(n_frames_stft, self.n_fft), 
            strides=strides
        )

        # Apply window and execute Fast Fourier Transform
        # rfft returns 513 bins for a 1024 n_fft. Transpose to match Librosa's (Freq, Time) shape.
        stft_matrix = np.fft.rfft(frames * self.window, axis=1).T

        # 4. Magnitude & Librosa amplitude_to_db
        magnitude = np.abs(stft_matrix)
        amin = 1e-5 # Librosa's default noise floor prevention
        magnitude = np.maximum(amin, magnitude)
        spectrogram_db = 20 * np.log10(magnitude)

        # 5. Tensor Formatting [513, 1723]
        current_frames = spectrogram_db.shape[1]
        if current_frames < self.target_frames:
            pad_w = self.target_frames - current_frames
            # Pad empty space with -100 dB (digital silence)
            spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_w)), mode='constant', constant_values=-100.0)
        elif current_frames > self.target_frames:
            spectrogram_db = spectrogram_db[:, :self.target_frames]

        # Return flat C-contiguous byte array directly to Kotlin
        return np.ascontiguousarray(spectrogram_db, dtype=np.float32).tobytes()