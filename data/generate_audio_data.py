"""
Environmental Audio Dataset Generator for Audio Classification with CNNs.
Synthesizes 120 realistic environmental sound clips across 6 classes:
['siren', 'drilling', 'dog_bark', 'rain', 'engine_idle', 'gunshot']
standardized to 16,000 Hz 16-bit PCM mono audio.
"""

import os
import json
import random
import numpy as np
import soundfile as sf
import scipy.signal

CLASSES = ["siren", "drilling", "dog_bark", "rain", "engine_idle", "gunshot"]
SAMPLE_RATE = 16000
DURATION = 2.0  # 2.0 seconds per clip


def generate_siren(duration: float, sr: int) -> np.ndarray:
    """Frequency-modulated emergency siren sweep."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sweep frequency between 650 Hz and 1250 Hz at 0.75 Hz rate
    f_mod = 950.0 + 300.0 * np.sin(2 * np.pi * 0.75 * t)
    phase = 2 * np.pi * np.cumsum(f_mod) / sr
    sig = np.sin(phase) + 0.3 * np.sin(2 * phase)
    return sig.astype(np.float32)


def generate_drilling(duration: float, sr: int) -> np.ndarray:
    """Mechanical drilling sound with rapid impacts and metal friction."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base_freq = random.uniform(140.0, 180.0)
    # Sum harmonics
    sig = np.zeros_like(t)
    for h in [1, 2, 3, 5, 7, 9, 12]:
        sig += (1.0 / np.sqrt(h)) * np.sin(2 * np.pi * h * base_freq * t)
    # Rapid impact amplitude modulation (35 Hz vibration)
    impact = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 32.0 * t))
    noise = np.random.normal(0, 0.25, len(t))
    b, a = scipy.signal.butter(2, 2000 / (sr / 2), btype='high')
    friction = scipy.signal.lfilter(b, a, noise)
    return (sig * 0.6 * impact + friction * 0.4).astype(np.float32)


def generate_dog_bark(duration: float, sr: int) -> np.ndarray:
    """Impulsive canine bark burst with harmonic formants and pause intervals."""
    total_len = int(sr * duration)
    sig = np.zeros(total_len, dtype=np.float32)
    t_full = np.linspace(0, duration, total_len, endpoint=False)
    
    # Generate 2 to 3 barks within 2 seconds
    num_barks = random.randint(2, 3)
    bark_starts = np.linspace(0.15, duration - 0.6, num_barks)
    bark_dur = 0.35
    
    for start_t in bark_starts:
        start_idx = int(start_t * sr)
        end_idx = min(start_idx + int(bark_dur * sr), total_len)
        chunk_len = end_idx - start_idx
        t_bark = np.linspace(0, bark_dur, chunk_len, endpoint=False)
        
        # Fundamental pitch ~420 Hz with formant resonances
        f0 = random.uniform(380.0, 440.0)
        bark_wave = np.sin(2 * np.pi * f0 * t_bark) + 0.6 * np.sin(2 * np.pi * 2.2 * f0 * t_bark) + 0.3 * np.sin(2 * np.pi * 3.5 * f0 * t_bark)
        # Envelope: sharp exponential attack, fast decay
        env = np.exp(-9.0 * t_bark) * (t_bark / 0.05)
        env = np.clip(env, 0, 1.0)
        sig[start_idx:end_idx] += bark_wave * env
        
    return sig


def generate_rain(duration: float, sr: int) -> np.ndarray:
    """Stochastic broadband rainfall with filtered pink noise and droplet transient bursts."""
    num_samples = int(sr * duration)
    # Filtered pink/brown noise
    white = np.random.normal(0, 1.0, num_samples)
    b, a = scipy.signal.butter(2, [300 / (sr / 2), 4500 / (sr / 2)], btype='band')
    rain_base = scipy.signal.lfilter(b, a, white)
    
    # Sparse droplet impulses
    droplets = np.zeros(num_samples, dtype=np.float32)
    num_drops = int(duration * 40)
    drop_indices = np.random.randint(0, num_samples - 400, num_drops)
    t_drop = np.linspace(0, 0.02, 320)
    for idx in drop_indices:
        f_drop = random.uniform(2200, 3800)
        drop_pulse = np.sin(2 * np.pi * f_drop * t_drop) * np.exp(-150.0 * t_drop)
        droplets[idx:idx + 320] += drop_pulse * random.uniform(0.3, 0.8)
        
    sig = rain_base * 0.7 + droplets * 0.3
    return sig.astype(np.float32)


def generate_engine_idle(duration: float, sr: int) -> np.ndarray:
    """Low-frequency combustion engine idle rumble with rhythmic mechanical subharmonics."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental rotation frequency ~ 28 Hz (idle ~840 RPM)
    f_rpm = random.uniform(26.0, 32.0)
    sig = np.zeros_like(t)
    for harmonic in [1, 2, 3, 4, 6, 8]:
        amp = 1.0 / (harmonic ** 0.8)
        sig += amp * np.sin(2 * np.pi * harmonic * f_rpm * t)
        
    # Rhythmic cylinder exhaust bursts
    combustion_pulse = 0.5 + 0.5 * (np.sin(2 * np.pi * (2 * f_rpm) * t) ** 4)
    rumble = sig * combustion_pulse
    
    # Low-pass filter below 500 Hz
    b, a = scipy.signal.butter(3, 450 / (sr / 2), btype='low')
    rumble = scipy.signal.lfilter(b, a, rumble)
    return rumble.astype(np.float32)


def generate_gunshot(duration: float, sr: int) -> np.ndarray:
    """High-amplitude acoustic blast impulse followed by acoustic room reverberation."""
    total_len = int(sr * duration)
    sig = np.zeros(total_len, dtype=np.float32)
    
    # Blast onset at 0.15s
    start_idx = int(0.15 * sr)
    blast_dur = 0.65
    end_idx = min(start_idx + int(blast_dur * sr), total_len)
    chunk_len = end_idx - start_idx
    t_blast = np.linspace(0, blast_dur, chunk_len, endpoint=False)
    
    # Intense initial shockwave + resonant tail
    shock = np.random.normal(0, 1.0, chunk_len) * np.exp(-35.0 * t_blast)
    tail = np.sin(2 * np.pi * 180.0 * t_blast) * np.exp(-7.0 * t_blast)
    
    b, a = scipy.signal.butter(2, 1500 / (sr / 2), btype='low')
    reverb = scipy.signal.lfilter(b, a, shock * 0.7 + tail * 0.5)
    sig[start_idx:end_idx] = reverb
    return sig


GENERATORS = {
    "siren": generate_siren,
    "drilling": generate_drilling,
    "dog_bark": generate_dog_bark,
    "rain": generate_rain,
    "engine_idle": generate_engine_idle,
    "gunshot": generate_gunshot
}


def main():
    random.seed(42)
    np.random.seed(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    num_samples_per_class = 20
    dataset_records = []
    sample_counter = 1
    
    print(f"Synthesizing environmental audio clips ({num_samples_per_class} per class x {len(CLASSES)} classes)...")
    
    for cls in CLASSES:
        gen_fn = GENERATORS[cls]
        print(f"Generating class [{cls}]...")
        
        for i in range(num_samples_per_class):
            sample_id = f"snd_{sample_counter:03d}"
            filename = f"{sample_id}_{cls}.wav"
            final_path = os.path.join(audio_dir, filename)
            
            raw_audio = gen_fn(DURATION, SAMPLE_RATE)
            
            # Amplitude normalization with dynamic headroom
            max_amp = np.max(np.abs(raw_audio))
            if max_amp > 0:
                raw_audio = (raw_audio / max_amp) * random.uniform(0.85, 0.95)
                
            sf.write(final_path, raw_audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
            
            # 75% train / 25% test split (15 train / 5 test per class)
            split = "test" if (i >= 15) else "train"
            
            record = {
                "id": sample_id,
                "filename": filename,
                "filepath": os.path.relpath(final_path, os.path.dirname(base_dir)).replace(os.sep, "/"),
                "category": cls,
                "duration_sec": DURATION,
                "sample_rate": SAMPLE_RATE,
                "split": split
            }
            dataset_records.append(record)
            sample_counter += 1
            
    dataset_json_path = os.path.join(base_dir, "dataset.json")
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=2)
        
    num_train = sum(1 for r in dataset_records if r["split"] == "train")
    num_test = sum(1 for r in dataset_records if r["split"] == "test")
    print(f"\nGenerated {len(dataset_records)} audio clips successfully!")
    print(f"Split distribution: {num_train} train ({num_train/len(dataset_records)*100:.1f}%), {num_test} test ({num_test/len(dataset_records)*100:.1f}%)")
    print(f"Dataset metadata saved to: {dataset_json_path}")


if __name__ == "__main__":
    main()
