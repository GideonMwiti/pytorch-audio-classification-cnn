"""
Project 61: Audio Classification with Deep 2D-CNNs
Classifies environmental sound clips from 64-band Log-Mel Spectrogram representations
using a deep 4-stage 2D Convolutional Neural Network with SpecAugment regularization.
"""

import os
import sys
import time
import json
import random
import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Configure plot styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

CLASSES = ["siren", "drilling", "dog_bark", "rain", "engine_idle", "gunshot"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for i, c in enumerate(CLASSES)}


# =====================================================================
# 1. Mel Spectrogram Audio Extraction Engine
# =====================================================================
class MelSpectrogramExtractor:
    """Converts 1D continuous audio signals into 64-band Log-Mel Spectrograms."""

    def __init__(self, sr: int = 16000, n_mels: int = 64, n_fft: int = 1024, 
                 hop_length: int = 250, target_frames: int = 128):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames
        self.fbank = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> np.ndarray:
        """Constructs triangular Mel frequency filterbank matrix."""
        mel_min = 0.0
        mel_max = 2595.0 * np.log10(1.0 + (self.sr / 2.0) / 700.0)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)
        
        fbank = np.zeros((self.n_mels, int(np.floor(self.n_fft / 2 + 1))))
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-8)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-8)
        return fbank

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts normalized Log-Mel Spectrogram.
        Returns shape: (1, n_mels=64, target_frames=128)
        """
        # 1. Short-Time Fourier Transform (STFT)
        _, _, Zxx = scipy.signal.stft(
            audio, 
            fs=self.sr, 
            nperseg=self.n_fft, 
            noverlap=self.n_fft - self.hop_length, 
            window='hann'
        )
        power_spectrum = np.abs(Zxx) ** 2
        
        # 2. Apply Mel filterbank & Logarithmic compression
        mel_energies = np.dot(self.fbank, power_spectrum)
        mel_energies = np.where(mel_energies <= 0, 1e-10, mel_energies)
        log_mel = 10.0 * np.log10(mel_energies)  # [64, T]
        
        # 3. Standardize temporal length to target_frames (128)
        curr_frames = log_mel.shape[1]
        if curr_frames < self.target_frames:
            pad_width = self.target_frames - curr_frames
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        elif curr_frames > self.target_frames:
            start = (curr_frames - self.target_frames) // 2
            log_mel = log_mel[:, start:start + self.target_frames]
            
        # 4. Zero-mean, unit-variance normalization
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-8
        log_mel = (log_mel - mean) / std
        
        # Add channel dimension: (1, 64, 128)
        return np.expand_dims(log_mel, axis=0).astype(np.float32)


# =====================================================================
# 2. SpecAugment Spectrogram Augmentation
# =====================================================================
def apply_spec_augment(spec: torch.Tensor, max_f_mask: int = 8, max_t_mask: int = 16) -> torch.Tensor:
    """Applies frequency masking and time masking to the spectrogram tensor."""
    aug_spec = spec.clone()
    _, f_bins, t_frames = aug_spec.shape
    
    # Frequency Masking
    f_width = random.randint(0, max_f_mask)
    if f_width > 0:
        f_start = random.randint(0, f_bins - f_width)
        aug_spec[:, f_start:f_start + f_width, :] = 0.0
        
    # Time Masking
    t_width = random.randint(0, max_t_mask)
    if t_width > 0:
        t_start = random.randint(0, t_frames - t_width)
        aug_spec[:, :, t_start:t_start + t_width] = 0.0
        
    return aug_spec


# =====================================================================
# 3. PyTorch Dataset & DataLoader
# =====================================================================
class EnvironmentalAudioDataset(Dataset):
    """Dataset for environmental sound clips with pre-extracted spectrograms."""

    def __init__(self, records: list, project_root: str, extractor: MelSpectrogramExtractor, is_training: bool = False):
        self.records = records
        self.project_root = project_root
        self.extractor = extractor
        self.is_training = is_training
        self.data = []
        self._prepare()

    def _prepare(self):
        for rec in self.records:
            full_path = os.path.join(self.project_root, rec["filepath"])
            audio, _ = sf.read(full_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            spec = self.extractor.extract(audio)
            label = CLASS2IDX[rec["category"]]
            self.data.append((spec, label, rec))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spec, label, rec = self.data[idx]
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        if self.is_training and random.random() < 0.65:
            spec_tensor = apply_spec_augment(spec_tensor)
        return spec_tensor, torch.tensor(label, dtype=torch.long), rec


# =====================================================================
# 4. Deep Audio Spectrogram 2D-CNN Architecture
# =====================================================================
class AudioSpectrogramCNN(nn.Module):
    """
    Deep 4-stage 2D-CNN for Audio Spectrogram Classification:
    - Stage 1: Conv2d(1 -> 32) + BN + ReLU + MaxPool(2,2) -> (32, 32, 64)
    - Stage 2: Conv2d(32 -> 64) + BN + ReLU + MaxPool(2,2) -> (64, 16, 32)
    - Stage 3: Conv2d(64 -> 128) + BN + ReLU + MaxPool(2,2) -> (128, 8, 16)
    - Stage 4: Conv2d(128 -> 256) + BN + ReLU + AdaptiveAvgPool -> (256, 1, 1)
    - Classification MLP: Linear(256 -> 128) + Dropout(0.4) + Linear(128 -> 6)
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Stage 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Stage 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Stage 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 64, 128]
        feat = self.features(x)  # [batch, 256, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [batch, 256]
        logits = self.classifier(feat)  # [batch, num_classes]
        return logits


# =====================================================================
# 5. Training & Evaluation Pipeline
# =====================================================================
def train_and_evaluate(dataset_path: str, output_dir: str):
    """Trains the 2D-CNN audio classifier and benchmarks performance on test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device.type.upper()}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(dataset_path)))
    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]
    
    print(f"Dataset: {len(records)} clips | Train: {len(train_records)} | Test: {len(test_records)}")
    
    extractor = MelSpectrogramExtractor()
    print("Extracting 64-band Log-Mel Spectrograms...")
    train_dataset = EnvironmentalAudioDataset(train_records, project_root, extractor, is_training=True)
    test_dataset = EnvironmentalAudioDataset(test_records, project_root, extractor, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = AudioSpectrogramCNN(num_classes=len(CLASSES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"AudioCNN Initialized! Total parameters: {total_params:,} ({total_params/1e3:.1f}K)")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    num_epochs = 20
    train_losses = []
    train_accs = []
    
    print("\nStarting CNN Training (20 Epochs with SpecAugment)...")
    print("-" * 65)
    start_train_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for specs, labels, _ in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:5.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    train_duration = time.time() - start_train_time
    print(f"Training completed in {train_duration:.2f}s!")
    
    # -----------------------------------------------------------------
    # Test Evaluation
    # -----------------------------------------------------------------
    print(f"\nEvaluating on Held-Out Test Set ({len(test_records)} Clips)...")
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    latencies = []
    test_details = []
    
    with torch.no_grad():
        for specs, labels, recs in test_loader:
            specs, labels = specs.to(device), labels.to(device)
            
            t0 = time.time()
            logits = model(specs)
            lat = (time.time() - t0) * 1000.0 / len(labels)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(labels)):
                true_lbl = labels[i].item()
                pred_lbl = preds[i].item()
                conf = probs[i, pred_lbl].item()
                y_true.append(true_lbl)
                y_pred.append(pred_lbl)
                y_probs.append(probs[i].cpu().numpy().tolist())
                latencies.append(lat)
                
                is_correct = (true_lbl == pred_lbl)
                rec_id = recs["id"][i] if isinstance(recs["id"], list) else recs[i]["id"]
                rec_fn = recs["filename"][i] if isinstance(recs["filename"], list) else recs[i]["filename"]
                
                test_details.append({
                    "id": rec_id,
                    "filename": rec_fn,
                    "ground_truth": IDX2CLASS[true_lbl],
                    "predicted": IDX2CLASS[pred_lbl],
                    "confidence": float(conf),
                    "is_correct": bool(is_correct),
                    "latency_ms": float(lat)
                })

    test_acc = (sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    mean_latency = float(np.mean(latencies))
    
    print("=" * 75)
    print(f"Environmental Audio Classification Test Accuracy : {test_acc:.2f}%")
    print(f"Macro Precision                                : {prec * 100:.2f}%")
    print(f"Macro Recall                                   : {rec * 100:.2f}%")
    print(f"Macro F1-Score                                 : {f1 * 100:.2f}%")
    print(f"Mean Inference Latency                         : {mean_latency:.2f} ms")
    print("=" * 75)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    # Save model weights
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "audio_cnn_model.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Trained model weights saved to: {weights_path}")
    
    # Save benchmark metrics JSON
    results_json_path = os.path.join(output_dir, "audio_classification_results.json")
    results_payload = {
        "model_architecture": "4-Stage 2D-CNN (AudioSpectrogramCNN)",
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "test_accuracy": test_acc,
        "macro_precision": float(prec * 100.0),
        "macro_recall": float(rec * 100.0),
        "macro_f1": float(f1 * 100.0),
        "mean_latency_ms": mean_latency,
        "confusion_matrix": conf_mat.tolist(),
        "test_samples": test_details
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Benchmark metrics saved to: {results_json_path}")
    
    # Generate Executive Diagnostic Visualizations
    generate_diagnostic_report(
        records=records,
        project_root=project_root,
        extractor=extractor,
        train_losses=train_losses,
        train_accs=train_accs,
        conf_mat=conf_mat,
        test_acc=test_acc,
        macro_f1=f1 * 100.0,
        output_dir=output_dir
    )


# =====================================================================
# 6. Diagnostic Visualization Generator
# =====================================================================
def generate_diagnostic_report(records: list, project_root: str, extractor: MelSpectrogramExtractor,
                               train_losses: list, train_accs: list, conf_mat: np.ndarray,
                               test_acc: float, macro_f1: float, output_dir: str):
    """Generates an executive 4-panel diagnostic chart for Audio Classification."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('#F8F9FA')

    # -------------------------------------------------------------
    # Panel 1: Time-Domain Audio Waveforms (All 6 Classes)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor('#FFFFFF')
    
    palette = ["#2563EB", "#D97706", "#059669", "#0284C7", "#7C3AED", "#DC2626"]
    offset = 0.0
    for idx, cls in enumerate(CLASSES):
        sample = next(r for r in records if r["category"] == cls)
        audio, _ = sf.read(os.path.join(project_root, sample["filepath"]))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        t_axis = np.linspace(0, 2.0, len(audio))
        sig = (audio / (np.max(np.abs(audio)) + 1e-8)) * 0.4
        
        ax1.plot(t_axis, sig + offset, color=palette[idx], linewidth=0.9, label=cls.replace('_', ' ').title())
        offset += 0.95
        
    ax1.set_yticks(np.arange(len(CLASSES)) * 0.95)
    ax1.set_yticklabels([c.replace('_', ' ').title() for c in CLASSES], fontsize=8, fontweight='bold')
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_title("A) Time-Domain Waveforms Across 6 Environmental Sound Classes", fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    # -------------------------------------------------------------
    # Panel 2: 64-Band Log-Mel Spectrograms Comparison
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor('#FFFFFF')
    
    # Contrast Siren vs Rain
    sample_siren = next(r for r in records if r["category"] == "siren")
    sample_rain = next(r for r in records if r["category"] == "rain")
    
    audio_siren, _ = sf.read(os.path.join(project_root, sample_siren["filepath"]))
    audio_rain, _ = sf.read(os.path.join(project_root, sample_rain["filepath"]))
    
    spec_siren = extractor.extract(audio_siren)[0]  # [64, 128]
    spec_rain = extractor.extract(audio_rain)[0]
    
    divider = np.zeros((4, spec_siren.shape[1]))
    stacked = np.vstack([spec_siren, divider, spec_rain])
    
    im2 = ax2.imshow(stacked, aspect='auto', cmap='viridis', origin='lower')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label("Normalized Log-Mel Energy", fontsize=8)
    
    ax2.axhline(65.5, color='#FF0055', linestyle='--', linewidth=1.5)
    ax2.text(5, 115, "Siren (Harmonic Sweep Modulations)", color='#FFFFFF', fontsize=9, fontweight='bold')
    ax2.text(5, 10, "Rain (Broadband Stochastic Noise)", color='#FFFFFF', fontsize=9, fontweight='bold')
    ax2.set_title("B) 64-Band Log-Mel Spectrogram Comparison (Siren vs Rain)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("Time Frame Index (hop: 15.6 ms)", fontsize=9)
    ax2.set_ylabel("Mel Frequency Bin Index", fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor('#FFFFFF')
    
    cm_norm = conf_mat.astype('float') / (conf_mat.sum(axis=1)[:, np.newaxis] + 1e-8)
    im3 = ax3.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1.0)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.85)
    cbar3.set_label("Classification Accuracy", fontsize=8)
    
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = conf_mat[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax3.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center", 
                     color=text_color, fontsize=8, fontweight='bold')
                     
    ax3.set_xticks(range(len(CLASSES)))
    ax3.set_yticks(range(len(CLASSES)))
    ax3.set_xticklabels([c.replace('_', '\n') for c in CLASSES], fontsize=8)
    ax3.set_yticklabels([c.replace('_', ' ') for c in CLASSES], fontsize=8)
    ax3.set_xlabel("Predicted Sound Class", fontsize=9)
    ax3.set_ylabel("Ground Truth Class", fontsize=9)
    ax3.set_title(f"C) Confusion Matrix on Held-Out Test Set (Accuracy: {test_acc:.1f}%)", fontsize=11, fontweight='bold', pad=10)

    # -------------------------------------------------------------
    # Panel 4: SpecAugment & CNN Training Dynamics
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor('#FFFFFF')
    epochs = range(1, len(train_losses) + 1)
    
    line1 = ax4.plot(epochs, train_losses, color='#EF4444', linewidth=2.0, label='Cross-Entropy Loss')
    ax4.set_xlabel("Training Epoch", fontsize=9)
    ax4.set_ylabel("Training Loss", color='#EF4444', fontsize=9)
    ax4.tick_params(axis='y', labelcolor='#EF4444')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(epochs, train_accs, color='#10B981', linewidth=2.0, linestyle='--', label='Training Accuracy (%)')
    ax4_twin.set_ylabel("Accuracy (%)", color='#10B981', fontsize=9)
    ax4_twin.tick_params(axis='y', labelcolor='#10B981')
    ax4_twin.set_ylim(0, 105)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right', fontsize=8)
    ax4.set_title(f"D) 2D-CNN Training Convergence (Macro F1: {macro_f1:.1f}%)", fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "audio_classification_report.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic report successfully saved to: {chart_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "data", "dataset.json")
    output_dir = os.path.join(base_dir, "outputs")
    train_and_evaluate(dataset_path, output_dir)


if __name__ == "__main__":
    main()
