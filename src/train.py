import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix

device = "cuda" if torch.cuda.is_available() else exit()
os.makedirs("checkpoints", exist_ok=True)
print(device)


# =====================================================
# Dataset
# =====================================================
class CellDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.filenames = sorted([
            f for f in os.listdir(image_folder)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_folder, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = np.stack([img] * 3, axis=2)

        if self.transform:
            img = self.transform(img)

        label = 1 if "abnormal" in fname.lower() else 0
        return img, torch.tensor(label, dtype=torch.float32)


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


def find_best_threshold(targets, preds, metric='f1'):

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    all_metrics = {'threshold': [], 'f1': [], 'recall': [], 'precision': [], 'accuracy': []}

    for thresh in thresholds:
        preds_binary = (np.array(preds) > thresh).astype(int)

        f1 = f1_score(targets, preds_binary, zero_division=0)
        recall = recall_score(targets, preds_binary, zero_division=0)
        precision = precision_score(targets, preds_binary, zero_division=0)
        accuracy = (preds_binary == targets).mean()

        tn, fp, fn, tp = confusion_matrix(targets, preds_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sensitivity + specificity - 1

        all_metrics['threshold'].append(thresh)
        all_metrics['f1'].append(f1)
        all_metrics['recall'].append(recall)
        all_metrics['precision'].append(precision)
        all_metrics['accuracy'].append(accuracy)

        if metric == 'f1':
            score = f1
        elif metric == 'recall':
            score = recall
        elif metric == 'precision':
            score = precision
        elif metric == 'accuracy':
            score = accuracy
        elif metric == 'youden':
            score = youden
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score, all_metrics


def plot_threshold_analysis(targets, preds, fold, prefix):
    best_threshold_f1, best_f1, metrics_f1 = find_best_threshold(targets, preds, metric='f1')
    best_threshold_youden, best_youden, metrics_youden = find_best_threshold(targets, preds, metric='youden')

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(metrics_f1['threshold'], metrics_f1['f1'], label='F1 Score', linewidth=2)
    plt.plot(metrics_f1['threshold'], metrics_f1['recall'], label='Recall', linewidth=2)
    plt.plot(metrics_f1['threshold'], metrics_f1['precision'], label='Precision', linewidth=2)
    plt.axvline(best_threshold_f1, color='r', linestyle='--', label=f'Best F1 threshold: {best_threshold_f1:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Fold {fold} - F1, Recall, Precision vs Threshold')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(metrics_youden['threshold'], metrics_youden['accuracy'], label='Accuracy', linewidth=2)
    youden_scores = [metrics_youden['recall'][i] + metrics_youden['precision'][i] - 1 for i in
                     range(len(metrics_youden['threshold']))]
    plt.plot(metrics_youden['threshold'], youden_scores, label='Youden Index', linewidth=2)
    plt.axvline(best_threshold_youden, color='r', linestyle='--',
                label=f'Best Youden threshold: {best_threshold_youden:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Fold {fold} - Accuracy and Youden Index vs Threshold')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"checkpoints/{prefix}_fold_{fold}_threshold_analysis.png")
    plt.close()

    return best_threshold_f1, best_threshold_youden


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, return_predictions=False):
    model.eval()
    preds, targets = [], []
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)

    preds_binary = (np.array(preds) > 0.5).astype(int)
    auc = roc_auc_score(targets, preds)
    f1 = f1_score(targets, preds_binary, zero_division=0)
    recall = recall_score(targets, preds_binary, zero_division=0)

    if return_predictions:
        return avg_loss, auc, f1, recall, np.array(preds), np.array(targets)
    else:
        return avg_loss, auc, f1, recall

def run_training(data_path):
    full_dataset = CellDataset(data_path)
    labels = np.array([
        1 if "abnormal" in f.lower() else 0
        for f in full_dataset.filenames
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    all_best_thresholds = []

    prefix = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n========== Fold {fold + 1} ==========")

        train_subset = Subset(CellDataset(data_path, train_transform), train_idx)
        val_subset = Subset(CellDataset(data_path, val_transform), val_idx)

        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 1)
        )
        model = model.to(device)

        pos_weight = torch.tensor(
            [(len(labels) - sum(labels)) / sum(labels)]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        train_losses, val_losses = [], []
        auc_list = []

        val_predictions_list = []
        val_targets_list = []

        best_auc = 0
        best_epoch = 0
        patience = 5
        counter = 0

        best_preds = None
        best_targets = None

        for epoch in range(30):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, auc, f1, recall, val_preds, val_targets = validate(
                model, val_loader, criterion, return_predictions=True
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            auc_list.append(auc)

            val_predictions_list.append(val_preds)
            val_targets_list.append(val_targets)

            print(f"Epoch {epoch + 1:02d} | "
                  f"TrainLoss {train_loss:.4f} | "
                  f"ValLoss {val_loss:.4f} | "
                  f"AUC {auc:.4f} | "
                  f"F1(0.5) {f1:.4f} | "
                  f"Recall(0.5) {recall:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                counter = 0
                torch.save(model.state_dict(),
                           f"checkpoints/{prefix}_best_model_fold{fold + 1}.pth")
                best_preds = val_preds
                best_targets = val_targets
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break

        print(f"\n--- Finding optimal threshold for Fold {fold + 1} ---")

        best_threshold_f1, best_f1_score, _ = find_best_threshold(
            best_targets, best_preds, metric='f1'
        )
        best_threshold_youden, best_youden_score, _ = find_best_threshold(
            best_targets, best_preds, metric='youden'
        )

        print(f"Optimal threshold (by F1): {best_threshold_f1:.3f} (F1={best_f1_score:.4f})")
        print(f"Optimal threshold (by Youden): {best_threshold_youden:.3f} (Youden={best_youden_score:.4f})")

        plot_threshold_analysis(best_targets, best_preds, fold + 1, prefix)

        preds_optimal = (best_preds > best_threshold_f1).astype(int)
        optimal_f1 = f1_score(best_targets, preds_optimal, zero_division=0)
        optimal_recall = recall_score(best_targets, preds_optimal, zero_division=0)
        optimal_precision = precision_score(best_targets, preds_optimal, zero_division=0)

        print(f"\nPerformance with optimal threshold {best_threshold_f1:.3f}:")
        print(f"  F1: {optimal_f1:.4f}")
        print(f"  Recall: {optimal_recall:.4f}")
        print(f"  Precision: {optimal_precision:.4f}")

        fold_results.append(best_auc)
        all_best_thresholds.append(best_threshold_f1)

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", marker='o')
        plt.plot(val_losses, label="Val Loss", marker='o')
        plt.axvline(best_epoch, color='r', linestyle='--',
                    label=f"Best AUC Epoch {best_epoch + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold + 1} Loss Curve (Best AUC={best_auc:.4f})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"checkpoints/{prefix}_fold_{fold + 1}_best_loss.png")
        plt.close()

    print("\n======== Final Result ========")
    print(f"Mean AUC across folds: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"Mean optimal threshold: {np.mean(all_best_thresholds):.4f} (+/- {np.std(all_best_thresholds):.4f})")

    with open(f"checkpoints/{prefix}_summary.txt", 'w') as f:
        f.write(f"Mean AUC: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}\n")
        f.write(f"Mean optimal threshold: {np.mean(all_best_thresholds):.4f} +/- {np.std(all_best_thresholds):.4f}\n")
        f.write(f"Per fold thresholds: {all_best_thresholds}\n")


if __name__ == "__main__":
    run_training("../datasets/train_full/")