"""
Visual Diagnostics Module.
Generates publication-quality figures for learning curves, ROC-AUC, confusion matrices, and score distributions.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


class Visualizer:
    """Diagnostic visualization suite for genomic machine learning models."""

    def __init__(self, output_dir: Optional[str] = "assets"):
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def plot_training_history(self, history, show: bool = True, save: bool = True):
        """Plots training and validation loss and accuracy trajectories."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], label='Training Loss', color='#2563EB', lw=2.2)
        ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', color='#DC2626', lw=2.2, linestyle='--')
        ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.plot(epochs, [a * 100 if a <= 1.0 else a for a in history.history['accuracy']],
                 label='Training Accuracy', color='#059669', lw=2.2)
        ax2.plot(epochs, [a * 100 if a <= 1.0 else a for a in history.history['val_accuracy']],
                 label='Validation Accuracy', color='#D97706', lw=2.2, linestyle='--')
        ax2.set_title('Classification Accuracy (%) vs. Epochs', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        if save and self.output_dir:
            filepath = os.path.join(self.output_dir, 'training_history.png')
            plt.savefig(filepath, bbox_inches='tight')
            print(f"[+] Saved figure to {filepath}")
        if show:
            plt.show()
        plt.close()

    def plot_predictions_distribution(self, predictions: np.ndarray, y_true: Optional[np.ndarray] = None, show: bool = True, save: bool = True):
        """Plots probability distribution histogram of pathogenicity scores."""
        plt.figure(figsize=(8, 5), dpi=300)
        preds_flat = np.array(predictions).flatten()

        if y_true is not None:
            y_arr = np.array(y_true).flatten()
            sns.histplot(preds_flat[y_arr == 0], color='#10B981', label='Benign Variants', bins=35, kde=True, alpha=0.55)
            sns.histplot(preds_flat[y_arr == 1], color='#EF4444', label='Pathogenic Variants', bins=35, kde=True, alpha=0.55)
        else:
            sns.histplot(preds_flat, bins=40, kde=True, color='#8B5CF6')

        plt.axvline(0.5, color='#1F2937', linestyle='--', lw=1.8, label='Decision Boundary (0.50)')
        plt.title('Pathogenicity Probability Score Distribution', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Predicted Probability $P(Y=1)$', fontsize=11, fontweight='semibold')
        plt.ylabel('Variant Count', fontsize=11, fontweight='semibold')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        if save and self.output_dir:
            filepath = os.path.join(self.output_dir, 'predictions_distribution.png')
            plt.savefig(filepath, bbox_inches='tight')
            print(f"[+] Saved figure to {filepath}")
        if show:
            plt.show()
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, show: bool = True, save: bool = True):
        """Plots Receiver Operating Characteristic (ROC) curve with AUC metric."""
        fpr, tpr, _ = roc_curve(np.array(y_true).flatten(), np.array(y_pred).flatten())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6), dpi=300)
        plt.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.8, linestyle='--', label='Random Chance (AUC = 0.500)')
        plt.fill_between(fpr, tpr, alpha=0.15, color='#2563EB')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='semibold')
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='semibold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        if save and self.output_dir:
            filepath = os.path.join(self.output_dir, 'roc_curve.png')
            plt.savefig(filepath, bbox_inches='tight')
            print(f"[+] Saved figure to {filepath}")
        if show:
            plt.show()
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, show: bool = True, save: bool = True):
        """Plots confusion matrix heatmap."""
        preds = (np.array(y_pred).flatten() >= 0.5).astype(int)
        cm = confusion_matrix(np.array(y_true).flatten(), preds)

        plt.figure(figsize=(6, 5), dpi=300)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Benign (0)', 'Pathogenic (1)'],
                    yticklabels=['Benign (0)', 'Pathogenic (1)'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        plt.title('Confusion Matrix (Test Cohort)', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Predicted Clinical Label', fontsize=11, fontweight='semibold')
        plt.ylabel('True Ground Truth Label', fontsize=11, fontweight='semibold')

        plt.tight_layout()
        if save and self.output_dir:
            filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(filepath, bbox_inches='tight')
            print(f"[+] Saved figure to {filepath}")
        if show:
            plt.show()
        plt.close()
