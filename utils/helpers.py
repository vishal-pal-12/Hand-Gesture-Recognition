"""
============================================================
  utils/helpers.py  --  Shared Utilities & Visualizer
  DeafVoice AI: Sign Language Communicator for the Deaf & Mute
  Finger-Wise Multi-Hand Gesture System
============================================================
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.gesture_classifier import (
    CLASS_NAMES,
    GESTURE_LABELS,
    DEAF_SPOKEN_PHRASES,
    GESTURE_COLORS,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)

IDX_TO_LABEL = {i: label for i, label in enumerate(GESTURE_LABELS)}
CLASS_TO_LABEL = {name: label for name, label in zip(CLASS_NAMES, GESTURE_LABELS)}
DEAF_ASSIST_TRANSLATIONS = DEAF_SPOKEN_PHRASES


def safe_load_model(model_path):
    """Safely loads a Keras trained model file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Loaded trained DeafVoice model successfully from {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_path}: {e}")
        raise


def plot_confusion_matrix(y_true, y_pred, save_path=None, title='Deaf Sign Confusion Matrix'):
    """Generates and saves a high-res confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    short_labels = ["HELLO", "NO", "YES", "FINE", "THANK U", "HELP", "WATER", "PERFECT", "DOCTOR", "LOVE"]

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_labels,
                yticklabels=short_labels)
    plt.title(title, fontsize=13, pad=12, fontweight='bold')
    plt.ylabel('True Sign Label', fontsize=11)
    plt.xlabel('Predicted Sign Label', fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[INFO] Confusion matrix heatmap saved -> {save_path}")
    else:
        plt.show()


def plot_training_history(history, save_path=None):
    """Plots training & validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    ax1.plot(epochs_range, acc, 'b-', label='Train Accuracy', lw=2)
    if val_acc:
        ax1.plot(epochs_range, val_acc, 'r--', label='Validation Accuracy', lw=2)
    ax1.set_title('Deaf Sign Recognition Accuracy', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, loss, 'b-', label='Train Loss', lw=2)
    if val_loss:
        ax2.plot(epochs_range, val_loss, 'r--', label='Validation Loss', lw=2)
    ax2.set_title('Loss Convergence', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[INFO] Training history graph saved -> {save_path}")
    else:
        plt.show()
