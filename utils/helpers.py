"""
============================================================
  utils/helpers.py  --  Shared Utilities & Metrics Visualizer
  Hand Gesture Recognition using Deep CNN
  Procedia Computer Science 171 (2020) 2353–2361
============================================================
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Deterministic class mapping for 10 NUS static hand postures
CLASS_NAMES = [
    'posture_a',
    'posture_b',
    'posture_c',
    'posture_d',
    'posture_e',
    'posture_f',
    'posture_g',
    'posture_h',
    'posture_i',
    'posture_j'
]

# Dedicated Deaf & Mute Assistive Communication Signs
GESTURE_LABELS = [
    'HELLO (Greeting)',
    'NO (Stop / Disagree)',
    'YES (Agree / Confirm)',
    'THANK YOU (Gratitude)',
    'HELP (Need Assistance)',
    'WATER (Thirsty / Drink)',
    'GOOD (Understood / Fine)',
    'PERFECT (All Good / Okay)',
    'DOCTOR (Medical / Pain)',
    'I LOVE YOU (Goodbye / Affection)'
]

# Spoken voice phrases for hearing listeners
DEAF_SPOKEN_PHRASES = [
    "Hello, Nice to meet you",
    "No, Please stop",
    "Yes, I agree and understand",
    "Thank you very much",
    "Please help me, I need assistance",
    "I need water to drink, please",
    "I am good, I understand",
    "Everything is perfect and all good",
    "I need a doctor or medical attention",
    "I love you, Goodbye"
]

DEAF_ASSIST_TRANSLATIONS = DEAF_SPOKEN_PHRASES

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {i: label for i, label in enumerate(GESTURE_LABELS)}
CLASS_TO_LABEL = {name: label for name, label in zip(CLASS_NAMES, GESTURE_LABELS)}

# BGR color palette for 10 distinct gestures
GESTURE_COLORS = [
    (240, 160, 40),   # Posture 1: Blue-ish
    (50, 205, 50),    # Posture 2: Lime Green
    (220, 20, 60),    # Posture 3: Crimson
    (255, 140, 0),    # Posture 4: Dark Orange
    (148, 0, 211),    # Posture 5: Dark Violet
    (0, 215, 255),    # Posture 6: Gold / Yellow
    (0, 165, 255),    # Posture 7: Orange
    (255, 105, 180),  # Posture 8: Hot Pink
    (0, 255, 127),    # Posture 9: Spring Green
    (30, 144, 255)    # Posture 10: Dodger Blue
]


# ----------------------------------------------------------
# DATASET LOADING FROM DIRECTORY
# ----------------------------------------------------------
def load_dataset_from_folder(folder_path, img_size=100, max_samples_per_class=None):
    """
    Loads static hand posture images from directory containing class subfolders:
    posture_a through posture_j.

    Images are rescaled to (img_size, img_size, 3) and normalized to [0, 1].

    Returns:
        X: float32 numpy array, shape (N, img_size, img_size, 3)
        y: int32 class indices [0..9], shape (N,)
        filenames: list of relative filepaths
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    X_list, y_list, fnames_list = [], [], []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(folder_path, cls_name)
        if not os.path.exists(cls_dir):
            raise FileNotFoundError(f"Missing expected class folder: {cls_dir}")

        fnames = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        if max_samples_per_class is not None:
            fnames = fnames[:max_samples_per_class]

        for fname in fnames:
            fpath = os.path.join(cls_dir, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[:2] != (img_size, img_size):
                img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

            X_list.append(img_rgb)
            y_list.append(cls_idx)
            fnames_list.append(os.path.join(cls_name, fname))

    X = np.array(X_list, dtype=np.float32) / 255.0
    y = np.array(y_list, dtype=np.int32)
    return X, y, fnames_list


def preprocess_image(image_input, img_size=100):
    """
    Accepts an RGB/BGR numpy array or image file path.
    Preprocesses to shape (1, img_size, img_size, 3) normalized to [0, 1].
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not read image file: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        if len(image_input.shape) == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 3:
            rgb = image_input.copy()
        else:
            raise ValueError(f"Unexpected image shape: {image_input.shape}")

    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    return np.expand_dims(norm, axis=0)


# ----------------------------------------------------------
# VISUALIZATION & REPORTING
# ----------------------------------------------------------
def plot_class_distribution(y_train, y_test=None, save_path='results/class_distribution.png'):
    """Generates bar chart showing sample counts per posture class."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CLASS_NAMES))
    w = 0.35 if y_test is not None else 0.5

    counts_train = [np.sum(y_train == i) for i in range(len(CLASS_NAMES))]
    rects1 = ax.bar(x - (w / 2 if y_test is not None else 0), counts_train, width=w,
                    label='Train Split', color='#2563EB', alpha=0.85, edgecolor='black')

    if y_test is not None:
        counts_test = [np.sum(y_test == i) for i in range(len(CLASS_NAMES))]
        rects2 = ax.bar(x + w / 2, counts_test, width=w,
                        label='Test Split', color='#10B981', alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({GESTURE_LABELS[i].split('(')[-1][:-1]})" for i, c in enumerate(CLASS_NAMES)],
                       fontsize=9)
    ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    ax.set_title('Static Hand Posture Dataset Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Class distribution plot saved -> {save_path}")


def plot_training_history(history, save_path='results/training_history.png'):
    """
    Generates simulation graph showing training & validation accuracy and loss
    matching Figure 4 of the research paper.
    """
    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # Accuracy Plot
    ax1.plot(epochs, [a * 100 for a in history.history['accuracy']],
             color='#0EA5E9', linewidth=2.2, label='Training')
    if 'val_accuracy' in history.history:
        ax1.plot(epochs, [a * 100 for a in history.history['val_accuracy']],
                 color='#1E293B', linestyle='--', linewidth=2.0, marker='o', markersize=4, label='Validation')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_title('Hand Posture CNN Training Convergence (Accuracy & Loss)', fontsize=13, fontweight='bold', pad=10)

    # Loss Plot
    ax2.plot(epochs, history.history['loss'],
             color='#F97316', linewidth=2.2, label='Training')
    if 'val_loss' in history.history:
        ax2.plot(epochs, history.history['val_loss'],
                 color='#1E293B', linestyle='--', linewidth=2.0, marker='o', markersize=4, label='Validation')
    ax2.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Training history graph saved -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """Generates normalized and raw confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(10, 8))
    labels = [f"P{i+1}" for i in range(len(CLASS_NAMES))]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=True,
                linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted Posture Class', fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel('Ground Truth Posture Class', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('Normalized Confusion Matrix — Hand Posture CNN', fontsize=13, fontweight='bold', pad=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved -> {save_path}")


def visualize_sample_predictions(model, X_test, y_test, num_samples=10, save_path='results/sample_predictions.png'):
    """Plots a row of test sample images with their predicted vs true labels."""
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    preds = model.predict(X_test[indices], verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for idx, (img, true_lbl, pred_lbl, prob) in enumerate(zip(X_test[indices], y_test[indices], pred_labels, preds)):
        ax = axes[idx]
        ax.imshow(img)
        is_correct = (true_lbl == pred_lbl)
        color = 'green' if is_correct else 'red'
        confidence = prob[pred_lbl] * 100

        ax.set_title(
            f"True: P{true_lbl+1}\nPred: P{pred_lbl+1} ({confidence:.1f}%)",
            color=color, fontsize=10, fontweight='bold'
        )
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Sample predictions visual saved -> {save_path}")


# ----------------------------------------------------------
# MODEL UTILS
# ----------------------------------------------------------
def safe_load_model(model_path):
    """Safely loads a Keras model handling version differences and compilation."""
    import tensorflow as tf
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"[INFO] Loaded model successfully from {model_path}")
        return model
    except Exception as e:
        print(f"[WARNING] Standard loading failed ({e}). Retrying with compile=True...")
        return tf.keras.models.load_model(model_path)
