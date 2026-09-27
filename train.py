# train.py -- DeafVoice AI: Training Sign Language Model for the Deaf & Mute
import os
import sys

# Auto-redirect to project venv python if not already running in it
project_root = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(project_root, 'venv', 'Scripts', 'python.exe')
if os.path.exists(venv_py) and os.path.normcase(sys.executable) != os.path.normcase(venv_py):
    import subprocess
    res = subprocess.run([venv_py] + sys.argv, cwd=project_root)
    sys.exit(res.returncode)

import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, precision_recall_fscore_support

from models.architecture import build_paper_cnn
from utils.helpers import (
    CLASS_NAMES,
    GESTURE_LABELS,
    DEAF_SPOKEN_PHRASES,
    CLASS_TO_LABEL,
    load_dataset_from_folder,
    plot_class_distribution,
    plot_training_history,
    plot_confusion_matrix,
    visualize_sample_predictions,
    safe_load_model
)

np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = 100
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001
DEFAULT_MOMENTUM = 0.90
NUM_CLASSES = 10

MODEL_BEST_PATH = 'models/best_model.keras'
MODEL_FINAL_PATH = 'models/hand_posture_cnn_final.keras'
RESULTS_DIR = 'results'


def evaluate_model(model, X_test, y_test, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    print("\n" + "=" * 65)
    print("  DEAFVOICE AI: MODEL EVALUATION ON DEAF SIGN LANGUAGE TEST DATA")
    print("=" * 65)

    y_probs = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    accuracy = float(np.mean(y_pred == y_test))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

    print("\n--- STATISTICAL PERFORMANCE MEASURES (Macro-Averaged) ---")
    print(f"  Deaf Sign Accuracy  : {accuracy * 100:.2f} %")
    print(f"  Macro Precision     : {prec * 100:.2f} %")
    print(f"  Macro Recall        : {rec * 100:.2f} %")
    print(f"  Macro F1-Score      : {f1 * 100:.2f} %")
    print("-" * 65)

    print("\n--- PER-CLASS SIGN LANGUAGE TRANSLATION REPORT ---")
    report = classification_report(
        y_test, y_pred,
        target_names=[f"{GESTURE_LABELS[i]:<32}" for i in range(len(CLASS_NAMES))],
        digits=4
    )
    print(report)

    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)

    samples_path = os.path.join(results_dir, 'sample_predictions.png')
    visualize_sample_predictions(model, X_test, y_test, num_samples=10, save_path=samples_path)

    metrics = {
        'task': 'Deaf and Mute Assistive Sign Language Recognition',
        'accuracy_pct': round(accuracy * 100, 2),
        'precision_macro_pct': round(float(prec) * 100, 2),
        'recall_macro_pct': round(float(rec) * 100, 2),
        'f1_score_macro_pct': round(float(f1) * 100, 2),
        'total_test_samples': int(len(y_test)),
        'num_sign_categories': NUM_CLASSES
    }
    metrics_path = os.path.join(results_dir, 'metrics_report.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Detailed metrics JSON saved -> {metrics_path}")

    return metrics


def train_pipeline(epochs=DEFAULT_EPOCHS,
                   batch_size=DEFAULT_BATCH_SIZE,
                   lr=DEFAULT_LR,
                   momentum=DEFAULT_MOMENTUM,
                   optimizer_type='adam',
                   from_scratch=False,
                   eval_only=False):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 65)
    print("  DEAFVOICE AI: TRAINING SIGN-TO-SPEECH MODEL FOR THE DEAF")
    print("  Empowering Deaf & Speech-Impaired People with Deep CNN")
    print("  10 Core Communication Signs: HELLO, NO, YES, THANK YOU, HELP, etc.")
    print(f"  Optimizer: {optimizer_type.upper()} (lr={lr}) | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 65)

    train_dir = 'data/processed/train'
    test_dir = 'data/processed/test'

    print("\n[1/5] Loading Deaf Sign Language datasets from disk ...")
    X_train, y_train = load_dataset_from_folder(train_dir, img_size=IMG_SIZE)
    X_test, y_test = load_dataset_from_folder(test_dir, img_size=IMG_SIZE)

    plot_class_distribution(y_train, y_test, save_path=os.path.join(RESULTS_DIR, 'class_distribution.png'))

    if eval_only:
        if os.path.exists(MODEL_BEST_PATH):
            model = safe_load_model(MODEL_BEST_PATH)
        elif os.path.exists(MODEL_FINAL_PATH):
            model = safe_load_model(MODEL_FINAL_PATH)
        else:
            raise FileNotFoundError("No trained model checkpoint found for evaluation.")
        return evaluate_model(model, X_test, y_test)

    print("\n[2/5] Initializing Deep CNN Architecture ...")
    model = build_paper_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.summary()

    if optimizer_type.lower() == 'sgdm':
        optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=False)
    else:
        optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n[3/5] Setting up Data Augmentation & Callbacks ...")
    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_BEST_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n[4/5] Training Deaf Sign Recognition Deep CNN ...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODEL_FINAL_PATH)
    print(f"\n[INFO] Final model saved -> {MODEL_FINAL_PATH}")
    plot_training_history(history, save_path=os.path.join(RESULTS_DIR, 'training_curves.png'))

    print("\n[5/5] Performing Final Evaluation on Test Signs ...")
    best_model = safe_load_model(MODEL_BEST_PATH)
    metrics = evaluate_model(best_model, X_test, y_test)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeafVoice AI: Train Sign Language CNN Model")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--momentum', type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgdm'])
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    args = parser.parse_args()
    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        optimizer_type=args.optimizer,
        from_scratch=args.from_scratch,
        eval_only=args.eval_only
    )
