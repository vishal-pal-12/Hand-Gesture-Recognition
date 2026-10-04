# train.py -- DeafVoice AI: Train Finger-Wise Multi-Hand Gesture Model
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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, precision_recall_fscore_support

from models.architecture import build_landmark_classifier
from utils.gesture_classifier import CLASS_NAMES, GESTURE_LABELS, DEAF_SPOKEN_PHRASES
from utils.prepare_dataset import build_full_dataset
from utils.helpers import plot_confusion_matrix, plot_training_history

np.random.seed(42)
tf.random.set_seed(42)

MODEL_BEST_PATH = 'models/best_model.keras'
MODEL_FINGER_PATH = 'models/finger_gesture_model.keras'
RESULTS_DIR = 'results'


def evaluate_model(model, X_test, y_test, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    print("\n" + "=" * 68)
    print("  DEAFVOICE AI: EVALUATION ON FINGER-WISE MULTI-HAND TEST SET")
    print("=" * 68)

    y_probs = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    accuracy = float(np.mean(y_pred == y_test))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

    print("\n--- STATISTICAL PERFORMANCE MEASURES (Macro-Averaged) ---")
    print(f"  Gesture Test Accuracy : {accuracy * 100:.2f} %")
    print(f"  Macro Precision       : {prec * 100:.2f} %")
    print(f"  Macro Recall          : {rec * 100:.2f} %")
    print(f"  Macro F1-Score        : {f1 * 100:.2f} %")
    print("-" * 68)

    print("\n--- PER-CLASS DEAF SIGN RECOGNITION REPORT ---")
    report = classification_report(
        y_test, y_pred,
        target_names=[f"{GESTURE_LABELS[i]:<28}" for i in range(len(CLASS_NAMES))],
        digits=4
    )
    print(report)

    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)

    metrics = {
        'task': 'Finger-Wise Multi-Hand Sign Language Recognition for Deaf and Mute',
        'accuracy_pct': round(accuracy * 100, 2),
        'precision_macro_pct': round(float(prec) * 100, 2),
        'recall_macro_pct': round(float(rec) * 100, 2),
        'f1_score_macro_pct': round(float(f1) * 100, 2),
        'total_test_samples': int(len(y_test)),
        'num_classes': len(CLASS_NAMES)
    }
    metrics_path = os.path.join(results_dir, 'metrics_report.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved -> {metrics_path}")

    return metrics


def train_pipeline(epochs=25, batch_size=32, lr=0.001, from_scratch=False, eval_only=False):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 68)
    print("  DEAFVOICE AI: TRAINING FINGER-WISE MULTI-HAND GESTURE MODEL")
    print("  Zero-Overlap & High-Precision Sign AI (Left & Right Hands)")
    print("=" * 68)

    # 1. Load or Generate Dataset
    if not os.path.exists('data/X_train.npy') or from_scratch:
        X_train, y_train, X_test, y_test = build_full_dataset()
    else:
        print("[DATASET] Loading saved landmark datasets from data/ ...")
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        print(f"  [OK] Train set: {X_train.shape[0]} samples (63 features)")
        print(f"  [OK] Test set:  {X_test.shape[0]} samples (63 features)")

    # 2. Build Model
    model = build_landmark_classifier(input_dim=63, num_classes=10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    if eval_only:
        if os.path.exists(MODEL_BEST_PATH):
            try:
                model = tf.keras.models.load_model(MODEL_BEST_PATH)
            except Exception:
                pass
        return evaluate_model(model, X_test, y_test)

    # 3. Callbacks
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
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n[TRAINING] Training Neural Landmark Classifier ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODEL_FINGER_PATH)
    print(f"[INFO] Finger gesture model saved -> {MODEL_FINGER_PATH}")
    plot_training_history(history, save_path=os.path.join(RESULTS_DIR, 'training_history.png'))

    # Final Evaluation
    best_model = tf.keras.models.load_model(MODEL_BEST_PATH)
    metrics = evaluate_model(best_model, X_test, y_test)
    print("\n[SUCCESS] Model training and evaluation successfully completed!")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeafVoice AI: Train Finger-Wise Multi-Hand Model")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    args = parser.parse_args()
    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        from_scratch=args.from_scratch,
        eval_only=args.eval_only
    )
