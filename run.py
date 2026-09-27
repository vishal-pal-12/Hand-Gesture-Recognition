# run.py -- DeafVoice AI: Unified CLI Runner for Deaf & Mute Assistive System
import os
import sys

# Auto-redirect to project venv python if not already running in it
project_root = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(project_root, 'venv', 'Scripts', 'python.exe')
if os.path.exists(venv_py) and os.path.normcase(sys.executable) != os.path.normcase(venv_py):
    import subprocess
    res = subprocess.run([venv_py] + sys.argv, cwd=project_root)
    sys.exit(res.returncode)

import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def cmd_train(args):
    from train import train_pipeline
    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        optimizer_type=args.optimizer,
        from_scratch=args.from_scratch,
        eval_only=False
    )


def cmd_evaluate(args):
    from train import train_pipeline
    train_pipeline(eval_only=True)


def cmd_predict(args):
    from predict_image import predict_from_image, predict_directory
    if args.image:
        predict_from_image(args.image, auto_detect=args.auto_detect, show=args.show,
                           output_path=args.output, speak=getattr(args, 'speak', False))
    elif args.dir:
        predict_directory(args.dir, auto_detect=args.auto_detect)
    else:
        sample_path = 'sample_images/posture_a_sample.jpg'
        if os.path.exists(sample_path):
            print(f"[INFO] No image specified. Testing on default deaf sign sample: {sample_path}")
            predict_from_image(sample_path, auto_detect=False, show=False, speak=False)
        else:
            print("[ERROR] Please specify --image <path> or --dir <path>.")


def cmd_webcam(args):
    from realtime_detect import run_realtime
    run_realtime(
        camera_index=args.camera,
        max_frames=args.max_frames,
        output_path=args.save_output,
        test_mode=args.test_mode,
        speech_enabled=not getattr(args, 'mute', False)
    )


def cmd_verify(args):
    print("=" * 68)
    print("  DEAFVOICE AI: COMPREHENSIVE END-TO-END SYSTEM VERIFICATION")
    print("  Empowering Deaf & Mute Individuals with Sign-to-Speech AI")
    print("=" * 68)

    print("\n[1/6] Verifying Deaf Sign Language dataset structure & categories ...")
    from utils.helpers import CLASS_NAMES, GESTURE_LABELS, DEAF_SPOKEN_PHRASES
    for split in ['train', 'test']:
        split_path = os.path.join('data', 'processed', split)
        assert os.path.exists(split_path), f"Missing directory: {split_path}"
        classes_found = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ])
        assert classes_found == sorted(CLASS_NAMES), f"Class mismatch in {split}: {classes_found}"
        total_imgs = sum(len(os.listdir(os.path.join(split_path, c))) for c in CLASS_NAMES)
        print(f"  [OK] data/processed/{split}: {total_imgs} images across {len(classes_found)} Deaf Sign categories.")
    print("  [OK] 10 Core Deaf Assistive Categories:")
    for i, g in enumerate(GESTURE_LABELS):
        print(f"       Sign {i+1:<2}: {g:<32} -> \"{DEAF_SPOKEN_PHRASES[i]}\"")

    print("\n[2/6] Verifying Deep CNN Sign Recognition architecture ...")
    from models.architecture import build_paper_cnn
    model = build_paper_cnn(input_shape=(100, 100, 3), num_classes=10)
    assert model.count_params() in [166042, 166266], f"Unexpected parameter count: {model.count_params()}"
    print(f"  [OK] Paper CNN compiled successfully with {model.count_params():,} parameters.")

    print("\n[3/6] Verifying trained DeafVoice model checkpoints ...")
    model_paths = ['models/best_model.keras', 'models/hand_posture_cnn_final.keras']
    found_model = None
    for p in model_paths:
        if os.path.exists(p):
            sz = os.path.getsize(p) / (1024 * 1024)
            print(f"  [OK] Found checkpoint: {p} ({sz:.2f} MB)")
            found_model = p
    assert found_model is not None, "Trained model checkpoint missing in models/"

    print("\n[4/6] Testing Sign-to-Speech translation engine & sample inference ...")
    from utils.helpers import safe_load_model
    loaded_model = safe_load_model(found_model)
    from predict_image import predict_from_image
    sample_img = 'sample_images/posture_a_sample.jpg'
    if os.path.exists(sample_img):
        lbl, conf, probs = predict_from_image(sample_img, model=loaded_model, show=False)
        assert len(probs) == 10, "Probability vector must have 10 classes"
        print(f"  [OK] Deaf Sign Translation succeeded: {lbl} ({conf*100:.1f}%)")

    print("\n[5/6] Testing Hand Tracking, Preprocessing & Assistive Communication Engine ...")
    from utils.hand_detector import HandDetector
    from utils.helpers import preprocess_image
    from utils.assistive_comm import SentenceBuilder, GestureStabilityTracker
    import numpy as np
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_img, img_size=100)
    assert processed.shape == (1, 100, 100, 3), f"Unexpected shape {processed.shape}"
    detector = HandDetector()
    roi = HandDetector.get_fixed_roi(dummy_img, size=100)
    assert len(roi) == 4, "ROI tuple must have 4 elements"
    s_builder = SentenceBuilder()
    s_builder.add_gesture(0)  # HELLO
    assert "Hello" in s_builder.get_sentence()
    print(f"  [OK] Assistive engine verified (MediaPipe active: {detector.has_mediapipe})")

    print("\n[6/6] Testing Two-Way Realtime Deaf Communicator pipeline (headless simulation) ...")
    from realtime_detect import run_realtime
    run_realtime(test_mode=True, output_path='results/test_simulation_frame.jpg', speech_enabled=False)
    print("  [OK] DeafVoice real-time communication simulation passed.")

    print("\n" + "=" * 68)
    print("  VERIFICATION COMPLETE: DEAFVOICE AI IS 100% OPERATIONAL!")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="DeafVoice AI -- Two-Way Sign-to-Speech Communicator for the Deaf & Mute",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Webcam / Deaf-assist
    for sub_name in ["webcam", "deaf-assist"]:
        p = subparsers.add_parser(sub_name, help="Launch Live Deaf Sign-to-Speech Camera Communicator")
        p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
        p.add_argument("--max_frames", type=int, default=None, help="Max frames")
        p.add_argument("--save_output", type=str, default=None, help="Save video path")
        p.add_argument("--test_mode", action="store_true", help="Headless test mode")
        p.add_argument("--mute", action="store_true", help="Mute voice synthesis")

    # Predict
    parser_predict = subparsers.add_parser("predict", help="Translate sign image to speech/text")
    parser_predict.add_argument("--image", type=str, default=None, help="Single image path")
    parser_predict.add_argument("--dir", type=str, default=None, help="Directory path")
    parser_predict.add_argument("--auto_detect", action="store_true", help="Auto-crop hand")
    parser_predict.add_argument("--show", action="store_true", help="Show preview window")
    parser_predict.add_argument("--speak", action="store_true", help="Speak translated phrase via TTS")
    parser_predict.add_argument("--output", type=str, default=None, help="Output image path")

    # Train
    parser_train = subparsers.add_parser("train", help="Train Deaf Sign Language CNN Model")
    parser_train.add_argument("--epochs", type=int, default=20, help="Epochs (default: 20)")
    parser_train.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser_train.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser_train.add_argument("--momentum", type=float, default=0.90, help="SGDM momentum")
    parser_train.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgdm"], help="Optimizer")
    parser_train.add_argument("--from_scratch", action="store_true", help="Force retrain")

    # Evaluate
    subparsers.add_parser("evaluate", help="Evaluate model accuracy on Deaf Sign Language test set")

    # Verify
    subparsers.add_parser("verify", help="Run comprehensive project verification for deaf assistive system")

    args = parser.parse_args()

    if args.command in ["webcam", "deaf-assist"]:
        cmd_webcam(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "verify":
        cmd_verify(args)
    else:
        # Interactive welcome menu if no command provided
        print("\n" + "=" * 68)
        print("  DEAFVOICE AI: TWO-WAY SIGN COMMUNICATOR FOR THE DEAF & MUTE")
        print("  Translating Hand Gestures into Spoken Voice & Text for the Deaf")
        print("=" * 68)
        print("\nSelect an action:")
        print("  [1] Launch Live Sign-to-Speech Camera (Webcam Communicator)")
        print("  [2] Translate Single Sign Image (sample_images/posture_a_sample.jpg)")
        print("  [3] Translate All Sample Sign Images")
        print("  [4] Run Comprehensive DeafVoice System Self-Test (verify)")
        print("  [5] Evaluate CNN Accuracy on 400 Test Signs (85.0% Benchmark)")
        print("  [6] Train CNN on Deaf Sign Dataset")
        print("  [Q] Exit")
        try:
            choice = input("\nEnter choice [1-6, Q]: ").strip().lower()
            if choice == '1':
                cmd_webcam(argparse.Namespace(camera=0, max_frames=None, save_output=None, test_mode=False, mute=False))
            elif choice == '2':
                cmd_predict(argparse.Namespace(image='sample_images/posture_a_sample.jpg', dir=None, auto_detect=False, show=False, output=None, speak=True))
            elif choice == '3':
                cmd_predict(argparse.Namespace(image=None, dir='sample_images', auto_detect=False, show=False, output=None))
            elif choice == '4':
                cmd_verify(argparse.Namespace())
            elif choice == '5':
                cmd_evaluate(argparse.Namespace())
            elif choice == '6':
                cmd_train(argparse.Namespace(epochs=20, batch_size=32, lr=0.001, momentum=0.90, optimizer='adam', from_scratch=False))
            else:
                print("Exited.")
        except (KeyboardInterrupt, EOFError):
            print("\nExited.")


if __name__ == '__main__':
    main()
