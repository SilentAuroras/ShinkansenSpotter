import argparse
import shutil
import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# Labels for relabeling YOLO labels based on the folder the images are in
CLASS_MAP = {
    "N700": 0,
    "E5": 1,
    "E6": 2,
}

# Map YOLO detections of train = class_id = 6 to correct id based on folder
def relabel_training():
    for cls_name, class_id in CLASS_MAP.items():

        # Debug
        directory_path = f"training-data/labels/training/{cls_name}"
        print(f"Relabeling training labels for: {cls_name} : {class_id}")

        # Iterate over folders and relabel based on folder membership
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                # Open existing label
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = []

                # Replace train with labels based on folder
                for line in lines:
                    if line.startswith("6"):

                        # replace leading 6 with new class_id
                        new_lines.append(str(class_id) + line[1:])

                # Overwrite file with filtered & relabeled lines
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

# Generate YOLO label files
def label_images():

    # Explicit define
    global auto_label_results

    # Debug
    print("[*] Auto generating labels using YOLO11 base model")
    model = YOLO('yolo11x.pt')

    # Set folders - label training and validation data
    input_dir = "training-data/images/training"
    output_dir = "training-data/labels/training"

    # Clear out label folder so doesn't create N7002 or N7003
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Loop over folders
    for root, dirs, files in os.walk(input_dir):
        for sub_dir in dirs:
            sub_dir_path = os.path.join(root, sub_dir)

            # Train Model
            print(f"[*] Model auto-labeling for: {sub_dir}")
            auto_label_results = model.predict(
                source=sub_dir_path,
                save_txt=True,
                save_conf=False,
                imgsz=640,
                project=output_dir,
                name=sub_dir,
                val = False,
            )

            # Move labels up one folder to fix folder nesting
            labels_src = os.path.join(output_dir, sub_dir, "labels")
            labels_dst = os.path.join(output_dir, sub_dir)
            if os.path.exists(labels_src):
                for f in os.listdir(labels_src):
                    shutil.move(
                        os.path.join(labels_src, f),
                        os.path.join(labels_dst, f)
                    )
                os.rmdir(labels_src)

    # Debug auto-labeling testing, show bounding boxes on YOLO pass 1
    # for test in auto_label_results:
    #    test.show()

    # Debug
    print("[+] Done auto labeling images")

# Normalize image sizes for YOLO
def normalize_images():

    # Iterate through raw folder directories
    folder = "training-data/images/raw/"
    for root, dirs, files in os.walk(folder):

        # Loop through raw images sub folders (ex. N700)
        for sub in dirs:

            # Create normalized directory for cleaned images
            cleaned_dir = f"training-data/images/training/{sub}/"
            if not os.path.exists(cleaned_dir):
                os.mkdir(cleaned_dir)

            # Normalize image sizes
            print(f"[*] Normalizing folder: {sub}...")
            for sub_root, sub_dirs, sub_files in os.walk(root):
                for file in sub_files:

                    # Normalize file size
                    img = cv2.imread(f"{folder}{sub}/{file}")

                    # Skip if error
                    if img is None:
                        print(f"[-] Error reading image: {img}")
                        pass

                    # Resize
                    resized = cv2.resize(img, (640, 640))

                    # Write new file to cleansed directory
                    cv2.imwrite(f"{cleaned_dir}/{file}", resized)

    # Set implicit none
    print("[+] Normalization done...")
    return None

# Training function
def train(gpu):

    # No model exists, create and train a new one
    print("[*] Training...")

    # Use existing base model
    model = YOLO('yolo11x.pt')

    # Train model off of shinkansen images and classification
    if not gpu:
        # CPU Training
        model.train(
            # Training settings
            data='data.yaml',
            epochs=5,
            imgsz=640,
            name='shinkansen',
            val=False,
        )

    elif gpu:
        # GPU Training
        model.train(
            # Training settings
            data='data.yaml',
            imgsz = 640,
            name = 'shinkansen',
            val = False,
            # More epochs since using GPU
            epochs = 30,
            # Cuda GPU 0
            device = 0,
            # Set GPU load
            batch = 2,
        )

    # Debug
    print("[+] Custom training done")

    # Save best model to the root directory
    runs_dir = Path("runs")
    runs = list(runs_dir.rglob("best.pt"))

    # Check if runs exists
    if not runs:
        raise FileNotFoundError("[-] No best.pt found")

    # Choose most recently modified best.pt to shinkansen.pt
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    src = runs[0]
    dst = Path.cwd() / "shinkansen.pt"
    shutil.copy2(src, dst)

    # Copy best.pt to root for retraining
    print(f"[*] Copied `{src}` -> `{dst}`")

def classify(model, file, filetype):

    # Debug
    print("[*] Classifying...")

    # Set model
    model = YOLO(model)

    # Classify using trained shinkansen model
    if filetype == "image":
        output = model.predict(
            source=file,
            imgsz = 640,
            # Set confidence filters to limit boxes
            conf = 0.7,
            iou = 0.7,
        )

        # Return results only, display separately
        return output

    elif filetype == "video":

        # CV2 Read Video File
        capture = cv2.VideoCapture(file)

        # Loop through video frames
        while True:

            # Read the frame
            success,frame = capture.read()

            # If cant read frame or at the end, stop
            if not success:
                break

            # Use model to predict frame
            results = model.predict(
                source = frame,
                imgsz=640,
                conf=0.4,
                iou=0.1,
                device=0,
            )

            # Show the detection on the frame
            annotated = results[0].plot()
            cv2.imshow("detected", annotated)

            # Add a small delay to allow OpenCV to render GUI
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        # Release
        capture.release()
        cv2.destroyAllWindows()

    # Implicit return
    return None

if __name__ == "__main__":

    # Setup argparse
    parser = argparse.ArgumentParser(description="Shinkansen YOLO Detection")
    parser.add_argument("-f", "--fresh", required=False, help="Fresh run")
    parser.add_argument("-i", "--image", required=False, help="path to input image")
    parser.add_argument("-x", "--video", required=False, help="path to input video")
    parser.add_argument("-g", "--gpu", required=False, help="Enable GPU training", default=True)
    parser.add_argument("-m", "--model", required=False, help="path to model", default="shinkansen.pt")
    parser.add_argument("-n", "--normalize", required=False, help="perform image size normalization", default=True)
    args = parser.parse_args()

    # Parse argparse
    if not args.image or not args.video:
        parser.print_help()
        # sys.exit(1)

    # Fresh run, delete models and labels
    if args.fresh is not None:
        print("[*] Removing old models and labels")

        # Delete models
        if os.path.exists("shinkansen.pt"):
            os.remove("shinkansen.pt")

        # Delete yolo model
        if os.path.exists("yolo11x.pt"):
            os.remove("yolo11x.pt")

        # Delete labels and cache
        if os.path.exists("training-data/labels/training/"):
            os.removedirs("training-data/labels/training/")

    # Training model if not present
    if not os.path.exists(args.model):
        print("[-] No model present, training new model...")

        # Normalize image sizes for YOLO
        normalize_images()

        # Generate YOLO txt labels
        label_images()

        # Remap labels to match folder currently in
        relabel_training()

        # Train using custom data
        train(args.gpu)

    # If no source provided use default
    if (args.image is None) and (args.video is None):
        args.image = "test-n700.png"

    # Show boxes on video or images
    if args.video:
        print("[+] Showing boxes on video...")

        # Use model to detect video
        results = classify(args.model, args.video, "video")

    else:
        print("[+] Showing boxes on image...")

        # Use model to detect image or video
        results = classify(args.model, args.image, "image")
        for result in results:
            result.show()