import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
matplotlib.use("Agg")


#----------------------
#1. Importing model and getting label
#----------------------
def get_label(image_path, model):
    folder_path = 'AI/OmrPredict/ForStudent/predict'

    # Check if the folder exists
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and all its contents have been deleted.")
        except OSError as e:
            print(f"Error: {e.strerror}")
    else:
        print(f"Folder '{folder_path}' does not exist.")

    output_folder = 'AI/OmrPredict/ForStudent/predict'

    # Run YOLO prediction and save results (image + label)
    results = model.predict(
        image_path,
        conf=0.6,
        save=True,
        save_txt=True,
        project=output_folder,
        name='results',
        exist_ok=True
    )

    # Paths where YOLO saves outputs
    saved_image_folder = Path(f"{output_folder}/results")
    saved_label_folder = saved_image_folder / 'labels'

    saved_labels = list(saved_label_folder.glob('*.txt'))

    if saved_labels:
        label_file_path = saved_labels[0]  
        with open(label_file_path, 'r') as file:
            label_data = file.read()
    else:
        label_data = None
        print("No label files found.")

    return label_data


#----------------------
#2. Show score for each subject
#----------------------

def crop_left_strip(image): 
    height, width = image.shape[:2]
    crop = height - 12
    cropped_image = image[5:crop, :]
    return cropped_image


def detect_filled_bubbles(roi, show=False):
    """Detect and visualize shaded bubbles within a cropped column image."""
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_answers = []

    # Define bubble detection parameters
    min_area = 50
    max_area = 500
    fill_threshold = 0.5

    any_bubble_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            bubble_roi = binary[y:y + h, x:x + w]
            filled_area = cv2.countNonZero(bubble_roi)
            if filled_area / (w * h) > fill_threshold:
                cx, cy = x + w // 2, y + h // 2
                detected_answers.append((cx, cy))
                cv2.circle(roi, (cx, cy), 5, (0, 255, 0), 2)
                any_bubble_detected = True

    if not any_bubble_detected:
        detected_answers.append((0, 0))

    return detected_answers


def convert_to_2d_list(data_str):
    """Convert the string to a 2D list"""
    lines = data_str.strip().split('\n')
    data_list = [list(map(float, line.split())) for line in lines]
    return data_list


def final_answers(image_path, data_str):
    """Process OMR image and return detected answers"""
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    labels = convert_to_2d_list(data_str)

    # Filter class 0 boxes and sort by x_center
    boxes = sorted([label for label in labels if label[0] == 0], key=lambda x: x[1])

    target_width = 95
    target_height = 750

    # Define option ranges
    option_ranges = {
        'A': (0, 20),
        'B': (22, 42),
        'C': (44, 64),
        'D': (66, 100)
    }

    detected_options = []
    for idx, label in enumerate(boxes):
        class_id, center_x, center_y, w, h = label
       
        # Convert normalized coordinates to pixel values
        x_center = int(center_x * width)
        y_center = int(center_y * height)
        box_width = int(w * width)
        box_height = int(h * height)

        # Calculate bounding box corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Extract and resize ROI
        roi = image[y1:y2, x1:x2].copy()
        roi_resize = cv2.resize(roi, (target_width, target_height))
        roi = crop_left_strip(roi_resize)

        section_height = roi.shape[0] / 50.0

        # Draw horizontal lines
        for j in range(1, 50):
            y_line = int(j * section_height)
            cv2.line(roi, (0, y_line), (roi.shape[1], y_line), (0, 255, 0), 1)

        cv2.line(roi, (0, roi.shape[0] - 1), (roi.shape[1], roi.shape[0] - 1), (0, 255, 0), 1)

        # Process each section
        for j in range(50):
            y_start = int(j * section_height)
            y_end = int((j + 1) * section_height)

            if j == 49:
                y_end = roi.shape[0]

            section = roi[y_start:y_end, :]
            detected_answers = detect_filled_bubbles(section)

            # Map to options
            for cx, cy in detected_answers:
                for option, (min_x, max_x) in option_ranges.items():
                    if min_x <= cx < max_x:
                        detected_options.append(option)
                        break
                else:
                    detected_options.append('0')

    return detected_options


def show_score_for_each_subject(result):
    """Calculate scores for each subject"""
    Chem = {}
    Phy = {}
    Bio = {}
    
    for k, v in enumerate(result):
        s = k + 1
        if s <= 50:
            Chem[s] = v
        elif s <= 100:
            Phy[s] = v
        elif s <= 201:
            Bio[s] = v
    
    return {'Chemistry': Chem, 'Physics': Phy, 'Biology': Bio}


#----------------------
#3. Show the ticked image for confirmation
#----------------------
def detect_filled_bubbles_for_subject(roi, subject_name, show=False):
    output_dir = "AI/OmrPredict/ForStudent/StudentDetectedSubjects"
    os.makedirs(output_dir, exist_ok=True)

    # Safe grayscale conversion
    if len(roi.shape) == 2:
        gray = roi
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_answers = []
    min_area = 50
    max_area = 500
    fill_threshold = 0.5
    any_bubble_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            bubble_roi = binary[y:y+h, x:x+w]
            filled_area = cv2.countNonZero(bubble_roi)

            if filled_area / (w*h) > fill_threshold:
                cx, cy = x + w//2, y + h//2
                detected_answers.append((cx, cy))
                cv2.circle(roi, (cx, cy), 5, (0, 255, 0), 2)
                any_bubble_detected = True

    if not any_bubble_detected:
        detected_answers.append((0, 0))

    if show:
        save_path = f"{output_dir}/{subject_name}.jpg"
        cv2.imwrite(save_path, roi)

    return detected_answers


def check_answers(image_path, data_str):
    output_dir = "AI/OmrPredict/ForStudent/StudentDetectedSubjects"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    labels = convert_to_2d_list(data_str)
    boxes = sorted([label for label in labels if label[0] == 0], key=lambda x: x[1])

    target_width = 95
    target_height = 750
    
    all_detected = []

    for idx, label in enumerate(boxes):
        class_id, cx, cy, w, h = label

        x_center = int(cx * width)
        y_center = int(cy * height)
        bw = int(w * width)
        bh = int(h * height)

        x1 = x_center - bw // 2
        y1 = y_center - bh // 2
        x2 = x_center + bw // 2
        y2 = y_center + bh // 2

        roi = image[y1:y2, x1:x2].copy()
        roi = cv2.resize(roi, (target_width, target_height))
        roi = crop_left_strip(roi)

        subject_name = f"Subject_{idx+1}"

        # detect AND save in one place
        detected = detect_filled_bubbles_for_subject(roi, subject_name, show=True)
        all_detected.append(detected)

    return all_detected


def save_subject_images():
    source_dir = "AI/OmrPredict/ForStudent/StudentDetectedSubjects"
    dest_dir = r"img\subjects"

    os.makedirs(dest_dir, exist_ok=True)

    mapping = {
        "Subject_1.jpg": "chemistry.jpg",
        "Subject_2.jpg": "physics.jpg",
        "Subject_3.jpg": "biology1.jpg",
        "Subject_4.jpg": "biology2.jpg"
    }

    saved_files = {}

    for src_name, dest_name in mapping.items():
        src_path = os.path.join(source_dir, src_name)
        dest_path = os.path.join(dest_dir, dest_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            saved_files[dest_name.rsplit('.', 1)[0]] = dest_path
        else:
            saved_files[dest_name] = None

    return saved_files
#----------------------
# Main processing function for Flask
#----------------------
def process_omr(image_path, model_path=r"OmrModel\rectangleOmrOri_yolo_model.pt"):
    """
    Main function to process OMR sheet for Flask app
    
    Args:
        image_path: Path to the uploaded OMR image
        model_path: Path to the YOLO model
    
    Returns:
        dict: Processing results with answers and scores
    """
    try:
        # Load model
        model = YOLO(model_path)
        
        # Get labels
        labels = get_label(image_path, model)
        
        if labels is None:
            return {
                'success': False,
                'error': 'No labels detected in the image'
            }
        
        # Get detected answers
        answers = final_answers(image_path, labels)
        
        # Calculate subject scores
        subject_scores = show_score_for_each_subject(answers)
        
        # Generate confirmation images
        check_answers(image_path, labels)
        subject_images = save_subject_images()
        
        return {
            'success': True,
            'answers': answers,
            'subject_scores': subject_scores,
            'total_answers': len(answers),
            'subject_images': subject_images
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }