# OMR Answer Detection System

A full-stack Optical Mark Recognition (OMR) system designed to detect, extract, and evaluate student responses from scanned OMR sheets. The project uses a custom-trained YOLO model for bubble detection, OpenCV for image processing, and a Flask backend with a modern frontend interface.

---

## 🚀 Features

### **1. OMR Sheet Upload & Processing**
- Upload scanned OMR sheets (JPG/PNG).
- Automatic detection of answer regions using YOLO.
- Extraction and classification of filled bubbles.

### **2. Subject-wise Answer Segmentation**
- Automatically crops and saves subject regions:
  - Chemistry
  - Physics
  - Biology 1
  - Biology 2
- Saves subject images to `uploads/subjects/` for retrieval.

### **3. Full Web Interface**
- Built with **Flask backend** + HTML/CSS/JS frontend.
- Displays the uploaded OMR sheet.
- Shows processed subject answer images.
- Returns detected answers and scores.

### **4. Clean Folder Management**
- System cleans and regenerates `StudentDetectedSubjects` for each request.
- Prevents accumulation of old detections.

---

## 🖼️ Project Structure
```
OMR_ANSWER_DETECTION/
│
├── AI/
│   └── OmrPredict/ForStudent/StudentDetectedSubjects/   # Auto-generated subjects
│
├── OmrModel/                                            # YOLO model files
├── images/                                              # Static assets
├── img/                                                 # Additional images
├── templates/
│   └── index.html                                       # Frontend UI
│
├── uploads/                                             # Uploaded sheets + subject images
│   └── subjects/                                        # chemistry.jpg, physics.jpg...
│
├── omr_processor.py                                     # Main OMR processing logic
├── web_app.py                                           # Flask app
├── requirements.txt
└── README.md
```

---

## 📸 Sample Images
You may insert sample subject images like this:

```md
### Chemistry Extraction
![Chemistry](uploads/subjects/chemistry.jpg)


### Physics Extraction
![Physics](uploads/subjects/physics.jpg)
```

---

## 🔧 Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/Babarinde1/OMR_ANSWER_DETECTION.git
cd OMR_ANSWER_DETECTION
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Flask server**
```bash
python web_app.py
```

Open your browser at:
```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

### **1. YOLO Detection**
Detects all answer boxes using class `0`.

### **2. ROI Extraction**
Crops specific vertical slices for each subject.

### **3. Bubble Analysis**
Detects filled versus empty bubbles based on pixel intensity.

### **4. Result Output**
Returns:
- Answers
- Subject-wise scores
- Total answers
- Subject preview images

---

## 📁 API Endpoints

### **POST /process_omr**
Upload and process OMR sheet.

### **GET /uploads/<filename>**
Returns uploaded OMR sheet.

### **GET /subjects/<filename>**
Returns cropped subject images.

---

## 🙌 Contributions
Pull requests are welcome! For major updates, please open an issue first.

---

## 📜 License
This project is licensed under the MIT License.
