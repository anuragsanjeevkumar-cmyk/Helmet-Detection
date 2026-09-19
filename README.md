## 🚀 Live Demo

🔗 [Try HelmetGuard AI Live](https://helmet-detection-4wkipytt328mfv24tknqmd.streamlit.app/)


# 🪖 HelmetGuard AI

### AI-Powered Helmet Detection System using YOLO11

> A computer vision system that detects whether motorcycle riders are wearing helmets using a custom-trained YOLO11 object detection model. The system supports image, video, and real-time webcam detection through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Road safety is a major concern in areas with heavy two-wheeler traffic. Manual monitoring of helmet compliance is difficult, time-consuming, and difficult to scale.

**HelmetGuard AI** uses deep learning and computer vision to automatically detect:

* 🟢 **With Helmet**
* 🔴 **Without Helmet**

The project uses **YOLO11** for object detection and provides multiple ways to use the trained model:

```text
Image → Helmet Detection
Video → Frame-by-frame Helmet Detection
Webcam → Real-time Helmet Detection
Streamlit → Interactive AI Dashboard
```

---

## ✨ Features

### 🖼️ Image Detection

Upload a traffic or rider image and the system detects helmets with bounding boxes and confidence scores.

### 🎥 Video Detection

Process traffic videos frame-by-frame and generate an annotated detection video.

### 📷 Real-Time Webcam Detection

Use your laptop webcam for live helmet detection.

### 🎯 Confidence Control

The Streamlit application provides a confidence threshold slider so users can control detection sensitivity.

### 📊 Detection Statistics

The application reports:

* Total detected objects
* Riders with helmets
* Riders without helmets

### 🎨 Interactive Dashboard

A modern Streamlit interface provides:

* Dark-themed UI
* Image upload
* Video upload
* Detection results
* Statistics
* Model information
* Confidence controls

---

# 🧠 Technology Stack

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Programming language                 |
| YOLO11       | Object detection                     |
| Ultralytics  | YOLO framework                       |
| OpenCV       | Computer vision and video processing |
| Streamlit    | Interactive web application          |
| Roboflow     | Dataset preparation                  |
| Google Colab | GPU-based model training             |
| Git & GitHub | Version control                      |

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Input Image/Video │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     YOLO11 Model    │
                    │     best.pt         │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Object Detection         │
                  │                          │
                  │ 🟢 With Helmet           │
                  │ 🔴 Without Helmet        │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Detection Results        │
                  │                          │
                  │ Bounding Boxes           │
                  │ Confidence Scores        │
                  │ Detection Statistics     │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │    Streamlit Dashboard   │
                  └──────────────────────────┘
```

---

# 📂 Project Structure

```text
Helmet-Detection/
│
├── model/
│   └── best.pt
│
├── input/
│   ├── images/
│   │   └── test.jpg
│   │
│   └── videos/
│       └── traffic.mp4
│
├── output/
│   ├── images/
│   └── videos/
│
├── src/
│   ├── detect_image.py
│   ├── detect_video.py
│   └── detect_webcam.py
│
├── runs/
│   └── detect/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Model Performance

The final selected model is the **V1 YOLO11n model**.

### Validation Results

| Metric    |     Score |
| --------- | --------: |
| Precision | **76.9%** |
| Recall    | **76.8%** |
| mAP@50    | **80.1%** |
| mAP@50-95 | **41.1%** |

### What these metrics mean

**Precision — 76.9%**

When the model predicts an object, approximately 76.9% of those predictions are correct according to the validation evaluation.

**Recall — 76.8%**

The model detects approximately 76.8% of the relevant objects represented in the validation evaluation.

**mAP@50 — 80.1%**

The model achieves approximately 80.1% mean Average Precision at an IoU threshold of 0.50.

**mAP@50-95 — 41.1%**

This metric evaluates detection performance across a range of IoU thresholds and is therefore more demanding than mAP@50.

---

# 🔬 Model Development

Two model-training experiments were performed.

## V1 — Baseline Model

Configuration:

```python
model = YOLO("yolo11n.pt")

model.train(
    data="Bike-Helmet-Detection-1/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0
)
```

Results:

```text
Precision    = 76.9%
Recall       = 76.8%
mAP@50       = 80.1%
mAP@50-95    = 41.1%
```

---

## V2 — Augmentation Experiment

The second experiment introduced additional training augmentation:

```python
degrees=10
translate=0.1
scale=0.5
fliplr=0.5
mosaic=1.0
```

Training configuration:

```python
epochs=50
imgsz=640
batch=16
device=0
```

V2 results:

```text
Precision    = 81.2%
Recall       = 68.6%
mAP@50       = 76.5%
mAP@50-95    = 41.7%
```

### Model Selection

V2 increased precision but reduced recall and mAP@50.

Therefore, **V1 was retained as the final model** because it provided a better balance for the project's current use case.

---

# 📦 Dataset

The project uses a Roboflow **Bike Helmet Detection** dataset prepared for YOLO training.

### Classes

```text
0 → With Helmet
1 → Without Helmet
```

### Training Object Distribution

```text
With Helmet      2081
Without Helmet   1050
```

This shows that the training annotations contain more helmet-positive objects than helmet-negative objects.

The class imbalance is one of the factors considered when analyzing the model's performance.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

```bash
cd Helmet-Detection
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Image Detection

```bash
python src/detect_image.py
```

Place your test image inside:

```text
input/images/
```

---

## Video Detection

```bash
python src/detect_video.py
```

Place your video inside:

```text
input/videos/
```

---

## Webcam Detection

```bash
python src/detect_webcam.py
```

The default webcam will open and perform real-time detection.

---

# 🌐 Streamlit Application

Run:

```bash
streamlit run app.py
```

The application provides:

```text
🪖 HelmetGuard AI

├── Image Detection
├── Video Detection
├── Confidence Threshold
├── Detection Statistics
└── Model Information
```

---

# 🎯 Detection Classes

### 🟢 With Helmet

The model identifies riders wearing a helmet.

### 🔴 Without Helmet

The model identifies riders who are not wearing a helmet.

Each detection can include:

```text
Class
Bounding Box
Confidence Score
```

---

# ⚠️ Current Limitations

Although the model performs well on the validation dataset, real-world traffic scenes can be more challenging.

Observed challenges include:

* Crowded traffic scenes
* Small or distant riders
* Different camera angles
* Occlusion between riders
* Lighting variations
* Different helmet styles
* Motion blur
* Domain differences between training images and real traffic videos

The model can therefore miss some riders or produce lower-confidence detections in complex traffic scenes.

This system should be considered a **computer-vision project/demo**, not a production enforcement system.

---

# 🔮 Future Improvements

Potential improvements include:

### 1. Larger and More Diverse Dataset

Add more real-world traffic images containing:

* Different road conditions
* Different camera positions
* Different lighting
* Dense traffic
* Helmetless riders

### 2. Improve Class Balance

Increase representation of the **Without Helmet** class.

### 3. Hard-Example Training

Collect examples where the current model makes mistakes and add them to the training dataset.

### 4. Model Experiments

Evaluate larger YOLO variants and compare their accuracy and inference speed.

### 5. Real-Time Optimization

Optimize the model for edge devices and real-time CCTV applications.

### 6. Automatic Violation Logging

Future versions could record:

```text
Timestamp
Camera ID
Detection Class
Confidence
Frame/Image
```

for further analysis.

### 7. CCTV Integration

The system could eventually be connected to live CCTV streams for continuous monitoring.

---

# 🧪 Project Workflow

```text
1. Dataset Collection
        ↓
2. Dataset Preparation
        ↓
3. YOLO Annotation Format
        ↓
4. Model Training
        ↓
5. Validation
        ↓
6. Model Comparison
        ↓
7. Image Testing
        ↓
8. Video Testing
        ↓
9. Webcam Testing
        ↓
10. Streamlit Deployment
```

---

# 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Object Detection
* YOLO architecture
* Transfer learning
* Dataset preparation
* Bounding boxes
* Confidence scores
* Precision
* Recall
* mAP
* Model validation
* Data augmentation
* Video inference
* Real-time computer vision
* Streamlit application development
* Python virtual environments
* Git/GitHub project organization

---

# 👨‍💻 Author

**Anurag Sanjeev Kumar**

B.Tech — Artificial Intelligence & Machine Learning

---

# ⭐ Project Highlights

```text
🧠 YOLO11 Object Detection
🪖 Helmet Compliance Detection
📷 Image Detection
🎥 Video Detection
📹 Real-Time Webcam Detection
📊 Detection Analytics
🌐 Streamlit Dashboard
🔬 Model Experimentation
📈 Performance Evaluation
```

---

## 📜 Disclaimer

This project is developed for educational and research purposes. Detection results depend on the quality and distribution of the training data and may not be reliable enough for automated real-world enforcement decisions.
