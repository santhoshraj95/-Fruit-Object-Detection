# -Fruit-Object-Detection
Identify and localize fruits (banana, orange, and apple) within an image by drawing bounding boxes around them and labeling each detected fruit correctly. The model should work efficiently on unseen images and maintain accuracy across different lighting and orientation conditions.
🍎🍌🍊 Fruit Object Detection using YOLOv8

Deep Learning | Computer Vision | Streamlit | AWS Deployment

📌 Project Overview

This project implements an end-to-end Object Detection system to identify and localize fruits (apple, banana, orange) in images using YOLOv8.
The complete pipeline includes data preprocessing, annotation conversion, model training, evaluation, visualization, and deployment readiness.

The final application is designed to be deployed using Streamlit and hosted on AWS for real-time fruit detection.

🎯 Domain

Manufacturing Quality Check | Computer Vision – Object Detection

❓ Problem Statement

Detect and localize fruits (banana, orange, apple) in images by drawing bounding boxes and labeling them correctly.
The model should generalize well to unseen images, handle lighting variations, orientation changes, and partial occlusions.

💼 Business Use Cases

🛒 Smart Retail – Automated fruit recognition & counting

🌾 Agriculture – Yield estimation from fruit detection

🏭 Food Industry – Real-time fruit sorting on conveyor belts

🥗 Health Tech – Fruit recognition for calorie tracking apps

🧠 Skills Gained

Image preprocessing & augmentation

Object Detection with CNNs

Transfer Learning using YOLOv8

Annotation conversion (Pascal VOC → YOLO)

Model evaluation using mAP, Precision, Recall, F1

Visualization of predictions

Deployment-ready model preparation

📂 Dataset Details
Dataset Structure
Fruit/
├── Train/
│   ├── images/
│   └── labels/
├── Test/
│   ├── images/
│   └── labels/
└── data.yaml

Dataset Information

Train Images: 240 (apple, banana, orange)

Test Images: 60 (20 per class)

Each image may contain multiple fruits

Includes occlusion & lighting variations

Annotations provided as Pascal VOC XML, converted to YOLO format

🔧 Tech Stack

Python

YOLOv8 (Ultralytics)

PyTorch

OpenCV

Google Colab

Streamlit

AWS

🔄 Project Workflow
1️⃣ Data Extraction

Mounted Google Drive in Colab

Extracted Train & Test zip files

2️⃣ Annotation Conversion

Converted Pascal VOC XML → YOLO format

Normalized bounding boxes

Class mapping:

apple → 0
banana → 1
orange → 2

3️⃣ Data Preprocessing

Resized images to 640×640

Normalized pixel values (0–1)

4️⃣ Data Augmentation

Horizontal & vertical flips

Random rotations

Brightness & contrast variations

Gaussian noise

🤖 Model Training

Model: YOLOv8n (pre-trained on COCO)

Transfer Learning applied

Train/Validation split: 80/20

Training parameters:

Epochs: 50

Image size: 640

Batch size: 8

📊 Model Evaluation
Achieved Results

mAP@0.5: 0.93

F1-Score: 0.90

High confidence detection across test images

Metrics Used

Precision

Recall

F1-Score

mean Average Precision (mAP)

IoU (Intersection over Union)

Inference latency (real-time readiness)

🖼️ Visualization

Bounding boxes drawn on test images

Labels with confidence scores displayed

Outputs saved automatically for review

🚀 Deployment

Trained model exported as:

best.pt


Integrated into a Streamlit application

Hosted on AWS for real-time image inference

📦 Project Deliverables

✅ Jupyter Notebook / Python scripts

✅ Trained YOLOv8 model (best.pt)

✅ Evaluation metrics & validation report

✅ Detection visualizations

✅ Streamlit application

✅ AWS-hosted inference app

✅ Complete README documentation

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py

🏁 Conclusion

This project demonstrates a production-ready object detection system, covering the full lifecycle from raw data to cloud deployment.
It showcases how YOLOv8 and transfer learning can be effectively applied to real-world computer vision problems.
