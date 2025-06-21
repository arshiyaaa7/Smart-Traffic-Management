# 🚦 Real-Time AI Traffic Control using Traffic Length Metrics

An AI-driven traffic control system designed to improve urban mobility by adapting traffic signals based on real-time road conditions.


## 📁 Project Structure

The repository is organized into three main modules:

### 1️⃣ Traffic Management System
This is the core of the solution and includes the following four steps:


**Step 1: Frame Capture**  
📷 Capture frames from traffic surveillance cameras, lane-wise.

**Step 2: Object Detection with YOLO**  
🟩 Apply YOLO (You Only Look Once) to detect vehicles and draw bounding boxes for each lane.

**Step 3: Distance Estimation using Linear Regression**  
📏 Convert pixel values to physical distance using a linear regression model. This replaces traditional vehicle counting approaches.

**Step 4: Dynamic Signal Handling**  
🚦 Calculate time from the measured distance and adjust traffic signal timers dynamically to ensure smooth traffic flow based on real-time density.

⚠️ *Note:* We do not count vehicles. Our approach is unique and explained in detail in the accompanying research paper.
https://ieeexplore.ieee.org/document/10882298

![Traffic-Bounding Box](https://github.com/user-attachments/assets/35d56d5d-2506-4ac3-b27f-6fc97cb46765)


### 2️⃣ Accident Detection Module

🚑 Analyzes traffic frames to identify accidents or congestion.  
📡 Sends alerts to local authorities upon detection for rapid response.

![Crash1](https://github.com/user-attachments/assets/4225e608-dad5-4eae-9885-0afcde67b49e)

![crash3](https://github.com/user-attachments/assets/c456e320-4aef-42f9-9bcf-2b8471286abf)


### 3️⃣ Emergency Vehicle Detection

🚨 Detects emergency vehicles like ambulances in real-time using YOLO.  
🟢 Dynamically prioritizes lanes by adjusting traffic signals.

**Still in progress**

## 🔧 Setup & Dependencies
- Python ≥ 3.10  
- OpenCV  
- NumPy  
- TensorFlow or PyTorch  
- Custom YOLOv8 model

📌 *Note:* Models and datasets are not included. Users should train or provide their own for full functionality.


## 📜 Research Contribution

This project is backed by a research paper detailing our custom approach tailored for Indian traffic conditions, using bounding boxes and regression rather than vehicle counting.

## 👩‍💻 Contributors

- Arshiya Shaikh  
- Arpita Kamble 
- Yash Kishor Patil
- Prerna Khatri  

## 📬 Contact

For detailed insights or collaboration, feel free to reach out to the contributors.


