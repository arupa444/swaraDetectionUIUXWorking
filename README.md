# 🎶 Swara Detection from Classical Music Notations (TFLite + Streamlit)

This project is a **Streamlit-based web interface** for detecting *Swara* (musical notes) from images of Indian classical music notations written in Devanagari script. It integrates a custom-trained **TensorFlow Lite (TFLite)** model, enabling real-time object detection of swaras directly from scanned book pages like the *Kramik Pustak Vol. 2*.

---

## 📁 Project Structure

swaraDetectionUIUXWorking/
├── results/
│ ├── labelmap.txt # Stores class label mappings
│ └── result/ # Folder for storing detection outputs
│
├── tfLiteFile/
│ ├── interfaceForMacOs.py # Main Streamlit interface
│ ├── labelmap.txt # Label mapping used by the model
│ ├── trainedcom.tflite # Trained TFLite object detection model
│
├── requirements.txt # Python package dependencies
└── README.md # Project documentation

---

## 🚀 How It Works

1. **Upload** an image (JPG/PNG) of handwritten or printed classical notation.
2. **Run detection**: The TFLite model detects Swaras and draws bounding boxes.
3. **Results displayed** with confidence scores and overlaid annotations.
4. **Output saved** in a text file under `results/`.

---

## 🛠️ Tech Stack

- **TensorFlow Lite** — lightweight object detection
- **Streamlit** — for building the UI
- **OpenCV** — image processing
- **Label Studio** — dataset annotation
- **Python 3.9**

---

## 🧠 Model Details

- **Architecture:** SSD MobileNet V2 FPNlite 320
- **Input Size:** 320x320
- **Classes:** Multiple Swara classes including Sa, Re, Ga, Ma, etc.
- **Training Data:** Annotated swara snippets from Kramik Pustak Vol. 2
- **Accuracy:** Up to 98% on test data (avg: ~78%)

---

## ▶️ Run Locally

### Step 1: Clone this repo  
```bash
git clone https://github.com/your-repo/swara-detection-ui.git
cd swara-detection-ui

### Step 2: Install dependencies  
```bash
pip install -r requirements.txt
```

### Step 3: Launch Streamlit app 
```bash
streamlit run tfLiteFile/interfaceForMacOs.py
```

## 📦 Requirements
Make sure your requirements.txt includes:

tensorflow==2.8.0
streamlit
opencv-python
numpy
matplotlib
Pillow


## 🤝 Contributors
Arupa Nanda Swain

A. Anushruth Reddy

C. Viswanath

Vadali SS Bharadwaja

## 📚 Acknowledgment
Special thanks to Pandit Vishnu Narayan Bhatkhande’s Kramik Pustak Vol. 2 — a foundational resource for classical notation.


