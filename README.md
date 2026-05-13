# Face Recognition ML System

A high-accuracy, production-ready facial recognition pipeline built with Python. This system utilizes **RetinaFace** for robust face detection, **FaceNet** for generating 128-dimensional embeddings, and a **Support Vector Machine (SVM)** for classification.

The model currently achieves an internal training accuracy of **100.00%** and a real-world validation accuracy of **98.73%**.

---

##  Key Features

*   **RetinaFace Integration:** Superior detection accuracy compared to OpenCV, even with tilted or partially obscured faces.
*   **Progress Tracking:** Integrated `tqdm` bars for all long-running processes (Preprocessing & Embedding generation).
*   **Smart "Unknown" Detection:** A dual-layer security system using a confidence threshold and a dedicated "unknown" class to reject strangers.
*   **Memory Managed:** Optimized for CPU environments (Ubuntu) with automatic handling of TensorFlow memory allocations.
*   **Balanced Learning:** Uses `class_weight='balanced'` to handle datasets where some people have more photos than others (Class Imbalance handling).

---

##  Project Structure

```text
face-recognition-ml/
├── dataset/                # Raw images organized in subfolders (name_of_person/)
├── embeddings/             # Generated mathematical fingerprints (embeddings.pkl)
├── models/                 # Trained SVM brain and Label Encoder
├── src/
│   ├── preprocess.py       # Data cleaner & face validator
│   ├── generate_embeddings.py # Feature extractor using FaceNet
│   ├── train_model.py      # SVM trainer with 80/20 train-test split
│   └── predict.py          # Inference script for testing new images
├── test_images/            # Folder for testing unseen images
└── requirements.txt        # Project dependencies
```

#  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-recognition-ml.git
cd face-recognition-ml
```

### 2. Set up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  The Pipeline Workflow

To train the model from scratch, run the scripts in the following order:

### 1. Preprocessing

Validates every image in your `dataset/` folder using **RetinaFace**. It identifies "garbage" data and ensures the pipeline only learns from detectable faces.

```bash
python src/preprocess.py
```

### 2. Generate Embeddings

Converts the validated images into **128-dimensional mathematical vectors** (fingerprints) using the **FaceNet** architecture. Consistency is maintained by using the RetinaFace backend during extraction.

```bash
python src/generate_embeddings.py
```

### 3. Train Model

Trains the **SVM classifier** using an 80/20 train-test split. This script outputs both training accuracy and validation accuracy to monitor for overfitting.

```bash
python src/train_model.py
```

---

##  Inference (Prediction)

To test the model on a new image, place the image in `test_images/` and run:

```bash
python src/predict.py
```

### Threshold Logic

| Result | Condition |
|--------|-----------|
| ✅ **Matched** | Confidence > 65% and match found |
| 🚫 **Stranger** | Match falls into the `unknown` dataset folder |
| ❌ **Rejected** | Confidence < 65% → labeled as *"Unknown Person (Rejected by Threshold)"* |

---

##  Performance Summary

| Metric | Score |
|--------|-------|
| Total Faces Processed | 1974 |
| Unique Classes (People) | 21 |
| Training Accuracy | 100.00% |
| Validation Accuracy | **98.73%** |
