import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os

def train_classifier():
    print("Loading embeddings...")
    try:
        with open("embeddings/embeddings.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: embeddings.pkl not found. Run generate_embeddings.py first.")
        return

    X = data["embeddings"]
    labels = data["labels"]

    # UPGRADE 1: Print dataset stats so you know exactly what is being trained
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(f"\nDataset Breakdown: {len(X)} total faces.")
    for cls, count in zip(unique_classes, counts):
        print(f" - {cls}: {count} images")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # UPGRADE 2: Added class_weight='balanced' to prevent bias toward large Kaggle folders
    print("\nTraining SVM Classifier (This might take a few seconds)...")
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X, y)

    # UPGRADE 3: Calculate and print the training accuracy
    accuracy = clf.score(X, y)
    print(f"🎯 Internal Training Accuracy: {accuracy * 100:.2f}%")

    # Save the model and label encoder
    os.makedirs("models", exist_ok=True)
    with open("models/face_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\n✅ Model and Label Encoder saved to models/")

if __name__ == "__main__":
    train_classifier()