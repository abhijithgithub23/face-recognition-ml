import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # <--- NEW IMPORT
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

    unique_classes, counts = np.unique(labels, return_counts=True)
    print(f"\nDataset Breakdown: {len(X)} total faces.")
    for cls, count in zip(unique_classes, counts):
        print(f" - {cls}: {count} images")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # --- THE REALITY CHECK: 80% Training, 20% Testing ---
    # stratify=y ensures we keep the same ratio of images for each person
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining SVM Classifier on 80% of data...")
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Check accuracy on the 80% it studied
    train_accuracy = clf.score(X_train, y_train)
    print(f"🎯 Training Accuracy (The Open Book Test): {train_accuracy * 100:.2f}%")

    # Check accuracy on the 20% it has NEVER seen
    test_accuracy = clf.score(X_test, y_test)
    print(f"🕵️  Validation Accuracy (The REAL Test): {test_accuracy * 100:.2f}%")

    # Save the model and label encoder
    os.makedirs("models", exist_ok=True)
    with open("models/face_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\n✅ Model and Label Encoder saved to models/")

if __name__ == "__main__":
    train_classifier()