import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os

def train_classifier():
    print("Loading embeddings...")
    with open("embeddings/embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["embeddings"]
    labels = data["labels"]

    # Encode labels (e.g., 'abhi' -> 0, 'rahul' -> 1)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train SVM with probability=True so we can get confidence scores later
    print("Training SVM Classifier...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)

    # Save the model and label encoder
    with open("models/face_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Model and Label Encoder saved to models/")

if __name__ == "__main__":
    train_classifier()