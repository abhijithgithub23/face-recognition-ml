import pickle
import numpy as np
from deepface import DeepFace
import cv2

def predict_image(img_path, threshold=0.70):
    # Load model and encoder
    with open("models/face_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    try:
        # Extract embedding from test image
        embedding_objs = DeepFace.represent(
            img_path=img_path, 
            model_name="Facenet", 
            enforce_detection=True
        )
        embedding = np.array(embedding_objs[0]["embedding"]).reshape(1, -1)

        # Predict probability
        probs = clf.predict_proba(embedding)[0]
        max_prob = np.max(probs)
        best_class_idx = np.argmax(probs)

        # Apply threshold logic for Unknown faces
        if max_prob < threshold:
            prediction = "Unknown Person"
        else:
            prediction = le.inverse_transform([best_class_idx])[0]

        print(f"\n--- Results for {img_path} ---")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max_prob * 100:.2f}%")

        return prediction, max_prob

    except ValueError:
        print(f"Error: No face detected in {img_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Test it out! Make sure you put a picture of yourself in the test_images folder.
    test_img = "test_images/test1.jpg"
    predict_image(test_img)