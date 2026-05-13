import pickle
import numpy as np
from deepface import DeepFace

def predict_image(img_path, threshold=0.65): 
    # Load model and encoder
    try:
        with open("models/face_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: Model files not found. Train the model first.")
        return

    try:
        # Extract embedding from test image using RetinaFace
        embedding_objs = DeepFace.represent(
            img_path=img_path, 
            model_name="Facenet", 
            enforce_detection=True,
            detector_backend="retinaface"
        )
        embedding = np.array(embedding_objs[0]["embedding"]).reshape(1, -1)

        # Predict probability
        probs = clf.predict_proba(embedding)[0]
        max_prob = np.max(probs)
        best_class_idx = np.argmax(probs)
        
        # Get the actual predicted name from the Label Encoder
        predicted_name = le.inverse_transform([best_class_idx])[0]

        # Apply our Two-Layer Security Logic
        if max_prob < threshold:
            prediction = "Unknown Person (Rejected by Threshold)"
        elif predicted_name == "unknown":
            # If it matched the 'unknown' folder dataset
            prediction = "Unknown Person (Recognized as Stranger)"
        else:
            prediction = predicted_name

        print(f"\n--- Results for {img_path} ---")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max_prob * 100:.2f}%")

        return prediction, max_prob

    except ValueError:
        print(f"⚠️ Error: No face detected in {img_path}.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    test_img = "test_images/random-person2.jpeg"
    predict_image(test_img)