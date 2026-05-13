import os
import pickle
import numpy as np
from deepface import DeepFace
from tqdm import tqdm # <-- For the progress bar

def generate_embeddings(dataset_path="dataset", model_name="Facenet"):
    X = [] # Embeddings
    y = [] # Labels

    print("Scanning folders to count total images...")
    all_image_paths = []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            all_image_paths.append((os.path.join(person_dir, img_name), person_name))

    total_images = len(all_image_paths)
    print(f"Generating embeddings for {total_images} images using {model_name}...")

    # Wrap the loop in tqdm for visibility
    for img_path, person_name in tqdm(all_image_paths, desc="Extracting Features", unit="img"):
        try:
            # represent returns a list of dictionaries. 
            embedding_objs = DeepFace.represent(
                img_path=img_path, 
                model_name=model_name, 
                enforce_detection=True,
                detector_backend="retinaface" 
            )
            
            # We take the first face found in the image
            embedding = embedding_objs[0]["embedding"]
            
            X.append(embedding)
            y.append(person_name)
            
        except Exception as e:
            # Using tqdm.write so it doesn't break the progress bar layout
            tqdm.write(f"⚠️ Could not process {img_path}: {e}")

    # Save to embeddings folder
    print("\nSaving embeddings to disk...")
    with open("embeddings/embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": np.array(X), "labels": np.array(y)}, f)
        
    print("✅ Embeddings successfully saved to embeddings/embeddings.pkl")

if __name__ == "__main__":
    generate_embeddings()