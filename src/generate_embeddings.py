import os
import pickle
import numpy as np
from deepface import DeepFace

def generate_embeddings(dataset_path="dataset", model_name="Facenet"):
    X = [] # Embeddings
    y = [] # Labels

    print("Generating face embeddings...")

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                # represent returns a list of dictionaries. We take the first face found.
                embedding_objs = DeepFace.represent(
                    img_path=img_path, 
                    model_name=model_name, 
                    enforce_detection=True
                )
                embedding = embedding_objs[0]["embedding"]
                
                X.append(embedding)
                y.append(person_name)
                print(f"Processed: {img_name}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Save to embeddings folder
    with open("embeddings/embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": np.array(X), "labels": np.array(y)}, f)
        
    print("Embeddings successfully saved to embeddings/embeddings.pkl")

if __name__ == "__main__":
    generate_embeddings()