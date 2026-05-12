import os
from deepface import DeepFace

def validate_dataset(dataset_path="dataset"):
    print("Validating dataset... checking for detectable faces.")
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            try:
                # enforce_detection=True ensures the image has a recognizable face
                DeepFace.extract_faces(img_path=img_path, detector_backend='opencv', enforce_detection=True)
            except ValueError:
                print(f"No face detected in {img_path}. Removing image to keep dataset clean.")
                os.remove(img_path)
                
    print("Preprocessing and validation complete.")

if __name__ == "__main__":
    validate_dataset()