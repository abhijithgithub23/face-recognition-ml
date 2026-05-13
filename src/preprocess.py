import os
from deepface import DeepFace
from tqdm import tqdm  # <-- Import the progress bar library

def validate_dataset(dataset_path="dataset"):
    print("Scanning folders to count total images...")
    
    # 1. Gather all image paths first so we know the total count
    all_image_paths = []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            all_image_paths.append(os.path.join(person_dir, img_name))

    total_images = len(all_image_paths)
    print(f"Total images found: {total_images}")
    print("Starting face detection with RetinaFace...\n")
    
    skipped_count = 0
    
    # 2. Wrap our image list in tqdm() to generate the progress bar
    for img_path in tqdm(all_image_paths, desc="Processing Images", unit="img"):
        try:
            DeepFace.extract_faces(img_path=img_path, detector_backend='retinaface', enforce_detection=True)
        except ValueError:
            # We use tqdm.write() instead of print() so it doesn't break the progress bar visually
            tqdm.write(f"⚠️ No face detected in {img_path}. Ignoring.")
            skipped_count += 1
        except Exception as e:
            tqdm.write(f"❌ Error processing {img_path}: {e}")
                
    print(f"\nPreprocessing and validation complete. Ignored {skipped_count} images out of {total_images}.")

if __name__ == "__main__":
    validate_dataset()