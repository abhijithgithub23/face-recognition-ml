import cv2
import os

def collect_data(person_name, num_images=50):
    save_path = os.path.join("dataset", person_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Collecting images for {person_name}...")
    print("Press 's' to save an image. Press 'q' to quit.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_name = os.path.join(save_path, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved: {img_name} ({count + 1}/{num_images})")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collection complete.")

if __name__ == "__main__":
    # Change "abhi" to the name of the person you are collecting data for
    collect_data("abhi", num_images=50)