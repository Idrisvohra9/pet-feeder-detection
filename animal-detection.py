"""
Face Recognition Attendance System
This module handles the main application logic for face recognition and attendance tracking.
Supports both human and animal face recognition.
"""

from datetime import datetime
import os
import pickle
import csv
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import face_recognition
import cvzone


# Add animal detection cascades
def load_animal_cascades():
    """
    Load cascade classifiers for animal detection.

    Returns:
        Dictionary of animal cascade classifiers
    """
    cascades = {}
    cascade_dir = "Resources/Cascades"

    # Create directory if it doesn't exist
    if not os.path.exists(cascade_dir):
        os.makedirs(cascade_dir)
        print(f"Created {cascade_dir} directory. Please add animal cascade files.")
        return cascades

    # Try to load dog face cascade
    dog_cascade_path = os.path.join(cascade_dir, "haarcascade_dog_face.xml")
    if os.path.exists(dog_cascade_path):
        cascades["dog"] = cv2.CascadeClassifier(dog_cascade_path)
        print("Dog face cascade loaded successfully")
    else:
        print(f"Warning: Dog face cascade not found at {dog_cascade_path}")
        # Download dog cascade if not found
        try:
            import urllib.request

            dog_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(dog_cascade_url, dog_cascade_path)
            cascades["dog"] = cv2.CascadeClassifier(dog_cascade_path)
            print("Downloaded default face cascade for dog detection")
        except Exception as e:
            print(f"Error downloading dog cascade: {e}")

    # Add more animal cascades here as needed

    return cascades


# Modify find_encodings to handle animal images
def find_encodings(
    images_list: List[np.ndarray], image_types: List[str] = None
) -> List[np.ndarray]:
    """
    Generate face encodings for a list of images.

    Args:
        images_list: List of images to encode
        image_types: List of image types (human, dog, etc.)

    Returns:
        List of face encodings
    """
    encode_list = []

    # Load animal cascades
    animal_cascades = load_animal_cascades()

    for i, img in enumerate(images_list):
        # Convert BGR to RGB (face_recognition uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Default to human if image_types not provided
        img_type = (
            "human" if image_types is None or i >= len(image_types) else image_types[i]
        )

        try:
            if img_type == "human":
                # Human face encoding using face_recognition
                face_encodings = face_recognition.face_encodings(img_rgb)
                if face_encodings:
                    encode = face_encodings[0]
                    encode_list.append(encode)
                else:
                    print(f"Warning: No human face detected in image {i}")
            elif img_type in animal_cascades:
                # Animal face detection using cascade classifier
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                animal_faces = animal_cascades[img_type].detectMultiScale(gray, 1.3, 5)

                if len(animal_faces) > 0:
                    # Take the largest face
                    x, y, w, h = sorted(
                        animal_faces, key=lambda x: x[2] * x[3], reverse=True
                    )[0]
                    face_roi = img_rgb[y : y + h, x : x + w]

                    # Resize to standard size for consistency
                    face_roi = cv2.resize(face_roi, (128, 128))

                    # Flatten and normalize for simple feature vector
                    # This is a basic approach - could be improved with CNN features
                    features = face_roi.flatten() / 255.0

                    # Take a subset of features to keep size manageable
                    # This ensures compatibility with human face encodings
                    encode = features[:128]
                    encode_list.append(encode)
                else:
                    print(f"Warning: No {img_type} face detected in image {i}")
            else:
                print(f"Warning: Unsupported image type {img_type}")
        except Exception as e:
            print(f"Error encoding image {i}: {e}")

    return encode_list


# Update generate_encodings to handle animal images
def generate_encodings() -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Generate face encodings from images in the Images folder.

    Returns:
        Tuple of (encodings, student_ids, image_names)
    """
    # Importing student images
    FOLDERPATH = "Images"

    # Check if Images folder exists
    if not os.path.exists(FOLDERPATH):
        print(f"Error: {FOLDERPATH} directory not found")
        os.makedirs(FOLDERPATH)
        print(f"Created {FOLDERPATH} directory. Please add images and run again.")
        return [], [], []

    # Get list of images
    try:
        path_list = os.listdir(FOLDERPATH)
    except Exception as e:
        print(f"Error accessing {FOLDERPATH}: {e}")
        return [], [], []

    if not path_list:
        print(f"No images found in {FOLDERPATH}")
        return [], [], []

    print(f"Found {len(path_list)} images")

    # Create a backup folder for images
    backup_folder = "ImagesBackup"
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    img_list = []
    student_ids = []
    image_names = []  # Store original image filenames
    image_types = []  # Store image types (human, dog, etc.)

    # Process each image
    for path in path_list:
        try:
            # Read image
            img_path = os.path.join(FOLDERPATH, path)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Determine image type from filename prefix (e.g., dog_rex.jpg)
            filename = os.path.basename(path)
            if "_" in filename:
                img_type, name = filename.split("_", 1)
                if img_type not in ["human", "dog", "cat"]:
                    img_type = "human"  # Default to human if not recognized
            else:
                img_type = "human"  # Default to human if no prefix

            img_list.append(img)
            student_id = os.path.splitext(path)[0]
            student_ids.append(student_id)
            image_names.append(path)  # Store the original filename
            image_types.append(img_type)

            # Create a backup of the image
            import shutil

            src_path = os.path.join(FOLDERPATH, path)
            dst_path = os.path.join(backup_folder, path)
            shutil.copy2(src_path, dst_path)

        except Exception as e:
            print(f"Error processing image {path}: {e}")

    if not img_list:
        print("No valid images found")
        return [], [], []

    print(f"Processing {len(img_list)} images")
    print(f"Student IDs: {student_ids}")
    print(f"Image Names: {image_names}")
    print(f"Image Types: {image_types}")

    # Generate encodings
    print("Encoding Started ...")
    encode_list_known = find_encodings(img_list, image_types)

    if not encode_list_known:
        print("Error: No valid encodings generated")
        return [], [], []

    encode_list_known_with_ids = [
        encode_list_known,
        student_ids,
        image_names,
        image_types,
    ]
    print(f"Encoding Complete. Generated {len(encode_list_known)} encodings")

    # Save encodings to file
    try:
        with open("EncodeFile.p", "wb") as file:
            pickle.dump(encode_list_known_with_ids, file)
        print("File Saved Successfully")
    except Exception as e:
        print(f"Error saving encodings: {e}")

    return encode_list_known, student_ids, image_names


# Update load_encodings to handle animal images
def load_encodings(file_path: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load face encodings from file or generate if file doesn't exist.

    Args:
        file_path: Path to the encodings file

    Returns:
        Tuple of (encodings, student_ids, image_names)
    """
    if not os.path.exists(file_path):
        print(f"Encoding file {file_path} not found. Generating new encodings...")
        return generate_encodings()

    try:
        with open(file_path, "rb") as file:
            encode_list_known_with_ids = pickle.load(file)

        # Check if the file contains image types (newer format)
        if len(encode_list_known_with_ids) >= 4:
            encode_list_known, student_ids, image_names, image_types = (
                encode_list_known_with_ids
            )
            return encode_list_known, student_ids, image_names
        # Check if the file contains image names (older format)
        elif len(encode_list_known_with_ids) >= 3:
            encode_list_known, student_ids, image_names = encode_list_known_with_ids
            return encode_list_known, student_ids, image_names
        else:
            # Backward compatibility with oldest format
            encode_list_known, student_ids = encode_list_known_with_ids
            # Create default image names based on student IDs
            image_names = [f"{id}.png" for id in student_ids]
            return encode_list_known, student_ids, image_names
    except Exception as e:
        print(f"Error loading encodings: {e}. Generating new encodings...")
        return generate_encodings()


# Update main function to handle animal detection
def main() -> None:
    """Main function to run the face recognition attendance system."""
    try:
        # Initialize camera
        cap = initialize_camera()

        # Set up FPS calculation
        fps_start_time = datetime.now()
        fps_counter = 0
        fps = 0

        # Skip frame counter for processing optimization
        process_every_n_frames = 2  # Process every 2nd frame
        frame_count = 0

        # Load background image
        bg_path = "Resources/background.png"
        if not os.path.exists(bg_path):
            raise FileNotFoundError(f"Background image {bg_path} not found")

        img_background = cv2.imread(bg_path)
        if img_background is None:
            raise ValueError(f"Could not read background image {bg_path}")

        # Load mode images
        folder_mode_path = "Resources/Modes"
        img_mode_list = load_mode_images(folder_mode_path)

        # Load animal cascades
        animal_cascades = load_animal_cascades()

        # Load face encodings or generate if needed
        print("Loading/Generating Encode File ...")
        encode_list_known, student_ids, image_names = load_encodings("EncodeFile.p")
        print("Encode File Ready")

        # Initialize variables
        mode_type = 0
        counter = 0
        student_id = "-1"
        img_student = None
        current_image_name = ""
        attendance_info = None

        print("Starting face recognition...")

        while True:
            # Read frame from camera
            success, img = cap.read()
            if not success:
                print("Error: Failed to read frame from camera")
                # Try to reconnect
                cap.release()
                cap = initialize_camera()
                continue

            # Calculate FPS
            fps_counter += 1
            time_diff = (datetime.now() - fps_start_time).total_seconds()
            if time_diff >= 1.0:
                fps = fps_counter / time_diff
                fps_counter = 0
                fps_start_time = datetime.now()
                print(f"FPS: {fps:.2f}")

            # Skip frames to improve performance
            frame_count += 1
            process_this_frame = frame_count % process_every_n_frames == 0

            # Always resize the frame to match the expected dimensions
            img = cv2.resize(img, (640, 480))

            # Place current frame on background
            img_background[162 : 162 + 480, 55 : 55 + 640] = img

            # Only process face recognition on selected frames
            if process_this_frame:
                # Resize and convert image for face recognition
                img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                img_s_rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
                img_s_gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)

                # Initialize face detection variables
                face_cur_frame = []
                encode_cur_frame = []

                # Detect human faces
                human_faces = face_recognition.face_locations(img_s_rgb, model="hog")

                # Process human faces
                if human_faces:
                    face_cur_frame = human_faces
                    encode_cur_frame = face_recognition.face_encodings(
                        img_s_rgb, human_faces
                    )

                # Detect animal faces if no human faces found
                if not face_cur_frame and animal_cascades:
                    for animal_type, cascade in animal_cascades.items():
                        animal_faces = cascade.detectMultiScale(img_s_gray, 1.3, 5)

                        if len(animal_faces) > 0:
                            # Convert animal detections to face_recognition format (top, right, bottom, left)
                            for x, y, w, h in animal_faces:
                                # Add animal face location
                                face_cur_frame.append((y, x + w, y + h, x))

                                # Extract face ROI
                                face_roi = img_s_rgb[y : y + h, x : x + w]

                                # Resize to standard size
                                face_roi = cv2.resize(face_roi, (128, 128))

                                # Create simple feature vector
                                features = face_roi.flatten() / 255.0

                                # Take subset of features to match human encoding size
                                animal_encoding = features[:128]
                                encode_cur_frame.append(animal_encoding)

                img_background[44 : 44 + 633, 808 : 808 + 414] = img_mode_list[
                    mode_type
                ]

                # Process detected faces
                if face_cur_frame:
                    for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
                        # Compare with known faces
                        matches = face_recognition.compare_faces(
                            encode_list_known, encode_face, tolerance=0.6
                        )
                        face_dis = face_recognition.face_distance(
                            encode_list_known, encode_face
                        )

                        # Find best match
                        match_index = np.argmin(face_dis)

                        if matches[match_index]:
                            # Draw bounding box around face
                            y1, x2, y2, x1 = face_loc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                            img_background = cvzone.cornerRect(
                                img_background, bbox, rt=0
                            )

                            # Get student ID and image name
                            student_id = student_ids[match_index]
                            current_image_name = image_names[match_index]

                            # Initialize loading state
                            if counter == 0:
                                cvzone.putTextRect(
                                    img_background, "Loading", (275, 400)
                                )
                                cv2.imshow("Face Attendance", img_background)
                                cv2.waitKey(1)
                                counter = 1
                                mode_type = 1

                    # Process image data after face recognition
                    if counter != 0:
                        if counter == 1:
                            # Get attendance info from CSV
                            attendance_info = get_attendance_info(current_image_name)
                            print(f"Image recognized: {current_image_name}")

                            # Get image from backup
                            img_path = f"ImagesBackup/{current_image_name}"
                            if os.path.exists(img_path):
                                img_student = cv2.imread(img_path)
                                if img_student is None:
                                    print(f"Warning: Could not read image {img_path}")

                            # Update attendance
                            attendance_updated = update_attendance_csv(
                                current_image_name
                            )
                            if not attendance_updated:
                                mode_type = 3  # Already marked attendance
                                counter = 0
                                img_background[44 : 44 + 633, 808 : 808 + 414] = (
                                    img_mode_list[mode_type]
                                )
            else:
                # For skipped frames, just update the background with the current mode
                img_background[44 : 44 + 633, 808 : 808 + 414] = img_mode_list[
                    mode_type
                ]

            # Display FPS on the frame
            cv2.putText(
                img_background,
                f"FPS: {fps:.1f}",
                (55, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Continue with display logic for all frames
            if counter != 0 and mode_type != 3:
                if 10 < counter < 20:
                    mode_type = 2  # Update mode

                img_background[44 : 44 + 633, 808 : 808 + 414] = img_mode_list[
                    mode_type
                ]

                if counter <= 10 and attendance_info:
                    # Display attendance details
                    cv2.putText(
                        img_background,
                        str(attendance_info.get("total_attendance", "0")),
                        (861, 125),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        1,
                    )

                    # Display image name (instead of student ID)
                    cv2.putText(
                        img_background,
                        str(current_image_name),
                        (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    # Center image name as the main identifier
                    (w, h), _ = cv2.getTextSize(
                        os.path.splitext(current_image_name)[0],
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        1,
                    )
                    offset = (414 - w) // 2
                    cv2.putText(
                        img_background,
                        os.path.splitext(current_image_name)[0],  # Remove extension
                        (808 + offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (50, 50, 50),
                        1,
                    )

                    # Display matched image
                    if img_student is not None and img_student.size > 0:
                        # Resize the image to fit the display area
                        img_student = cv2.resize(img_student, (216, 216))
                        img_background[175 : 175 + 216, 909 : 909 + 216] = img_student

                counter += 1

                # Reset after displaying for a while
                if counter >= 20:
                    counter = 0
                    mode_type = 0
                    attendance_info = None
                    img_student = None
                    current_image_name = ""
                    img_background[44 : 44 + 633, 808 : 808 + 414] = img_mode_list[
                        mode_type
                    ]
            elif (
                process_this_frame
                and "face_cur_frame" in locals()
                and not face_cur_frame
            ):
                # No face detected
                mode_type = 0
                counter = 0
                current_image_name = ""

            # Display the result
            cv2.imshow("Face Attendance", img_background)
            key = cv2.waitKey(1)

            # Press 'q' to quit
            if key == ord("q"):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Initialize the camera for video capture with optimized settings for IP webcam.

    Args:
        camera_index: Index of the camera to use

    Returns:
        VideoCapture object
    """
    # IP webcam configuration
    ip = "192.168.1.3"
    address = (
        f"http://{ip}:8080/video"  # Changed from https to http for better performance
    )

    # Use RTSP if available for better performance
    rtsp_address = f"rtsp://{ip}:8080/h264_ulaw.sdp"

    # Try RTSP first (better performance)
    cap = cv2.VideoCapture(rtsp_address)
    if not cap.isOpened():
        # Fall back to HTTP
        cap = cv2.VideoCapture(address)
        if not cap.isOpened():
            # Last resort - try local camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise ValueError("Error: Could not open any camera")

    # Optimize buffer size (smaller buffer = less latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Set lower resolution if possible to improve performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Set codec to MJPG for better performance if supported
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print("Camera initialized successfully")
    return cap


def load_mode_images(folder_path: str) -> List[np.ndarray]:
    """
    Load mode images from the specified folder.

    Args:
        folder_path: Path to the folder containing mode images

    Returns:
        List of loaded images
    """
    if not os.path.exists(folder_path):
        print(f"Error: {folder_path} not found")
        os.makedirs(folder_path)
        raise FileNotFoundError(f"{folder_path} not found")

    mode_path_list = os.listdir(folder_path)
    img_mode_list = []

    for path in mode_path_list:
        img_path = os.path.join(folder_path, path)
        img = cv2.imread(img_path)
        if img is not None:
            img_mode_list.append(img)
        else:
            print(f"Warning: Could not read image {img_path}")

    if not img_mode_list:
        raise ValueError(f"No valid images found in {folder_path}")

    return img_mode_list


def update_attendance_csv(image_name: str) -> bool:
    """
    Update attendance in CSV file for the matched image.

    Args:
        image_name: Name of the matched image file

    Returns:
        Boolean indicating if attendance was updated successfully
    """
    csv_file = "attendance.csv"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create CSV if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_name", "last_attendance_time", "total_attendance"])

    # Read existing data
    rows = []
    image_exists = False

    try:
        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["image_name"] == image_name:
                    # Update existing entry
                    image_exists = True
                    # Check if enough time has passed (1 minute for testing)
                    if "last_attendance_time" in row:
                        last_time = datetime.strptime(
                            row["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
                        )
                        time_diff = (datetime.now() - last_time).total_seconds()
                        # if time_diff < 60:  # 60 seconds = 1 minute
                        #     print(f"Attendance already marked for {image_name}")
                        #     return False

                    row["last_attendance_time"] = current_time
                    row["total_attendance"] = str(
                        int(row.get("total_attendance", "0")) + 1
                    )
                rows.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")

    # Add new entry if image doesn't exist
    if not image_exists:
        rows.append(
            {
                "image_name": image_name,
                "last_attendance_time": current_time,
                "total_attendance": "1",
            }
        )

    # Write updated data back to CSV
    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            fieldnames = ["image_name", "last_attendance_time", "total_attendance"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Attendance updated for {image_name}")
        return True
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return False


def get_attendance_info(image_name: str) -> Dict[str, Any]:
    """
    Get attendance information for an image.

    Args:
        image_name: Name of the image file

    Returns:
        Dictionary with attendance information
    """
    csv_file = "attendance.csv"

    if not os.path.exists(csv_file):
        return {
            "image_name": image_name,
            "total_attendance": "0",
            "last_attendance_time": "Never",
        }

    try:
        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["image_name"] == image_name:
                    return row
    except Exception as e:
        print(f"Error reading CSV: {e}")

    return {
        "image_name": image_name,
        "total_attendance": "0",
        "last_attendance_time": "Never",
    }


if __name__ == "__main__":
    main()
