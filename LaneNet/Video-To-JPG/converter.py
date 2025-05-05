import cv2
import os

def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frame is returned

        # Rotate the frame by 180 degrees
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Save the rotated frame as a JPEG file
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, rotated_frame)

        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to '{output_dir}'")


script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "Original", "IMG_0223.mov")
output_folder = os.path.join(script_dir, "output_frames")

print(f"Video path: {video_path}")
print(f"Output folder: {output_folder}")

# Example usage
extract_frames(video_path, output_folder)
