import cv2
import os

def extract_frames_every_second(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Cannot determine FPS of the video.")
        return

    frame_interval = int(fps)  # Number of frames to skip to get 1 frame per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f} seconds")

    frame_idx = 0
    saved_frame_idx = 0
    while True:
        # Set the position of the next frame to capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frame is returned

        # Rotate the frame by 180 degrees if needed
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Resize the frame to 1280x720
        resized_frame = cv2.resize(rotated_frame, (1280, 720))

        # Save the resized and rotated frame as a JPEG file
        frame_filename = os.path.join(output_dir, f"frame_{saved_frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

        saved_frame_idx += 1
        frame_idx += frame_interval  # Move to the next second

    cap.release()
    print(f"Extracted {saved_frame_idx} frames to '{output_dir}'")

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "Original", "IMG_0223.mov")
output_folder = os.path.join(script_dir, "output_frames")

print(f"Video path: {video_path}")
print(f"Output folder: {output_folder}")

extract_frames_every_second(video_path, output_folder)
