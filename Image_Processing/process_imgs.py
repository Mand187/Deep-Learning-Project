import torch
from ultralytics import YOLO
import pandas as pd
import os
import cv2
import multiprocessing as mp
from pathlib import Path
import time # Optional: for timing

# Function to be executed by each worker process
def process_video_on_gpu(video_path, gpu_id, model_path, output_dir):
    """
    Processes a single video file on a specified GPU and returns tracking data.
    """
    try:
        process_name = mp.current_process().name
        print(f"{process_name}: Assigning to GPU {gpu_id} for video {video_path.name}")
        device = f'cuda:{gpu_id}'

        # Load the model inside the process onto the assigned GPU
        model = YOLO(model_path).to(device)

        results_data = [] # Use a list to collect data efficiently

        # Run tracking
        results = model.track(
            source=str(video_path),
            show=False,
            save=False,
            half=True,        # Use half-precision if supported and desired
            stream=True,
            device=device,    # Explicitly specify the device for tracking
            verbose=False     # Reduce console output from YOLO per frame
        )

        # Process results stream
        for frame, result in enumerate(results):
            # Check if tracking IDs are present in the current frame's results
            if result.boxes.id is not None:
                # Extract data, converting tensors to CPU lists immediately
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                boxes = result.boxes.xywh.cpu().tolist()
                # confs = result.boxes.conf.cpu().tolist() # Uncomment if confidence is needed

                for v_class, v_id, xywh in zip(classes, ids, boxes):
                    x, y, w, h = xywh
                    label = model.names[v_class] # Get class name from model
                    # Append data as a list
                    results_data.append([frame, v_id, label, int(x), int(y), int(w), int(h)])

        print(f"{process_name}: Finished processing {video_path.name} on GPU {gpu_id}. Found {len(results_data)} detections.")
        # Return the original video filename stem and the collected data
        return video_path.stem, results_data
    except Exception as e:
        print(f"Error in {process_name} processing {video_path.name} on GPU {gpu_id}: {e}")
        # Return filename stem and None to indicate failure
        return video_path.stem, None

# Function to save results to CSV (runs in the main process via callback)
def save_results_to_csv(result_tuple, output_dir):
    """
    Callback function to save the processed data to a CSV file.
    """
    filename_stem, data = result_tuple

    # Handle cases where processing failed or yielded no data
    if data is None:
        print(f"Skipping CSV generation for {filename_stem} due to processing error.")
        return
    if not data:
        print(f"No tracking data generated for {filename_stem}. Skipping CSV.")
        return

    try:
        # Create DataFrame from the collected list of lists
        df = pd.DataFrame(data, columns=['Frame', 'ID', 'Class', 'X', 'Y', 'Width', 'Height'])
        # Define output CSV path
        csv_filename = f"{filename_stem}_detections.csv"
        csv_path = output_dir / csv_filename
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"MainProcess: Saved results for {filename_stem} to {csv_path}")
    except Exception as e:
        print(f"MainProcess: Error saving CSV for {filename_stem}: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    video_dir = Path('dataset/videos')  # Use pathlib for easier path handling
    output_dir = Path('output_csvs')
    model_path = 'yolo12l.pt' # Define model path once
    num_gpus_to_use = 2       # Explicitly set to use 2 GPUs

    # --- Preparations ---
    start_time = time.time() # Optional: Start timer
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    if available_gpus < num_gpus_to_use:
        print(f"Error: Requested {num_gpus_to_use} GPUs, but only {available_gpus} are available.")
        exit()

    # Find video files
    video_files = [f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        exit()
    print(f"Found {len(video_files)} videos to process using {num_gpus_to_use} GPUs.")

    # --- Multiprocessing Setup ---
    # IMPORTANT: Use 'spawn' start method for CUDA compatibility with multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set start method to 'spawn': {e}. Using default.")


    # Create a pool of worker processes, one for each GPU we intend to use
    print(f"Creating process pool with size {num_gpus_to_use}")
    pool = mp.Pool(processes=num_gpus_to_use)

    # --- Distribute Tasks ---
    tasks = []
    for i, video_path in enumerate(video_files):
        gpu_id = i % num_gpus_to_use  # Cycle through GPU IDs 0, 1, 0, 1, ...
        # Add task arguments for the process_video_on_gpu function
        tasks.append((video_path, gpu_id, model_path, output_dir))

    # Use apply_async to submit all tasks and define the callback for saving
    async_results = []
    print("Submitting tasks to process pool...")
    for task_args in tasks:
        # Pass the output_dir to the callback using a lambda function
        res = pool.apply_async(process_video_on_gpu, args=task_args,
                               callback=lambda result: save_results_to_csv(result, output_dir))
        async_results.append(res)

    # --- Wait for Completion & Cleanup ---
    print("Waiting for all video processing tasks to complete...")
    # Wait for all tasks to finish
    for i, res in enumerate(async_results):
        try:
            res.get() # Wait for this specific result (optional, good for catching errors early)
            # print(f"Task {i+1}/{len(tasks)} completed.")
        except Exception as e:
            print(f"Error retrieving result for task {i+1}: {e}")


    pool.close()  # Prevent submitting more tasks
    pool.join()   # Wait for all worker processes to terminate

    # --- Finalization ---
    end_time = time.time() # Optional: End timer
    print(f"All videos processed. Total time: {end_time - start_time:.2f} seconds.")
    print(f"CSV files saved in: {output_dir}")