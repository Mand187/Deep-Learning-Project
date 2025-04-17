import torch
from ultralytics import YOLO
import pandas as pd
import os # Make sure os is imported
import cv2
import multiprocessing as mp
from pathlib import Path
import time # Optional: for timing
from tqdm import tqdm # Import tqdm
import signal # Import signal module
import sys # Import sys module

# Function to be executed by each worker process
def process_video_on_gpu(video_path, gpu_id, model_path, output_dir):
    """
    Processes a single video file on a specified GPU and returns tracking data.
    Includes a tqdm progress bar for frame processing.
    Sets CUDA_VISIBLE_DEVICES for process isolation.
    """
    try:
        # --- Enforce GPU Affinity ---
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Now, the assigned GPU will appear as 'cuda:0' to this process
        device = 'cuda:0'
        # --- ---

        process_name = mp.current_process().name
        # print(f"{process_name}: Assigning to visible GPU {gpu_id} (as {device}) for video {video_path.name}") # Adjusted print

        # Load the model inside the process onto the assigned GPU
        # Model loading should happen *after* CUDA_VISIBLE_DEVICES is set
        model = YOLO(model_path).to(device)

        results_data = [] # Use a list to collect data efficiently

        # --- Get total frames for tqdm ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
             print(f"Error opening video file: {video_path}")
             return video_path.stem, None # Handle error if video can't be opened
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # --- ---

        # Run tracking
        results = model.track(
            source=str(video_path),
            show=False,
            save=False,
            half=True,        # Use half-precision if supported and desired
            stream=True,
            device=device,    # Explicitly specify the device for tracking (now 'cuda:0' for this process)
            verbose=False,    # Reduce console output from YOLO per frame
            batch=500          # Add batch size argument
        )

        # --- Process results stream with tqdm ---
        # Use position=gpu_id to try and stack bars vertically (use original gpu_id for positioning)
        # leave=False cleans up the bar after completion
        # desc provides context
        progress_bar = tqdm(
            enumerate(results),
            total=total_frames,
            desc=f"GPU {gpu_id} | {video_path.name}", # Display original GPU ID
            position=gpu_id, # Assign position based on original GPU ID
            leave=False      # Remove bar once done
        )
        for frame, result in progress_bar:
        # --- ---
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

        # print(f"{process_name}: Finished processing {video_path.name} on GPU {gpu_id}. Found {len(results_data)} detections.") # Less verbose with tqdm
        # Return the original video filename stem and the collected data
        return video_path.stem, results_data
    except Exception as e:
        # Ensure CUDA_VISIBLE_DEVICES is printed in case of error for debugging
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        print(f"Error in {process_name} (Visible GPU(s): {cuda_visible}) processing {video_path.name} on assigned GPU {gpu_id}: {e}")
        # Return filename stem and None to indicate failure
        return video_path.stem, None
    finally:
        # Optional: Clean up env var if needed, though process exit usually handles this.
        # if "CUDA_VISIBLE_DEVICES" in os.environ:
        #     del os.environ["CUDA_VISIBLE_DEVICES"]
        pass

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


# --- Global Pool Variable ---
# Define pool globally or make it accessible to the handler
pool = None

# --- Signal Handler ---
def signal_handler(sig, frame):
    """Handles SIGINT and SIGTSTP signals to terminate the pool."""
    print(f'\nSignal {sig} received, terminating worker processes...')
    if pool:
        pool.terminate() # Forcefully terminate worker processes
        pool.join()      # Wait for termination to complete
    print("Processes terminated.")
    sys.exit(1) # Exit the main script


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    video_dir = Path('videos/trimmed')  # Use pathlib for easier path handling
    output_dir = Path('output_csvs')
    model_path = 'yolo12l.pt' # Define model path once
    num_gpus_to_use = 4       # Explicitly set to use 4 GPUs

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

    # --- Register Signal Handlers ---
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTSTP, signal_handler) # Handle Ctrl+Z

    # --- Multiprocessing Setup ---
    # IMPORTANT: Use 'spawn' start method for CUDA compatibility with multiprocessing
    try:
        # Check if start method is already set to spawn, avoid error if run multiple times in interactive session
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'.")
        else:
            print("Multiprocessing start method already set to 'spawn'.")
    except RuntimeError as e:
         # Handle cases where it might fail (e.g., context already started)
         current_method = mp.get_start_method(allow_none=True)
         print(f"Warning: Could not set start method to 'spawn': {e}. Using current method '{current_method}'.")

    try:
        # Create a pool of worker processes, one for each GPU we intend to use
        print(f"Creating process pool with size {num_gpus_to_use}")
        # Assign to the global pool variable so the handler can access it
        globals()['pool'] = mp.Pool(processes=num_gpus_to_use)


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
        print("Waiting for all video processing tasks to complete (Press Ctrl+C or Ctrl+Z to interrupt)...")
        # Wait for all tasks to finish
        all_done = False
        while not all_done:
            all_done = True
            for res in async_results:
                if not res.ready():
                    all_done = False
                    time.sleep(0.5) # Wait briefly before checking again
                    break
            if all_done:
                 # Retrieve results to catch any exceptions from workers
                 for i, res in enumerate(async_results):
                     try:
                         res.get()
                     except Exception as e:
                         # Error already printed in worker or callback, or print here
                         print(f"\nError observed in result for task {i+1}: {e}")


        print("\nAll tasks seem complete.")

    except Exception as e:
        print(f"\nAn error occurred in the main process: {e}")
    finally:
        # --- Ensure Pool Cleanup ---
        if pool:
            print("Closing pool...")
            # Check if pool was already terminated by signal handler
            # Terminate might be safer than close/join if interrupted uncleanly
            pool.terminate()
            pool.join()
            print("Pool closed.")

        # --- Finalization ---
        end_time = time.time() # Optional: End timer
        print(f"\nProcessing finished or interrupted. Total time: {end_time - start_time:.2f} seconds.")
        print(f"CSV files saved in: {output_dir}")