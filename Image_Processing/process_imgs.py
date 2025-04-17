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
import concurrent.futures # Import ThreadPoolExecutor
import threading # For potential Lock if needed, though futures might suffice

# --- Helper function for CPU-bound result processing ---
def _extract_frame_data(frame, result, model_names):
    """Extracts detection data for a single frame result (CPU-bound)."""
    frame_data = []
    if result.boxes.id is not None:
        try:
            # Perform CPU-bound operations here
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            boxes = result.boxes.xywh.cpu().tolist()
            # confs = result.boxes.conf.cpu().tolist() # If needed

            for v_class, v_id, xywh in zip(classes, ids, boxes):
                x, y, w, h = xywh
                label = model_names[v_class] # Get class name
                frame_data.append([frame, v_id, label, int(x), int(y), int(w), int(h)])
        except Exception as e:
            # Log error specific to this frame extraction
            print(f"Error extracting data for frame {frame}: {e}")
            # Decide if you want to return partial data or empty
    return frame_data

# Function to be executed by each worker process
def process_video_on_gpu(video_path, gpu_id, model_path, output_dir):
    """
    Processes a single video file on a specified GPU and returns tracking data.
    Includes a tqdm progress bar for frame processing.
    Sets CUDA_VISIBLE_DEVICES for process isolation.
    Uses a ThreadPoolExecutor for non-blocking result extraction.
    """
    # --- Local ThreadPoolExecutor for result processing ---
    # Adjust max_workers based on CPU cores available to the process
    # With 40 total cores and 4 processes, each process might utilize ~10 cores.
    # Start with a number like 8, as the main process thread also needs CPU.
    # The optimal value depends on how much the GIL is released during extraction and requires testing.
    result_extractor_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8) # Increased from 4
    extraction_futures = []
    final_results_data = [] # Renamed to avoid confusion

    try:
        # --- Enforce GPU Affinity ---
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = 'cuda:0'
        # ---

        process_name = mp.current_process().name
        model = YOLO(model_path).to(device)
        model_names = model.names # Get model names once

        # --- Get total frames for tqdm ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
             print(f"Error opening video file: {video_path}")
             # Ensure executor is shut down even on early exit
             result_extractor_executor.shutdown(wait=False)
             return video_path.stem, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # ---

        # Run tracking
        results = model.track(
            source=str(video_path),
            show=False,
            save=False,
            half=True,
            stream=True,
            device=device,
            verbose=False,
            batch=512
        )

        # --- Process results stream, submitting extraction to threads ---
        progress_bar = tqdm(
            enumerate(results),
            total=total_frames,
            desc=f"GPU {gpu_id} | {video_path.name}",
            position=gpu_id,
            leave=False
        )
        for frame, result in progress_bar:
            # Submit the CPU-bound work to the executor
            future = result_extractor_executor.submit(_extract_frame_data, frame, result, model_names)
            extraction_futures.append(future)

        # --- Wait for all extraction tasks to complete and aggregate results ---
        print(f"{process_name}: GPU processing done for {video_path.name}. Aggregating results...")
        # Add a progress bar for aggregation if many frames/futures
        aggregation_bar = tqdm(concurrent.futures.as_completed(extraction_futures), total=len(extraction_futures), desc=f"Aggregating GPU {gpu_id}", position=gpu_id, leave=False)
        for future in aggregation_bar:
            try:
                frame_data = future.result()
                if frame_data: # Only extend if data was extracted
                    final_results_data.extend(frame_data)
            except Exception as e:
                print(f"Error retrieving result from extraction future: {e}")

        # Sort results by frame number if order might be slightly off due to threading
        # (though as_completed yields in completion order, not submission order)
        # If strict frame order is critical, collect results in a dict keyed by frame
        # or sort at the end. Sorting is safer.
        final_results_data.sort(key=lambda row: row[0]) # Sort by frame number

        print(f"{process_name}: Finished processing {video_path.name} on GPU {gpu_id}. Found {len(final_results_data)} detections.")
        return video_path.stem, final_results_data

    except Exception as e:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        print(f"Error in {process_name} (Visible GPU(s): {cuda_visible}) processing {video_path.name} on assigned GPU {gpu_id}: {e}")
        return video_path.stem, None
    finally:
        # --- Ensure local executor is shut down ---
        result_extractor_executor.shutdown(wait=True) # Wait for threads here
        # ... existing finally block content ...
        pass

# --- Internal Saving Function (runs in thread pool) ---
def _actual_save_to_csv(filename_stem, data, output_dir):
    """Performs the actual DataFrame creation and saving."""
    if data is None:
        # print(f"ThreadSaver: Skipping CSV generation for {filename_stem} due to processing error.") # Optional: more specific logging
        return
    if not data:
        # print(f"ThreadSaver: No tracking data generated for {filename_stem}. Skipping CSV.") # Optional: more specific logging
        return

    try:
        df = pd.DataFrame(data, columns=['Frame', 'ID', 'Class', 'X', 'Y', 'Width', 'Height'])
        csv_filename = f"{filename_stem}_detections.csv"
        csv_path = output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        # print(f"ThreadSaver: Saved results for {filename_stem} to {csv_path}") # Optional: more specific logging
    except Exception as e:
        print(f"ThreadSaver: Error saving CSV for {filename_stem}: {e}")


# --- Modified Callback Function (Submits task to thread pool) ---
# Keep track of submitted save futures if needed for shutdown confirmation
save_futures = []
def save_results_to_csv(result_tuple, output_dir, executor):
    """
    Callback function to submit the saving task to a ThreadPoolExecutor.
    """
    filename_stem, data = result_tuple
    # Submit the actual saving work to the executor
    future = executor.submit(_actual_save_to_csv, filename_stem, data, output_dir)
    save_futures.append(future) # Optional: track futures


# --- Global Pool Variable ---
# Define pool globally or make it accessible to the handler
pool = None
save_executor = None # Global variable for the save executor

# --- Signal Handler ---
def signal_handler(sig, frame):
    """Handles SIGINT and SIGTSTP signals to terminate the pool and executor."""
    print(f'\nSignal {sig} received, terminating processes and threads...')
    if save_executor:
        print("Shutting down save executor (no new tasks)...")
        # Shutdown without waiting indefinitely, cancel pending if possible
        save_executor.shutdown(wait=False, cancel_futures=True)
    if pool:
        print("Terminating worker pool...")
        pool.terminate() # Forcefully terminate worker processes
        pool.join()      # Wait for termination to complete
    print("Processes and threads terminated.")
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

        # Create a ThreadPoolExecutor for saving CSV files
        # Adjust max_workers based on expected I/O load vs CPU cores
        save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        globals()['save_executor'] = save_executor # Make accessible to signal handler


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
            # Use functools.partial or lambda to pass executor to callback
            callback_with_executor = lambda result: save_results_to_csv(result, output_dir, save_executor)
            res = pool.apply_async(process_video_on_gpu, args=task_args,
                                   callback=callback_with_executor)
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
        # --- Ensure Pool and Executor Cleanup ---
        if pool:
            print("Closing worker pool...")
            pool.close() # Allow workers to finish current task
            pool.join()  # Wait for workers to exit
            print("Worker pool closed.")

        if save_executor:
            print("Shutting down save executor (waiting for saves to complete)...")
            # Wait for all submitted save tasks to finish
            concurrent.futures.wait(save_futures) # Wait for tracked futures
            save_executor.shutdown(wait=True)
            print("Save executor shut down.")

        # --- Finalization ---
        end_time = time.time() # Optional: End timer
        print(f"\nProcessing finished or interrupted. Total time: {end_time - start_time:.2f} seconds.")
        print(f"CSV files saved in: {output_dir}")