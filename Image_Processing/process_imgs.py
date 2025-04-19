import torch
from ultralytics import YOLO
import pandas as pd
import os # Make sure os is imported
# import tqdm # Removed tqdm import here, will add specific import below
from tqdm import tqdm # Import tqdm
# from decord import VideoReader # Import VideoReader # Removed decord
import cv2 # Import OpenCV
import multiprocessing as mp
from pathlib import Path
import time # Optional: for timing
# from tqdm import tqdm # Removed duplicate tqdm import
import signal # Import signal module
import sys # Import sys module
import concurrent.futures # Import ThreadPoolExecutor
# import threading # No longer needed


# --- GPU Task Function ---
def gpu_process_video(video_path, gpu_id, model_path):
    """
    Processes a single video file on a specified GPU using model.track.
    Sets CUDA_VISIBLE_DEVICES for process isolation.
    Uses stream=True and shows progress with tqdm.
    Returns (video_stem, results_list, model_names) on success, None on failure.
    """
    process_name = mp.current_process().name
    print(f"{process_name} (GPU {gpu_id}): Starting GPU task for {video_path.name}")
    results_list = [] # Initialize list to store results

    try:
        # --- Enforce GPU Affinity ---
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = 'cuda:0'
        # ---

        # --- Get total frames using cv2 ---
        total_frames = None # Initialize
        cap = None # Initialize cap outside try
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path.name} using OpenCV.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0: # Check if frame count is valid
                    print(f"Warning: OpenCV reported 0 or negative frames for {video_path.name}. Progress bar total might be inaccurate.")
                    total_frames = None # Reset to None if invalid
        except Exception as e:
            print(f"Warning: Could not get frame count for {video_path.name} using OpenCV: {e}. Progress bar total might be inaccurate.")
            total_frames = None # Indicate unknown total
        finally:
            if cap is not None and cap.isOpened():
                cap.release() # Ensure capture is released
        # ---

        model = YOLO(model_path).to(device)
        model_names = model.names # Get model names once

        # --- Run tracking (GPU intensive) ---
        start_gpu_time = time.time()
        # stream=True returns a generator
        results_generator = model.track(
            source=str(video_path),
            show=False,
            save=False,
            half=True,
            stream=True, # Process frame by frame with a generator
            device=device,
            verbose=False,
            batch=128 # Adjust as needed
        )

        # --- Iterate with tqdm progress bar ---
        progress_bar = tqdm(
            iterable=results_generator, # Iterate directly over the generator
            total=total_frames, # Use frame count from decord, or None
            desc=f"GPU {gpu_id}: {video_path.stem[:20]:<20}", # Shortened description
            position=gpu_id, # Use gpu_id for positioning multiple bars
            leave=False, # Leave the bar after completion
            unit="frame",
            ncols=100 # Adjust width as needed
        )

        for result in progress_bar:
            # Append results to list as they are generated
            # Detach results from GPU memory if they are tensors to avoid accumulation
            # Note: Ultralytics results objects often handle this, but be mindful
            results_list.append(result.cpu()) # Move result to CPU to free GPU memory

        # --- GPU processing finished ---
        gpu_duration = time.time() - start_gpu_time
        # Clear the completed progress bar line by printing spaces or using leave=False
        # print(f"\r{' ' * 100}\r", end='') # Optional: Clear line if leave=False isn't enough
        print(f"{process_name} (GPU {gpu_id}): GPU processing done for {video_path.name} in {gpu_duration:.2f}s.")

        # Return necessary data for the CPU task
        return video_path.stem, results_list, model_names # Return the collected list

    except Exception as e:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        print(f"Error in GPU task {process_name} (Visible GPU(s): {cuda_visible}) for {video_path.name}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None # Indicate failure
    finally:
        # Ensure CUDA context is cleaned up if necessary, though model deletion usually handles it
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{process_name} (GPU {gpu_id}): GPU task finished for {video_path.name}")


# --- Helper function for CPU-bound result extraction (used by CPU task) ---
def _extract_frame_data(frame, result, model_names):
    """Extracts detection data for a single frame result (CPU-bound)."""
    frame_data = []
    # Check if result object is valid and has expected attributes
    if result is None or not hasattr(result, 'boxes') or result.boxes.id is None:
        return frame_data # Return empty if no valid boxes/ids

    try:
        # Perform CPU-bound operations here
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xywh.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist() # Extract confidence scores

        for v_class, v_id, xywh, conf in zip(classes, ids, boxes, confs): # Include confs in zip
            x, y, w, h = xywh
            # Ensure model_names is accessible and v_class is a valid index
            if model_names and 0 <= v_class < len(model_names):
                 label = model_names[v_class] # Get class name
                 # Add conf to the appended list
                 frame_data.append([frame, v_id, label, conf, int(x), int(y), int(w), int(h)])
            else:
                 print(f"Warning: Invalid class index {v_class} for frame {frame}. Skipping.")

    except Exception as e:
        # Log error specific to this frame extraction
        print(f"Error extracting data for frame {frame}: {e}")
        # Decide if you want to return partial data or empty
    return frame_data

# --- CPU Task Function ---
def cpu_process_and_save(gpu_result_tuple, output_dir):
    """
    Takes results from gpu_process_video, extracts data using a ThreadPoolExecutor
    with progress bar, creates DataFrame, and saves CSV.
    Runs in a ThreadPoolExecutor in the main process.
    """
    if gpu_result_tuple is None:
        print("CPU Task: Received None from GPU task, skipping.")
        return False # Indicate failure

    video_stem, results_list, model_names = gpu_result_tuple
    print(f"CPU Task: Starting processing for {video_stem}")

    # --- Local ThreadPoolExecutor for result extraction ---
    # Adjust max_workers based on CPU cores and task nature (IO bound vs CPU bound)
    extraction_executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() // 2 or 1) # Example: Use half the cores
    extraction_futures = []
    final_results_data = []

    try:
        # --- Submit CPU-bound extraction tasks ---
        start_cpu_submit_time = time.time()
        num_result_frames = len(results_list)
        if num_result_frames == 0:
            print(f"CPU Task: No results found for {video_stem}.")
            extraction_executor.shutdown(wait=False) # No tasks submitted
            return True # Success (processed but no detections)

        for frame, result in enumerate(results_list):
            future = extraction_executor.submit(_extract_frame_data, frame, result, model_names)
            extraction_futures.append(future)
        submit_duration = time.time() - start_cpu_submit_time
        print(f"CPU Task ({video_stem}): Submitted {len(extraction_futures)} frames for extraction in {submit_duration:.2f}s. Aggregating...")

        # --- Wait for extraction and aggregate results with tqdm ---
        start_aggregate_time = time.time()
        # Wrap as_completed with tqdm for progress
        cpu_progress_bar = tqdm(
            concurrent.futures.as_completed(extraction_futures),
            total=len(extraction_futures),
            desc=f"CPU: {video_stem[:25]:<25}", # Shortened description
            unit="frame",
            ncols=100,
            leave=False # Keep the bar until the function finishes
        )
        for future in cpu_progress_bar:
            try:
                frame_data = future.result()
                if frame_data:
                    final_results_data.extend(frame_data)
            except Exception as e:
                print(f"CPU Task ({video_stem}): Error retrieving result from extraction future: {e}")
        aggregate_duration = time.time() - start_aggregate_time
        # print(f"\r{' ' * 100}\r", end='') # Optional: Clear line
        print(f"CPU Task ({video_stem}): Aggregation done in {aggregate_duration:.2f}s.")

        # --- Sort, Create DataFrame, and Save CSV ---
        if not final_results_data:
            print(f"CPU Task ({video_stem}): No detections found after extraction.")
            extraction_executor.shutdown(wait=True) # Ensure shutdown even if no data
            return True # Success

        start_save_time = time.time()
        try:
            final_results_data.sort(key=lambda row: row[0])
            # Update DataFrame columns to include 'Confidence'
            df = pd.DataFrame(final_results_data, columns=['Frame', 'ID', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height'])
            csv_filename = f"{video_stem}_detections.csv"
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)
            save_duration = time.time() - start_save_time
            print(f"CPU Task ({video_stem}): Successfully saved {len(df)} detections to {csv_path} in {save_duration:.2f}s.")
            return True # Success

        except Exception as e:
            save_duration = time.time() - start_save_time
            print(f"CPU Task ({video_stem}): Error saving CSV after {save_duration:.2f}s: {e}")
            return False # Failure

    except Exception as e:
        print(f"CPU Task ({video_stem}): Unexpected error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False # Failure
    finally:
        # --- Ensure local extraction executor is shut down ---
        # Wait=True ensures all submitted tasks complete before moving on
        extraction_executor.shutdown(wait=True)
        print(f"CPU Task: Finished processing for {video_stem}")


# --- Global Pool Variables ---
gpu_pool = None
cpu_executor = None
cpu_futures = [] # To track CPU tasks for shutdown

# --- Callback Function ---
def submit_cpu_task(gpu_result):
    """Callback executed in main process when a GPU task finishes."""
    if cpu_executor:
        if gpu_result:
            video_stem = gpu_result[0] # Get stem for logging
            print(f"Main: GPU task for {video_stem} completed. Submitting CPU task.")
            future = cpu_executor.submit(cpu_process_and_save, gpu_result, output_dir)
            cpu_futures.append(future)
        else:
            # Handle GPU task failure if needed (e.g., log which video failed)
            print("Main: GPU task failed, not submitting CPU task.")
    else:
        print("Main: CPU executor not available, cannot submit CPU task.")


# --- Signal Handler ---
def signal_handler(sig, frame):
    """Handles SIGINT and SIGTSTP signals to terminate the pool and executor."""
    print(f'\nSignal {sig} received, terminating processes and threads...')
    if cpu_executor:
        print("Shutting down CPU executor (no new tasks)...")
        # Consider wait=False, cancel_futures=True for faster exit on signal
        cpu_executor.shutdown(wait=False, cancel_futures=True)
    if gpu_pool:
        print("Terminating GPU worker pool...")
        gpu_pool.terminate()
        gpu_pool.join()
    print("Pools/Executors terminated.")
    sys.exit(1)

_main = None
# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    video_dir = Path('videos/trimmed')
    output_dir = Path('output_csvs_12x') # Ensure output_dir is defined before callback uses it
    model_path = 'yolo12x.pt'
    num_gpus_to_use = 4

    # --- Preparations ---
    main_start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    if available_gpus < num_gpus_to_use:
        print(f"Error: Requested {num_gpus_to_use} GPUs, but only {available_gpus} are available.")
        exit()

    # Find video files
    video_files = sorted([f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]) # Sort for consistent order
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        exit()
    print(f"Found {len(video_files)} videos to process using {num_gpus_to_use} GPUs.")

    # --- Register Signal Handlers ---
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # --- Multiprocessing & Threading Setup ---
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'.")
        else:
            print("Multiprocessing start method already set to 'spawn'.")
    except RuntimeError as e:
         current_method = mp.get_start_method(allow_none=True)
         print(f"Warning: Could not set start method to 'spawn': {e}. Using current method '{current_method}'.")

    results_summary = {} # Track overall success/failure per video
    try:
        print(f"Creating GPU process pool with size {num_gpus_to_use}")
        globals()['gpu_pool'] = mp.Pool(processes=num_gpus_to_use)

        # Create ThreadPoolExecutor for CPU tasks (saving)
        # With 40 cores, allow more concurrent CPU/IO tasks.
        # Let's try 16, allowing significant overlap. Tune based on observation.
        cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16) # Increased from num_gpus_to_use + 2
        globals()['cpu_executor'] = cpu_executor

        # --- Distribute GPU Tasks ---
        gpu_tasks_submitted = 0
        print("Submitting GPU tasks to process pool...")
        for i, video_path in enumerate(video_files):
            gpu_id = i % num_gpus_to_use
            # Submit GPU task with callback to trigger CPU task
            gpu_pool.apply_async(gpu_process_video,
                                 args=(video_path, gpu_id, model_path),
                                 callback=submit_cpu_task)
            gpu_tasks_submitted += 1

        # --- Wait for GPU Pool Completion (Implicitly triggers CPU tasks) ---
        print(f"Waiting for {gpu_tasks_submitted} GPU tasks to complete...")
        gpu_pool.close() # No more GPU tasks will be submitted
        gpu_pool.join()  # Wait for all submitted GPU tasks to finish
        print("All GPU tasks completed.")

        # --- Wait for CPU Tasks Completion ---
        print(f"Waiting for {len(cpu_futures)} CPU tasks to complete...")
        # Use concurrent.futures.wait for better tracking or error handling if needed
        all_cpu_done = False
        while not all_cpu_done:
             all_cpu_done = all(f.done() for f in cpu_futures)
             if not all_cpu_done:
                  time.sleep(0.5)

        # Optionally retrieve results/exceptions from CPU futures
        for future in cpu_futures:
             try:
                  # You might want cpu_process_and_save to return stem and success status
                  # to build the results_summary here. Currently it returns True/False.
                  future.result() # Check for exceptions
             except Exception as e:
                  print(f"Main: Error observed in CPU task future: {e}")

        print("\nAll CPU tasks completed.")

    except Exception as e:
        print(f"\nAn error occurred in the main process: {e}")
    finally:
        # --- Ensure Pool and Executor Cleanup ---
        # GPU Pool is already closed/joined if main try block completed
        if gpu_pool and not gpu_pool._state == mp.pool.CLOSE: # Check if not already closed
             print("Terminating GPU pool (in finally)...")
             gpu_pool.terminate()
             gpu_pool.join()

        if cpu_executor:
            print("Shutting down CPU executor (final)...")
            # Wait=True ensures all tasks finish before exiting script
            cpu_executor.shutdown(wait=True)
            print("CPU executor shut down.")

        # --- Finalization ---
        main_end_time = time.time()
        print(f"\nProcessing finished or interrupted. Total time: {main_end_time - main_start_time:.2f} seconds.")
        print(f"CSV files saved in: {output_dir}")
        # Add summary logic here if cpu_process_and_save returns more info
        failures = [name for name, success in results_summary.items() if not success]
        if failures:
            print(f"The following videos may have failed processing or saving: {failures}")
        else:
            print("All videos processed successfully.")
