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


# --- Global dicts for worker process models/GPUs (keyed by PID) ---
worker_models = {}
worker_gpu_ids = {}

# --- GPU Task Function ---
def gpu_process_video(video_path, model_path, gpu_id): # Added gpu_id back
    """
    Processes a single video file using a lazily-loaded model on the assigned GPU.
    Uses stream=True and shows progress with tqdm.
    Returns (video_stem, results_list, model_names) on success, None on failure.
    """
    global worker_models, worker_gpu_ids # Access global dicts
    process_name = mp.current_process().name
    pid = os.getpid()
    print(f"{process_name} (PID {pid}, GPU {gpu_id}): Starting GPU task for {video_path.name}")
    results_list = [] # Initialize list to store results

    # --- Lazy Model Loading --- 
    model = None
    if pid not in worker_models or worker_gpu_ids.get(pid) != gpu_id:
        print(f"{process_name} (PID {pid}, GPU {gpu_id}): Loading model {model_path}...")
        try:
            # Set CUDA_VISIBLE_DEVICES for this specific process
            # Important: This should ideally happen *before* any CUDA calls if not using initializer
            # It might be less reliable here than in an initializer, but let's try.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # Even after setting visible devices, torch might still see all if initialized before.
            # Explicitly setting device index might be needed.
            # device = f'cuda:{gpu_id}' # Try explicit device ID first
            device = 'cuda:0' # If CUDA_VISIBLE_DEVICES works, it should be cuda:0
            print(f"{process_name} (PID {pid}, GPU {gpu_id}): Attempting to load model onto {device} (Visible: {os.environ['CUDA_VISIBLE_DEVICES']})...")

            model = YOLO(model_path).to(device)
            # Optional warm-up
            # model(torch.zeros(1, 3, 640, 640).to(device))

            worker_models[pid] = model
            worker_gpu_ids[pid] = gpu_id
            print(f"{process_name} (PID {pid}, GPU {gpu_id}): Model loaded successfully onto {model.device}.")
        except Exception as e:
            print(f"Error loading model for {process_name} (PID {pid}, GPU {gpu_id}): {e}")
            import traceback
            traceback.print_exc()
            # Clear env var if set? Might affect subsequent calls in same process if task fails.
            # if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
            return None # Failed to load model
    else:
        model = worker_models[pid]
        print(f"{process_name} (PID {pid}, GPU {gpu_id}): Using pre-loaded model on {model.device}.")

    if model is None: # Should not happen if logic above is correct, but safety check
        print(f"Error: Model is None for {process_name} (PID {pid}, GPU {gpu_id}) after loading attempt.")
        return None

    try:
        device = model.device # Use the device the model is actually on

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

        model_names = model.names # Get model names once

        # --- Run tracking (GPU intensive) ---
        start_gpu_time = time.time()
        results_generator = model.track(
            source=str(video_path),
            show=False,
            save=False,
            half=True,
            stream=True,
            device=device,
            verbose=False,
            batch=512 # Keep removed
        )

        # --- Iterate with tqdm progress bar ---
        progress_bar = tqdm(
            iterable=results_generator,
            total=total_frames,
            desc=f"GPU {gpu_id}: {video_path.stem[:20]:<20}",
            position=gpu_id, # Position based on assigned gpu_id
            leave=True,
            unit="frame",
            ncols=100
        )

        # --- RE-ENABLED result collection ---
        frame_count = 0
        for result in progress_bar:
            results_list.append(result) # Collect results again
            frame_count += 1
        # --- END RE-ENABLE ---

        # --- GPU processing finished ---
        gpu_duration = time.time() - start_gpu_time
        print(f"{process_name} (PID {pid}, GPU {gpu_id}): GPU processing done for {video_path.name} ({frame_count} frames) in {gpu_duration:.2f}s.")

        # --- Return collected results ---
        return video_path.stem, results_list, model_names # Original return

    except Exception as e:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        print(f"Error in GPU task {process_name} (PID {pid}, Visible GPU(s): {cuda_visible}, Assigned: {gpu_id}) for {video_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Don't delete the model here, it's stored globally for the process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{process_name} (PID {pid}, GPU {gpu_id}): GPU task finished for {video_path.name}")


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

# --- Helper function for mapping extraction tasks ---
def _extraction_helper(args):
    """Unpacks arguments for _extract_frame_data for use with executor.map."""
    frame_index, result, model_names = args
    # Call the original extraction function
    return _extract_frame_data(frame_index, result, model_names)

# --- CPU Task Function ---
def cpu_process_and_save(gpu_result_tuple, output_dir):
    """
    Takes results from gpu_process_video, extracts data in parallel using
    a local ThreadPoolExecutor, creates DataFrame, and saves CSV.
    Runs in the main cpu_executor.
    """
    if gpu_result_tuple is None:
        print("CPU Task: Received None from GPU task, skipping.")
        return False # Indicate failure

    video_stem, results_list, model_names = gpu_result_tuple
    print(f"CPU Task: Starting processing for {video_stem}")

    # --- Use a local ThreadPoolExecutor for frame extraction parallelism ---
    # Define number of workers for frame extraction within this task
    extraction_workers = 8 # Or adjust based on testing/heuristics
    final_results_data = []

    try:
        start_cpu_extract_time = time.time()
        num_result_frames = len(results_list)
        if num_result_frames == 0:
            print(f"CPU Task: No results found for {video_stem}.")
            return True # Success (processed but no detections)

        # --- Prepare arguments for mapping ---
        # We pass the index, the result object, and model_names for each frame
        map_args = [(idx, res, model_names) for idx, res in enumerate(results_list)]

        print(f"CPU Task ({video_stem}): Starting parallel extraction for {num_result_frames} frames using {extraction_workers} workers...")

        # --- Parallelize extraction using the local executor and map ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=extraction_workers) as executor:
            # executor.map returns an iterator. We consume it into a list.
            # Each item in results_iterator will be the list returned by _extract_frame_data
            results_iterator = executor.map(_extraction_helper, map_args)

            # --- Flatten the list of lists efficiently --- 
            # tqdm can be wrapped around the iterator for progress
            extraction_progress = tqdm(
                results_iterator,
                total=num_result_frames,
                desc=f"CPU Extract: {video_stem[:20]:<20}",
                unit="frame",
                ncols=100,
                leave=False
            )
            final_results_data = [row for frame_rows in extraction_progress if frame_rows for row in frame_rows]
            # ---------------------------------------------

        extract_duration = time.time() - start_cpu_extract_time
        print(f"CPU Task ({video_stem}): Parallel extraction done in {extract_duration:.2f}s.")

        # --- Sort, Create DataFrame, and Save CSV (remains the same) ---
        if not final_results_data:
            print(f"CPU Task ({video_stem}): No detections found after extraction.")
            return True # Success

        start_save_time = time.time()
        try:
            # Sorting might be slightly more expensive now, but necessary
            final_results_data.sort(key=lambda row: row[0])
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
        # Local executor is managed by the 'with' statement
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
    total_cores = os.cpu_count() or 40 # Get total cores, default to 40 if detection fails
    cpu_workers = total_cores - num_gpus_to_use if total_cores > num_gpus_to_use else 1 # Calculate workers for CPU pool
    print(f"Total Cores: {total_cores}, GPUs: {num_gpus_to_use}, CPU Workers: {cpu_workers}")

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

        # --- Create GPU Pool WITHOUT Initializer --- 
        globals()['gpu_pool'] = mp.Pool(
            processes=num_gpus_to_use
        )

        # Create ThreadPoolExecutor for CPU tasks (saving & extraction)
        # Allocate remaining cores after reserving one per GPU process
        cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) # ADJUSTED
        globals()['cpu_executor'] = cpu_executor

        # --- Distribute GPU Tasks ---
        gpu_tasks_submitted = 0
        print("Submitting GPU tasks to process pool...")
        for i, video_path in enumerate(video_files):
            # Assign GPU ID based on task index modulo number of GPUs
            assigned_gpu_id = i % num_gpus_to_use
            print(f"Submitting {video_path.name} to GPU {assigned_gpu_id}")
            gpu_pool.apply_async(gpu_process_video,
                                 args=(video_path, model_path, assigned_gpu_id), # Pass assigned_gpu_id
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
