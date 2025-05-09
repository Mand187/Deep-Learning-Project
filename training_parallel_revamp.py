import config as cfg
import wandb
import traceback
import os
import torch
import torch.multiprocessing as mp # Use torch.multiprocessing
from torch.utils.data import TensorDataset, DataLoader
from torchtnt.utils.data import CudaDataPrefetcher
from Training.jutils import ColorPrinter, Colors, wandb_login
from Data.data_loading_jaskin import load_and_preprocess_data, create_tensor_from_dataframe, create_sequences, VehiclePositionDataset # Removed create_dataloaders from here
from Training.train_matt import Trainer
from NextNet.model_split import FrameTransformer
from Training.customLoss import ADELoss, FDELoss, RMSELoss, PaddedMSELoss
from datetime import datetime
import time # Added time import

printer = ColorPrinter()

# Removed TaskTuple class

# Function to be executed by each worker process
def worker_function(
    model_name,
    X_train_data, # Actual tensor
    Y_train_data, # Actual tensor
    X_test_data,  # Actual tensor
    Y_test_data,  # Actual tensor
    num_features, # Needed for DataLoader creation (though not directly used if features are part of X_train_data)
    prediction_length,
    num_ids,
    sequence_length,
    save_model_dir,
    model_kwargs,
    loss_fn_class_name,
    loss_fn_reduction,
    common_loss_fn_class_name,
    common_loss_fn_reduction,
    learning_rate,
    num_epochs,
    gpu_id,
    optimizer_kwargs,
    wandb_project_name,
    wandb_group_name,
    cfg_num_workers,
    cfg_train_batch_size,
    cfg_test_batch_size,
    cfg_pin_memory,
    cfg_num_input_features
):
    run = None
    try:
        # 1. Setup device
        device = torch.device(f"cuda:{gpu_id}") # PyTorch will see the assigned GPU as cuda:0
        printer.print(f"[{model_name} GPU:{gpu_id}] Worker started. Using device: {str(device)}", Colors.CYAN)

        # 2. W&B Init
        try:
            run = wandb.init(
                project=wandb_project_name,
                group=wandb_group_name,
                name=f"{model_name}-gpu{gpu_id}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}", # Unique name
                config={
                    "model_name": model_name, "learning_rate": learning_rate, "num_epochs": num_epochs,
                    "gpu_id": gpu_id, "prediction_length": prediction_length, "num_ids": num_ids,
                    "sequence_length": sequence_length, "batch_size": cfg_train_batch_size, "num_workers": cfg_num_workers,
                    **model_kwargs, **optimizer_kwargs,
                    "loss_function": loss_fn_class_name, "common_loss_function": common_loss_fn_class_name
                },
                reinit='create_new'
            )
            printer.print(f"[{model_name} GPU:{gpu_id}] W&B run initialized: {run.name}", Colors.BLUE)
        except Exception as e:
            printer.print(f"[{model_name} GPU:{gpu_id}] Failed to initialize W&B: {e}", Colors.RED)
            run = None

        # 3. Create DataLoaders and Prefetchers
        train_dataset = VehiclePositionDataset(
            X_train_data, 
            Y_train_data, 
            num_features=num_features,
        )
        test_dataset = VehiclePositionDataset(
            X_test_data,
            Y_test_data,
            num_features=num_features,
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg_train_batch_size, 
            prefetch_factor=cfg.NUM_TRAIN_BATCHES_TO_PREFETCH,
            shuffle=True, 
            num_workers=cfg.NUM_WORKERS * 2 // 3, 
            pin_memory=cfg_pin_memory, 
            persistent_workers=True if cfg_num_workers > 0 else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg_test_batch_size, 
            prefetch_factor=cfg.NUM_TEST_BATCHES_TO_PREFETCH,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS // 3,
            pin_memory=cfg_pin_memory,
            persistent_workers=True if cfg_num_workers > 0 else False
        )
        
        printer.print(f"[{model_name} GPU:{gpu_id}] DataLoaders created with {cfg_num_workers} workers.", Colors.GREEN)

        train_prefetcher = CudaDataPrefetcher(
            train_loader,
            device,
            num_prefetch_batches=cfg.NUM_TRAIN_BATCHES_TO_PREFETCH,
        )
        test_prefetcher = CudaDataPrefetcher(
            test_loader,
            device,
            num_prefetch_batches=cfg.NUM_TEST_BATCHES_TO_PREFETCH,
        )
        printer.print(f"[{model_name} GPU:{gpu_id}] CudaDataPrefetchers created.", Colors.GREEN)

        # 4. Instantiate Loss Functions
        loss_fn_map = {"ADELoss": ADELoss, "FDELoss": FDELoss, "RMSELoss": RMSELoss, "PaddedMSELoss": PaddedMSELoss}
        selected_loss_fn = loss_fn_map[loss_fn_class_name](reduction=loss_fn_reduction)
        selected_common_loss_fn = loss_fn_map[common_loss_fn_class_name](reduction=common_loss_fn_reduction)
        printer.print(f"[{model_name} GPU:{gpu_id}] Loss functions instantiated.", Colors.GREEN)

        # 5. Instantiate Model
        model = FrameTransformer(
            input_feature_size=cfg_num_input_features,
            num_ids=num_ids,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            **model_kwargs
        ).to(device)
        printer.print(f"[{model_name} GPU:{gpu_id}] Model FrameTransformer instantiated on {device}.", Colors.GREEN)

        # 6. Instantiate Trainer
        trainScript = Trainer(
            model,
            train_prefetcher,
            test_prefetcher,
            save_path=save_model_dir,
            model_name=model_name,
            device=device,
            wandb_run=run
        )
        trainScript.earlyStop(enable=True, patience=cfg.EARLY_STOPPING_PATIENCE, delta=cfg.EARLY_STOPPING_DELTA)
        printer.print(f"[{model_name} GPU:{gpu_id}] Trainer initialized.", Colors.GREEN)

        # 7. Train
        printer.print(f"[{model_name} GPU:{gpu_id}] Starting training for {num_epochs} epochs...", Colors.BLUE)
        results_tuple = trainScript.train(
            num_epochs=num_epochs,
            learningRate=learning_rate,
            criterion=selected_loss_fn,
            optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate, **optimizer_kwargs),
            common_loss_fn=selected_common_loss_fn
        )
        printer.print(f"[{model_name} GPU:{gpu_id}] Training completed. Results: {results_tuple}", Colors.GREEN)

    except Exception as e:
        printer.print(f"ERROR in worker_function for {model_name} on GPU {gpu_id}: {type(e).__name__}: {e}", Colors.RED)
        error_traceback = traceback.format_exc()
        printer.print(error_traceback, Colors.RED)
        if run:
            run.log({"error": str(e), "traceback": error_traceback})
            run.finish(exit_code=1) # Finish with error code
        # Re-raise so main process join() can potentially detect non-zero exit if needed,
        # though direct exception passing across Process is not straightforward without Queues.
        # For now, logging is the primary error reporting.
        # raise # Avoid re-raising directly as it might terminate the main script if not handled by mp.Process
    finally:
        if run and run._exit_code is None: # Only finish if not already finished with an error
            run.finish()
            printer.print(f"[{model_name} GPU:{gpu_id}] W&B run finished successfully.", Colors.BLUE)
        printer.print(f"[{model_name} GPU:{gpu_id}] Worker finished.", Colors.CYAN)


def main():
    try:
        wandb_login() # Call W&B login once in the main process
    except Exception as e:
        printer.print(f"W&B login failed: {e}. Proceeding without W&B.", Colors.RED)

    wandb_project_name = "Deep-Learning-Project-Refactor"
    wandb_group_name = f"parallel_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'Data')
    csv_dir = os.path.join(data_dir, 'csv')
    
    printer.print(f"Initializing data loading...", Colors.CYAN)
    df, transformer_max_ids_per_frame = load_and_preprocess_data(csv_folder=csv_dir)
    all_data_tensor, num_features = create_tensor_from_dataframe(df, transformer_max_ids_per_frame)
    printer.print(f"Data loaded. All data tensor shape: {all_data_tensor.shape}, Num features: {num_features}", Colors.GREEN)

    # Create all data variants in the main process
    data_store = {}
    prediction_lengths_secs = [1, 2, 3, 4] # Example prediction lengths in seconds
    for secs in prediction_lengths_secs:
        pred_len_frames = 30 * secs
        X_data, Y_data = create_sequences(all_data_tensor, prediction_length=pred_len_frames)
        data_store[f"X_{secs}s"] = X_data
        data_store[f"Y_{secs}s"] = Y_data
        printer.print(f"Created sequences for {secs}s: X shape {X_data.shape}, Y shape {Y_data.shape}", Colors.BLUE)

    # Define task configurations
    task_configs = []
    model_types = [
        {"name": "rmse_model", "loss": "RMSELoss", "common_loss": "ADELoss"},
        {"name": "ade_model", "loss": "ADELoss", "common_loss": "RMSELoss"}, # Example, adjust as needed
        # Add other model types if necessary
    ]

    for model_type_info in model_types:
        for secs in prediction_lengths_secs:
            model_name = f"{model_type_info['name']}_{secs}s"
            X_key = f"X_{secs}s"
            Y_key = f"Y_{secs}s"
            
            if X_key not in data_store or Y_key not in data_store:
                printer.print(f"Data for {secs}s not found in data_store. Skipping task {model_name}.", Colors.YELLOW)
                continue

            current_X_data = data_store[X_key]

            task_configs.append({
                "model_name": model_name,
                "X_train_data_key": X_key,
                "Y_train_data_key": Y_key,
                "X_test_data_key": X_key, # Using same data for train/test split, adjust if you have separate test tensors
                "Y_test_data_key": Y_key,
                "num_features": num_features,
                "prediction_length": 30 * secs,
                "num_ids": transformer_max_ids_per_frame,
                "sequence_length": current_X_data.size(1), # Get from actual tensor
                "save_model_dir": os.path.join(root_dir, 'Model', 'Saved_Model_Refactor'),
                "model_kwargs": {'hidden_size': cfg.HIDDEN_SIZE, 'num_heads': cfg.NUM_HEADS, 'dropout_rate': cfg.DROPOUT_RATE},
                "loss_fn_class_name": model_type_info["loss"],
                "loss_fn_reduction": "mean",
                "common_loss_fn_class_name": model_type_info["common_loss"],
                "common_loss_fn_reduction": "mean",
                "learning_rate": cfg.LEARNING_RATE,
                "num_epochs": cfg.EPOCHS, # Use from config
                "optimizer_kwargs": {}, # Add if any specific optimizer kwargs are needed
                "cfg_num_workers": cfg.NUM_WORKERS,
                "cfg_train_batch_size": cfg.TRAIN_BATCH_SIZE, # Explicitly name for clarity
                "cfg_test_batch_size": cfg.TEST_BATCH_SIZE,   # Add test batch size
                "cfg_pin_memory": cfg.PIN_MEMORY,
                "cfg_num_input_features": cfg.NUM_INPUT_FEATURES
            })

    num_gpus_available = torch.cuda.device_count()
    num_gpus_to_use = min(cfg.NUM_GPUS_TO_USE, num_gpus_available)
    if num_gpus_to_use == 0:
        printer.print("No GPUs available or configured for use. Exiting.", Colors.RED)
        return
    printer.print(f"Number of GPUs to use: {num_gpus_to_use}", Colors.CYAN)

    # --- Multiprocessing Setup ---
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            printer.print("Set multiprocessing start method to 'spawn'.", Colors.GREEN)
        else:
            printer.print("Multiprocessing start method already 'spawn'.", Colors.YELLOW)
    except RuntimeError as e:
        printer.print(f"Warning: Could not set start method to 'spawn': {e}. Current method: {mp.get_start_method(allow_none=True)}.", Colors.YELLOW)

    if not task_configs:
        printer.print("No tasks configured to run. Exiting.", Colors.RED)
        return

    active_processes_info = []  # Stores {'process': p, 'gpu_id': gpu_id, 'task_name': name}
    task_queue_indices = list(range(len(task_configs)))
    available_gpu_ids = list(range(num_gpus_to_use))
    completed_task_count = 0

    printer.print(f"Starting training for {len(task_configs)} tasks using {num_gpus_to_use} GPUs.", Colors.CYAN)

    while completed_task_count < len(task_configs):
        # Try to launch new processes if GPUs are available and tasks are waiting
        while available_gpu_ids and task_queue_indices:
            gpu_id_to_use = available_gpu_ids.pop(0)
            current_task_idx = task_queue_indices.pop(0)
            task_conf = task_configs[current_task_idx]

            args_for_worker = (
                task_conf["model_name"],
                data_store[task_conf["X_train_data_key"]],
                data_store[task_conf["Y_train_data_key"]],
                data_store[task_conf["X_test_data_key"]],
                data_store[task_conf["Y_test_data_key"]],
                task_conf["num_features"],
                task_conf["prediction_length"],
                task_conf["num_ids"],
                task_conf["sequence_length"],
                task_conf["save_model_dir"],
                task_conf["model_kwargs"],
                task_conf["loss_fn_class_name"],
                task_conf["loss_fn_reduction"],
                task_conf["common_loss_fn_class_name"],
                task_conf["common_loss_fn_reduction"],
                task_conf["learning_rate"],
                task_conf["num_epochs"],
                gpu_id_to_use, # Actual physical GPU index to use
                task_conf["optimizer_kwargs"],
                wandb_project_name,
                wandb_group_name,
                task_conf["cfg_num_workers"],
                task_conf["cfg_train_batch_size"],
                task_conf["cfg_test_batch_size"],
                task_conf["cfg_pin_memory"],
                task_conf["cfg_num_input_features"]
            )

            printer.print(f"Preparing to start task {task_conf['model_name']} on GPU {gpu_id_to_use}", Colors.BLUE)
            p = mp.Process(target=worker_function, args=args_for_worker)
            p.start()
            active_processes_info.append({
                'process': p, 
                'gpu_id': gpu_id_to_use, 
                'task_name': task_conf['model_name'],
                'pid': p.pid
            })
            printer.print(f"Started task {task_conf['model_name']} on GPU {gpu_id_to_use} (Process PID: {p.pid})", Colors.GREEN)

        # Check for completed processes
        next_active_processes_info = []
        for proc_info in active_processes_info:
            p = proc_info['process']
            gpu_id = proc_info['gpu_id']
            task_name = proc_info['task_name']
            pid = proc_info['pid']

            if not p.is_alive(): # Check if process has terminated
                # p.join() # Ensure it's properly joined to get exit code and clean up resources
                exitcode = p.exitcode # Get exitcode after checking is_alive and before join (join might block if called too early)
                p.join() # Now join to clean up
                printer.print(f"Process for task {task_name} (PID: {pid}) on GPU {gpu_id} finished. Exit code: {exitcode}", Colors.GREEN if exitcode == 0 else Colors.YELLOW)
                available_gpu_ids.append(gpu_id)  # Free up the GPU
                completed_task_count += 1
            else:
                next_active_processes_info.append(proc_info)  # Keep it in the list of active processes
        active_processes_info = next_active_processes_info

        if completed_task_count == len(task_configs):
            break # All tasks done

        time.sleep(1)  # Sleep for a short duration to avoid busy waiting

    printer.print(f"All {len(task_configs)} training tasks completed.", Colors.BOLD_GREEN)

if __name__ == '__main__':
    # Ensure the script can be imported without automatically running main()
    # This is good practice for multiprocessing with 'spawn' start method
    # as child processes might re-import the main module.
    main()