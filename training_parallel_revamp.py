import config as cfg
import wandb
import traceback
import os
import torch
import torch.multiprocessing as mp # Use torch.multiprocessing
from torch.utils.data import TensorDataset, DataLoader
from torchtnt.utils.data import CudaDataPrefetcher
from Training.jutils import ColorPrinter, Colors, wandb_login
from Data.data_loading_jaskin import load_and_preprocess_data, create_tensor_from_dataframe, create_sequences, VehiclePositionDataset
from Training.train_matt import Trainer
from NextNet.model_split import FrameTransformer
from Training.customLoss import ADELoss, FDELoss, RMSELoss, PaddedMSELoss
from datetime import datetime
import time

printer = ColorPrinter()

# Function to be executed by each worker process
def worker_function(
    run_id, # W&B run ID to resume
    assigned_gpu_id, # GPU assigned to this worker
    X_train_data, # Actual tensor
    Y_train_data, # Actual tensor
    X_test_data,  # Actual tensor
    Y_test_data   # Actual tensor
):
    run = None
    model_name_for_print = "Worker" # Fallback name for printer
    try:
        # 1. Setup device
        device = torch.device(f"cuda:{assigned_gpu_id}")
        # Initialize W&B run first to get config for printer and other settings
        # Resume the run initialized by the main process
        run = wandb.init(id=run_id, resume="must")
        if not run:
            printer.print(f"[GPU:{assigned_gpu_id}] Failed to resume W&B run with ID: {run_id}", Colors.RED)
            return # Cannot proceed without W&B config

        # Now that run is resumed, wandb.config is populated
        cfg_wandb = wandb.config
        model_name_for_print = cfg_wandb.model_name

        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] Worker started. Resumed W&B run: {run.name}. Using device: {str(device)}", Colors.CYAN)

        # 2. Create DataLoaders and Prefetchers
        train_dataset = VehiclePositionDataset(
            X_train_data,
            Y_train_data,
            num_features=cfg_wandb.num_features,
        )
        test_dataset = VehiclePositionDataset(
            X_test_data,
            Y_test_data,
            num_features=cfg_wandb.num_features,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg_wandb.cfg_train_batch_size,
            prefetch_factor=cfg_wandb.cfg_num_train_batches_to_prefetch,
            shuffle=True,
            num_workers=cfg_wandb.cfg_num_workers * 2 // 3 if cfg_wandb.cfg_num_workers > 0 else 0,
            pin_memory=cfg_wandb.cfg_pin_memory,
            persistent_workers=True if cfg_wandb.cfg_num_workers > 0 else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg_wandb.cfg_test_batch_size,
            prefetch_factor=cfg_wandb.cfg_num_test_batches_to_prefetch,
            shuffle=False,
            num_workers=cfg_wandb.cfg_num_workers // 3 if cfg_wandb.cfg_num_workers > 0 else 0,
            pin_memory=cfg_wandb.cfg_pin_memory,
            persistent_workers=True if cfg_wandb.cfg_num_workers > 0 else False
        )

        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] DataLoaders created with {cfg_wandb.cfg_num_workers} workers.", Colors.GREEN)

        train_prefetcher = CudaDataPrefetcher(
            train_loader,
            device,
            num_prefetch_batches=cfg_wandb.cfg_num_train_batches_to_prefetch,
        )
        test_prefetcher = CudaDataPrefetcher(
            test_loader,
            device,
            num_prefetch_batches=cfg_wandb.cfg_num_test_batches_to_prefetch,
        )
        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] CudaDataPrefetchers created.", Colors.GREEN)

        # 3. Instantiate Loss Functions
        loss_fn_map = {"ADELoss": ADELoss, "FDELoss": FDELoss, "RMSELoss": RMSELoss, "PaddedMSELoss": PaddedMSELoss}
        selected_loss_fn = loss_fn_map[cfg_wandb.loss_fn_class_name](reduction=cfg_wandb.loss_fn_reduction)
        selected_common_loss_fn = loss_fn_map[cfg_wandb.common_loss_fn_class_name](reduction=cfg_wandb.common_loss_fn_reduction)
        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] Loss functions instantiated.", Colors.GREEN)

        # 4. Instantiate Model
        # model_kwargs from wandb.config might be a wandb.sdk.lib.config.ConfigDict, convert to dict if necessary
        model_kwargs_dict = dict(cfg_wandb.model_kwargs) if hasattr(cfg_wandb.model_kwargs, 'items') else {}

        model = FrameTransformer(
            input_feature_size=cfg_wandb.cfg_num_input_features,
            num_ids=cfg_wandb.num_ids,
            sequence_length=cfg_wandb.sequence_length,
            prediction_length=cfg_wandb.prediction_length,
            **model_kwargs_dict
        ).to(device)
        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] Model FrameTransformer instantiated on {device}. Params: {sum(p.numel() for p in model.parameters())}", Colors.GREEN)


        # 5. Instantiate Trainer
        trainScript = Trainer(
            model,
            train_prefetcher,
            test_prefetcher,
            save_path=cfg_wandb.save_model_dir,
            model_name=cfg_wandb.model_name, # Trainer uses this for print statements and file paths
            device=device,
            wandb_run=run # Pass the resumed run object to Trainer
        )
        trainScript.earlyStop(
            enable=True,
            patience=cfg_wandb.cfg_early_stopping_patience,
            delta=cfg_wandb.cfg_early_stopping_delta
        )
        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] Trainer initialized.", Colors.GREEN)

        # 6. Train
        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] Starting training for {cfg_wandb.num_epochs} epochs...", Colors.BLUE)
        # optimizer_kwargs from wandb.config might be a wandb.sdk.lib.config.ConfigDict
        optimizer_kwargs_dict = dict(cfg_wandb.optimizer_kwargs) if hasattr(cfg_wandb.optimizer_kwargs, 'items') else {}
        results_tuple = trainScript.train(
            num_epochs=cfg_wandb.num_epochs,
            learningRate=cfg_wandb.learning_rate,
            criterion=selected_loss_fn,
            optimizer=torch.optim.AdamW(model.parameters(), lr=cfg_wandb.learning_rate, **optimizer_kwargs_dict),
            common_loss_fn=selected_common_loss_fn
        )
        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] Training completed. Results: {results_tuple}", Colors.GREEN)

    except Exception as e:
        error_context = f"ERROR in worker_function for {model_name_for_print} (Run ID: {run_id}) on GPU {assigned_gpu_id}"
        printer.print(f"{error_context}: {type(e).__name__}: {e}", Colors.RED)
        error_traceback = traceback.format_exc()
        printer.print(error_traceback, Colors.RED)
        if run: # Log error to W&B if run was initialized
            run.log({"error_type": type(e).__name__, "error_message": str(e), "traceback": error_traceback})
            run.finish(exit_code=1, quiet=True) # Finish with error code
        # No re-raise, allow process to terminate. Main process will see exit code.
    finally:
        if run and run._exit_code is None: # Only finish if not already finished (e.g. by an error)
            run.finish(quiet=True) # Default exit_code is 0
            printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] W&B run {run.name} finished successfully.", Colors.BLUE)
        elif not run:
             printer.print(f"[GPU:{assigned_gpu_id}] Worker finished, W&B run was not properly initialized/resumed for Run ID: {run_id}.", Colors.YELLOW)
        printer.print(f"[{model_name_for_print} GPU:{assigned_gpu_id}] Worker finished.", Colors.CYAN)


def main():
    try:
        wandb_login() # Call W&B login once in the main process
    except Exception as e:
        printer.print(f"W&B login failed: {e}. Proceeding without W&B (if possible).", Colors.RED)

    wandb_project_name = "Deep-Learning-Project-Refactor"
    wandb_group_name = f"parallel_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'Data')
    csv_dir = os.path.join(data_dir, 'csv')
    
    printer.print(f"Initializing data loading...", Colors.CYAN)
    df, transformer_max_ids_per_frame = load_and_preprocess_data(csv_folder=csv_dir)
    all_data_tensor, num_features_global = create_tensor_from_dataframe(df, transformer_max_ids_per_frame) # Renamed to avoid conflict
    printer.print(f"Data loaded. All data tensor shape: {all_data_tensor.shape}, Num features: {num_features_global}", Colors.GREEN)

    data_store = {}
    prediction_lengths_secs = [1, 2, 3, 4]
    for secs in prediction_lengths_secs:
        pred_len_frames = 30 * secs
        X_data, Y_data = create_sequences(all_data_tensor, prediction_length=pred_len_frames)
        data_store[f"X_{secs}s"] = X_data
        data_store[f"Y_{secs}s"] = Y_data
        printer.print(f"Created sequences for {secs}s: X shape {X_data.shape}, Y shape {Y_data.shape}", Colors.BLUE)

    task_configs_params = [] # Store dicts that will form the basis of wandb.config
    model_types = [
        {"name": "rmse_model", "loss": "RMSELoss", "common_loss": "ADELoss"},
        {"name": "ade_model", "loss": "ADELoss", "common_loss": "RMSELoss"},
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

            # Parameters that will go into wandb.config
            params_for_config = {
                "model_name": model_name,
                "X_data_key": X_key, # For traceability, not direct use by worker from config
                "Y_data_key": Y_key, # For traceability
                "num_features": num_features_global, # From global data loading
                "prediction_length": 30 * secs,
                "num_ids": transformer_max_ids_per_frame,
                "sequence_length": current_X_data.size(1),
                "save_model_dir": os.path.join(root_dir, 'Model', 'Saved_Model_Refactor'),
                "model_kwargs": {'hidden_size': cfg.HIDDEN_SIZE, 'num_heads': cfg.NUM_HEADS, 'dropout_rate': cfg.DROPOUT_RATE},
                "loss_fn_class_name": model_type_info["loss"],
                "loss_fn_reduction": "mean",
                "common_loss_fn_class_name": model_type_info["common_loss"],
                "common_loss_fn_reduction": "mean",
                "learning_rate": cfg.LEARNING_RATE,
                "num_epochs": cfg.EPOCHS,
                "optimizer_kwargs": {},
                "cfg_num_workers": cfg.NUM_WORKERS,
                "cfg_train_batch_size": cfg.TRAIN_BATCH_SIZE,
                "cfg_test_batch_size": cfg.TEST_BATCH_SIZE,
                "cfg_pin_memory": cfg.PIN_MEMORY,
                "cfg_num_input_features": cfg.NUM_INPUT_FEATURES, # This is num_features_global
                # Add other cfg values needed by worker
                "cfg_num_train_batches_to_prefetch": cfg.NUM_TRAIN_BATCHES_TO_PREFETCH,
                "cfg_num_test_batches_to_prefetch": cfg.NUM_TEST_BATCHES_TO_PREFETCH,
                "cfg_early_stopping_patience": cfg.EARLY_STOPPING_PATIENCE,
                "cfg_early_stopping_delta": cfg.EARLY_STOPPING_DELTA,
            }
            task_configs_params.append(params_for_config)

    num_gpus_available = torch.cuda.device_count()
    num_gpus_to_use = min(cfg.NUM_GPUS_TO_USE, num_gpus_available)
    if num_gpus_to_use == 0:
        printer.print("No GPUs available or configured for use. Exiting.", Colors.RED)
        return
    printer.print(f"Number of GPUs to use: {num_gpus_to_use}", Colors.CYAN)

    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            printer.print("Set multiprocessing start method to 'spawn'.", Colors.GREEN)
        else:
            printer.print("Multiprocessing start method already 'spawn'.", Colors.YELLOW)
    except RuntimeError as e:
        printer.print(f"Warning: Could not set start method to 'spawn': {e}. Current method: {mp.get_start_method(allow_none=True)}.", Colors.YELLOW)

    if not task_configs_params:
        printer.print("No tasks configured to run. Exiting.", Colors.RED)
        return

    active_processes_info = []
    task_queue_indices = list(range(len(task_configs_params)))
    available_gpu_ids = list(range(num_gpus_to_use))
    completed_task_count = 0

    printer.print(f"Starting training for {len(task_configs_params)} tasks using {num_gpus_to_use} GPUs.", Colors.CYAN)

    while completed_task_count < len(task_configs_params):
        while available_gpu_ids and task_queue_indices:
            gpu_id_to_use = available_gpu_ids.pop(0)
            current_task_idx = task_queue_indices.pop(0)
            
            # This dictionary contains all parameters for the worker, to be set in wandb.config
            current_task_wandb_config = task_configs_params[current_task_idx].copy()
            current_task_wandb_config["assigned_gpu_id"] = gpu_id_to_use # Worker needs to know its assigned GPU
            current_task_wandb_config["wandb_project_name"] = wandb_project_name # For reference in config
            current_task_wandb_config["wandb_group_name"] = wandb_group_name   # For reference in config

            run_id = wandb.util.generate_id()
            run_name = f"{current_task_wandb_config['model_name']}-gpu{gpu_id_to_use}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Pre-initialize W&B run in the main process to set config
            temp_run = wandb.init(
                project=wandb_project_name,
                group=wandb_group_name,
                name=run_name,
                id=run_id,
                config=current_task_wandb_config,
                reinit=True, # Important for loop
                settings=wandb.Settings(start_method="thread") # Avoids issues with multiple inits
            )
            if temp_run:
                temp_run.finish() # Finish immediately, config is now set for this run_id
                printer.print(f"Pre-initialized W&B run {run_name} (ID: {run_id}) for task {current_task_wandb_config['model_name']} on GPU {gpu_id_to_use}", Colors.MAGENTA)
            else:
                printer.print(f"Failed to pre-initialize W&B run for task {current_task_wandb_config['model_name']}", Colors.RED)
                # Decide how to handle: skip task, retry, or exit? For now, will attempt to proceed.
                # The worker's resume="must" will fail if pre-init failed.

            # Arguments for the worker process
            args_for_worker = (
                run_id,
                gpu_id_to_use,
                data_store[current_task_wandb_config["X_data_key"]],
                data_store[current_task_wandb_config["Y_data_key"]],
                data_store[current_task_wandb_config["X_data_key"]], # Assuming X_test_data_key is same as X_train_data_key
                data_store[current_task_wandb_config["Y_data_key"]], # Assuming Y_test_data_key is same as Y_train_data_key
            )

            printer.print(f"Preparing to start task {current_task_wandb_config['model_name']} (Run ID: {run_id}) on GPU {gpu_id_to_use}", Colors.BLUE)
            p = mp.Process(target=worker_function, args=args_for_worker)
            p.start()
            active_processes_info.append({
                'process': p,
                'gpu_id': gpu_id_to_use,
                'task_name': current_task_wandb_config['model_name'],
                'pid': p.pid,
                'run_id': run_id
            })
            printer.print(f"Started task {current_task_wandb_config['model_name']} (PID: {p.pid}, Run ID: {run_id}) on GPU {gpu_id_to_use}", Colors.GREEN)

        next_active_processes_info = []
        for proc_info in active_processes_info:
            p = proc_info['process']
            if not p.is_alive():
                exitcode = p.exitcode
                p.join() # Clean up
                printer.print(f"Process for task {proc_info['task_name']} (PID: {proc_info['pid']}, Run ID: {proc_info['run_id']}) on GPU {proc_info['gpu_id']} finished. Exit code: {exitcode}", Colors.GREEN if exitcode == 0 else Colors.YELLOW)
                available_gpu_ids.append(proc_info['gpu_id'])
                completed_task_count += 1
            else:
                next_active_processes_info.append(proc_info)
        active_processes_info = next_active_processes_info

        if completed_task_count == len(task_configs_params):
            break
        time.sleep(1)

    printer.print(f"All {len(task_configs_params)} training tasks completed.", Colors.BOLD_GREEN)

if __name__ == '__main__':
    main()