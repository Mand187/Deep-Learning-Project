import config as cfg
import os
import multiprocessing as mp
import torch
from torchtnt.utils.data import CudaDataPrefetcher
from Training.jutils import ColorPrinter, Colors
from Data.data_loading_jaskin import load_and_preprocess_data, create_tensor_from_dataframe, create_sequences, create_dataloaders 
from Training.train_matt import Trainer
from NextNet.model_split import FrameTransformer

from Training.customLoss import ADELoss, FDELoss, RMSELoss, PaddedMSELoss
printer = ColorPrinter()

# %
def train_model(
    model_name,
    train_loader,
    test_loader,
    save_model_dir,
    prediction_length,
    model_kwargs,
    loss_fn,
    learning_rate,
    num_epochs,
    gpu_id,
    optimizer_kwargs={}
):
    # print("Process pool created with 1 processes.") # Removed misleading print
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda:0')
    
    try:
        train_prefetcher = CudaDataPrefetcher(
            data_iterable=train_loader, 
            device=device, 
            num_prefetch_batches=cfg.NUM_BATCHES_TO_PREFETCH
        )
        test_prefetcher = CudaDataPrefetcher(
            data_iterable=test_loader, 
            device=device, 
            num_prefetch_batches=cfg.NUM_BATCHES_TO_PREFETCH
        )
        printer.print(f"[{model_name} GPU:{gpu_id}] Data loading complete.", Colors.CYAN)

        model_file_path = os.path.join(save_model_dir, model_name) 
        model = FrameTransformer(
            input_feature_size=cfg.NUM_INPUT_FEATURES, 
            num_ids=transformer_max_ids_per_frame, 
            sequence_length=X.size(1),  
            prediction_length=prediction_length,
            **model_kwargs
        )
        
        printer.print(f"[{model_name} GPU:{gpu_id}] Attempting to initialize Trainer...", Colors.BLUE)
        trainScript = Trainer(
            model,
            train_prefetcher,
            test_prefetcher,
            save_path=save_model_dir, # Pass the directory for saving models
            model_name=model_name,    # Pass the base model name
            device=device
        )
        printer.print(f"[{model_name} GPU:{gpu_id}] Trainer initialized. Device: {trainScript.device}", Colors.GREEN)

        trainScript.earlyStop(enable=True, patience=30, delta=0.01)
        
        printer.print(f"[{model_name} GPU:{gpu_id}] Attempting to call trainScript.train()...", Colors.BLUE)
        # train_losses, val_losses, _, _, epoch_times = trainScript.train( # Original
        results_tuple = trainScript.train( # Modified to get all return values
            num_epochs=num_epochs, 
            learningRate=learning_rate, 
            criterion=loss_fn, 
            optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate, **optimizer_kwargs)
        )
        printer.print(f"[{model_name} GPU:{gpu_id}] trainScript.train() completed.", Colors.GREEN)
        
        return model # Or more detailed results if needed
    except Exception as e:
        import traceback
        printer.print(f"ERROR in train_model for {model_name} on GPU {gpu_id}: {type(e).__name__}: {e}", Colors.RED)
        # traceback.print_exc() # This prints to stderr of the child process
        error_traceback = traceback.format_exc()
        printer.print(error_traceback, Colors.RED)
        # It's crucial that the callback or result handling in the main process can see this error.
        # Re-raising it is one way if the pool's error handling propagates it.
        # Alternatively, return a specific error object or the traceback string.
        # For now, re-raising will make it appear in the callback if it's an Exception.
        raise # Re-raise the exception to be caught by the pool's error handling / callback


if __name__ == '__main__':
    root_dir = os.getcwd()  # Use current working directory as root
    data_dir = os.path.join(root_dir, 'Data')
    csv_dir = os.path.join(data_dir, 'csv')
    csv_file = os.path.join(csv_dir, 'trimmed_IMG_4097_detections.csv')
    num_gpus_to_use = 4  # Number of GPUs to use
    
    printer.print(f"Initializing data loaders...", Colors.CYAN)
    
    df, transformer_max_ids_per_frame, = load_and_preprocess_data(csv_folder=csv_dir)
    all_data_tensor, num_features = create_tensor_from_dataframe(df, transformer_max_ids_per_frame)
    X, Y = create_sequences(all_data_tensor, prediction_length=cfg.PREDICTION_LENGTH)
    train_loader, test_loader = create_dataloaders(X, Y, num_features=num_features)

    tasks = [
        {
            'model_name': 'ade_model_1s.pth',
            'train_loader': train_loader,
            'test_loader': test_loader,
            'save_model_dir': os.path.join(root_dir, 'Model', 'Saved_Model'),
            'model_kwargs' : {
                'hidden_size': cfg.HIDDEN_SIZE,
                'num_heads': cfg.NUM_HEADS,
                'dropout_rate': cfg.DROPOUT_RATE
            },
            'loss_fn' : FDELoss,
            'optimizer_kwargs' : {
                
            },
            'gpu_id' : 0,
            'learning_rate' : cfg.LEARNING_RATE,
            'num_epochs' : 2,
        }
    ]
    print("Data directory: ", data_dir)
    print("CSV directory: ", csv_dir)
    print("CSV file: ", csv_file)
    num_gpus_to_use = 4  # Number of GPUs to use
    num_gpus_available = torch.cuda.device_count()
    num_gpus_to_use = min(num_gpus_to_use, num_gpus_available)
    print(f"Number of GPUs to use: {num_gpus_to_use}")

    # --- Multiprocessing Setup ---
    pool = None
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
        # globals()['pool'] = mp.Pool(processes=num_gpus_to_use) # Original
        pool = mp.Pool(processes=num_gpus_to_use) # Assign to local variable `pool`

        results_async = [] # To store async results if needed later for .get()

        for i, kwargs in enumerate(tasks):
            printer.print(f"Starting task {i+1}/{len(tasks)} on GPU {kwargs['gpu_id']}", Colors.GREEN)
            # Ensure arguments are passed in the correct order to train_model
            task_args_tuple = (
                kwargs['model_name'],
                kwargs['train_loader'],
                kwargs['test_loader'],
                kwargs['save_model_dir'],
                kwargs['model_kwargs'],
                kwargs['loss_fn'],
                kwargs['learning_rate'],
                kwargs['num_epochs'],
                kwargs['gpu_id'],  # This is the GPU ID the train_model function will use
                kwargs.get('optimizer_kwargs', {}) # Use .get for safety if optimizer_kwargs might be missing
            )
            
            # Capture the correct model_name and gpu_id for the callback
            # to avoid issues with closures in loops.
            model_name_for_callback = kwargs['model_name']
            gpu_id_for_task = kwargs['gpu_id']

            def callback_fn(result_or_exc, name, gpu):
                # This function will be executed in the main process
                if isinstance(result_or_exc, BaseException): # Catch BaseException for broader coverage
                    printer.print(f"CALLBACK ERROR from task {name} on GPU {gpu}: {type(result_or_exc).__name__}: {result_or_exc}", Colors.MAGENTA)
                    # If the exception object has a traceback, it might be limited.
                    # The detailed traceback should have been printed by the child process thanks to the try-except in train_model.
                else:
                    printer.print(f"CALLBACK: Training for {name} completed on GPU {gpu}. Result: {type(result_or_exc)}", Colors.GREEN)

            res = pool.apply_async(
                train_model,
                args=task_args_tuple,
                callback=lambda r, name=model_name_for_callback, gpu=gpu_id_for_task: callback_fn(r, name, gpu)
            )
            results_async.append(res)
            
        # Wait for all tasks to finish
        pool.close()
        pool.join()

        # Optionally, explicitly get results to raise exceptions in the main process if not caught by callback
        # for i, res_async in enumerate(results_async):
        #     try:
        #         task_result = res_async.get()
        #         printer.print(f"Task {i} final result retrieved.", Colors.CYAN)
        #     except Exception as e:
        #         printer.print(f"MAIN THREAD EXCEPTION for task {i} on .get(): {type(e).__name__}: {e}", Colors.RED)

        print("All tasks completed.")
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Terminating processes...")
        if pool:
            pool.terminate()
            pool.join()
        print("Processes terminated.")