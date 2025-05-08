import config as cfg
import traceback
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
class TaskTuple:
    
    def __init__(
        self,
        model_name,
        train_loader,
        test_loader,
        prediction_length,
        num_ids,
        sequence_length,
        save_model_dir,
        model_kwargs,
        loss_fn,
        learning_rate,
        num_epochs,
        optimizer_kwargs,
    ):
            self.model_name = model_name
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.prediction_length = prediction_length
            self.num_ids = num_ids
            self.sequence_length = sequence_length
            self.save_model_dir = save_model_dir
            self.model_kwargs = model_kwargs
            self.loss_fn = loss_fn
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            self.optimizer_kwargs = optimizer_kwargs
    def set_gpu_id(self, gpu_id):
        self.gpu_id = gpu_id
    def get_tuple(self):
        return (
            self.model_name,
            self.train_loader,
            self.test_loader,
            self.prediction_length,
            self.num_ids,
            self.sequence_length,
            self.save_model_dir,
            self.model_kwargs,
            self.loss_fn,
            self.learning_rate,
            self.num_epochs,
            self.gpu_id,
            self.optimizer_kwargs
        )


def train_model(
    model_name,
    train_loader,
    test_loader,
    prediction_length,
    num_ids,
    sequence_length,
    save_model_dir,
    model_kwargs,
    loss_fn,
    learning_rate,
    num_epochs,
    gpu_id,
    optimizer_kwargs
):
    try:
        # Explicitly initialize CUDA for this process
        # This might help with NVML issues in spawned processes
        if torch.cuda.is_available():
            torch.cuda.init() # Initialize CUDA context for the current process
            printer.print(f"[{model_name} GPU:{gpu_id}] CUDA initialized for process.", Colors.BLUE)
        else:
            printer.print(f"[{model_name} GPU:{gpu_id}] CUDA not available for process.", Colors.YELLOW)
            # Depending on requirements, you might want to raise an error or proceed on CPU if possible

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # After setting CUDA_VISIBLE_DEVICES, the device index for PyTorch will be 0 
        # if gpu_id was, for example, 2, but it's the only one visible to this process.
        # So, we should use cuda:0 if a GPU is indeed visible and intended.
        # However, let's first confirm if CUDA is available after init.
        if torch.cuda.is_available(): # Check again after potential init and env var setting
            # device = torch.device(f'cuda:0') # Assuming CUDA_VISIBLE_DEVICES makes this the '0th' visible GPU
            # Let PyTorch pick the current device if CUDA_VISIBLE_DEVICES is set to a single ID
            # Or, if CUDA_VISIBLE_DEVICES is a list, and gpu_id is an index into that original list,
            # this needs careful handling. For now, let's assume gpu_id is the one to use.
            # The safest is to let PyTorch determine based on CUDA_VISIBLE_DEVICES.
            # If CUDA_VISIBLE_DEVICES is set to a single GPU ID, say "2", then cuda:0 in this process refers to GPU 2.
            device = torch.device(f'cuda:{gpu_id}') 
        else:
            device = torch.device('cpu') # Fallback to CPU if CUDA isn't usable
        
        printer.print(f"[{model_name} GPU:{gpu_id}] Using device: {device}", Colors.CYAN)
    except Exception as e:
        printer.print(f"ERROR during initial setup in train_model for {model_name} on GPU {gpu_id}: {type(e).__name__}: {e}", Colors.RED)
        error_traceback = traceback.format_exc()
        printer.print(error_traceback, Colors.RED)
        raise
    
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

        model = FrameTransformer(
            input_feature_size=cfg.NUM_INPUT_FEATURES, 
            num_ids=num_ids, 
            sequence_length=sequence_length,  
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
        printer.print(f"ERROR in train_model for {model_name} on GPU {gpu_id}: {type(e).__name__}: {e}", Colors.ORANGE)
        # traceback.print_exc() # This prints to stderr of the child process
        error_traceback = traceback.format_exc()
        printer.print(error_traceback, Colors.RED)
        # It's crucial that the callback or result handling in the main process can see this error.
        # Re-raising it is one way if the pool's error handling propagates it.
        # Alternatively, return a specific error object or the traceback string.
        # For now, re-raising will make it appear in the callback if it's an Exception.
        raise # Re-raise the exception to be caught by the pool's error handling / callback


def main():
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

    """
    model_name,
    train_loader,
    test_loader,
    prediction_length,
    num_ids,
    sequence_length,
    save_model_dir,
    model_kwargs,
    loss_fn,
    learning_rate,
    num_epochs,
    gpu_id,
    optimizer_kwargs={}
    """
    tasks = [
        TaskTuple(
            model_name='ade_model_1s',
            train_loader=train_loader,
            test_loader=test_loader,
            prediction_length=cfg.PREDICTION_LENGTH,
            num_ids=transformer_max_ids_per_frame,
            sequence_length=X.size(1),
            save_model_dir=os.path.join(root_dir, 'Model', 'Saved_Model'),
            model_kwargs={
                'hidden_size': cfg.HIDDEN_SIZE,
                'num_heads': cfg.NUM_HEADS,
                'dropout_rate': cfg.DROPOUT_RATE
            },
            loss_fn=ADELoss(),
            learning_rate=cfg.LEARNING_RATE,
            num_epochs=50,
            optimizer_kwargs={}
        ),
        TaskTuple(
            model_name='fde_model_1s',
            train_loader=train_loader,
            test_loader=test_loader,
            prediction_length=cfg.PREDICTION_LENGTH,
            num_ids=transformer_max_ids_per_frame,
            sequence_length=X.size(1),
            save_model_dir=os.path.join(root_dir, 'Model', 'Saved_Model'),
            model_kwargs={
                'hidden_size': cfg.HIDDEN_SIZE,
                'num_heads': cfg.NUM_HEADS,
                'dropout_rate': cfg.DROPOUT_RATE
            },
            loss_fn=FDELoss(),
            learning_rate=cfg.LEARNING_RATE,
            num_epochs=50,
            optimizer_kwargs={}
        ),
        TaskTuple(
            model_name='rmse_model_1s',
            train_loader=train_loader,
            test_loader=test_loader,
            prediction_length=cfg.PREDICTION_LENGTH,
            num_ids=transformer_max_ids_per_frame,
            sequence_length=X.size(1),
            save_model_dir=os.path.join(root_dir, 'Model', 'Saved_Model'),
            model_kwargs={
                'hidden_size': cfg.HIDDEN_SIZE,
                'num_heads': cfg.NUM_HEADS,
                'dropout_rate': cfg.DROPOUT_RATE
            },
            loss_fn=RMSELoss(),
            learning_rate=cfg.LEARNING_RATE,
            num_epochs=50,
            optimizer_kwargs={}
        ),
        TaskTuple(
            model_name='mse_model_1s',
            train_loader=train_loader,
            test_loader=test_loader,
            prediction_length=cfg.PREDICTION_LENGTH,
            num_ids=transformer_max_ids_per_frame,
            sequence_length=X.size(1),
            save_model_dir=os.path.join(root_dir, 'Model', 'Saved_Model'),
            model_kwargs={
                'hidden_size': cfg.HIDDEN_SIZE,
                'num_heads': cfg.NUM_HEADS,
                'dropout_rate': cfg.DROPOUT_RATE
            },
            loss_fn=PaddedMSELoss(),
            learning_rate=cfg.LEARNING_RATE,
            num_epochs=50,
            optimizer_kwargs={}
        ),
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

        for i, task in enumerate(tasks):
            task.set_gpu_id(i % num_gpus_to_use)  # Assign GPU ID in a round-robin fashion
            printer.print(f"Starting task {i+1}/{len(tasks)} on GPU {task.gpu_id}", Colors.GREEN)
            # Ensure arguments are passed in the correct order to train_model
                #kwargs.values()
                # )
            # train_model(*task_args_tuple)
            # exit()
            
            # Capture the correct model_name and gpu_id for the callback
            # to avoid issues with closures in loops.
            model_name_for_callback = task.model_name
            gpu_id_for_task = task.gpu_id

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
                args=task.get_tuple(),  # Unpack the tuple to pass as arguments
                callback=lambda r, name=model_name_for_callback, gpu=gpu_id_for_task: callback_fn(r, name, gpu)
            )
            results_async.append(res)
            
        # Wait for all tasks to finish
        pool.close()
        pool.join()

        # Explicitly get results to raise/catch exceptions from tasks in the main process
        printer.print("Retrieving results/exceptions from tasks...", Colors.CYAN)
        for i, res_async in enumerate(results_async):
            try:
                task_result = res_async.get() # This will re-raise exceptions from the child process
                printer.print(f"Task {tasks[i].model_name} (GPU {tasks[i].gpu_id}) final result retrieved successfully.", Colors.CYAN)
            except Exception as e:
                # This catches exceptions that occurred in the train_model function and were propagated by apply_async
                printer.print(f"MAIN PROCESS EXCEPTION for task {tasks[i].model_name} (GPU {tasks[i].gpu_id}) on .get(): {type(e).__name__}: {e}", Colors.YELLOW)
                error_traceback = traceback.format_exc()
                printer.print(error_traceback, Colors.RED)
                # If the error_traceback was part of the exception object (it usually isn't directly),
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Terminating processes...")
        if pool:
            pool.terminate()
            pool.join()
        print("Processes terminated.")
if __name__ == '__main__':
    main()