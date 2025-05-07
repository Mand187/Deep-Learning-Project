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
    prediction_length,
    model_kwargs,
    loss_fn,
    learning_rate,
    num_epochs,
    gpu_id,
    optimizer_kwargs={}
):
    root_dir = os.getcwd()  # Use current working directory as root
    data_dir = os.path.join(root_dir, 'Data')
    csv_dir = os.path.join(data_dir, 'csv')
    csv_file = os.path.join(csv_dir, 'trimmed_IMG_4097_detections.csv')
    num_gpus_to_use = 4  # Number of GPUs to use

    print("Data directory: ", data_dir)
    print("CSV directory: ", csv_dir)
    print("CSV file: ", csv_file)


    model_dir = os.path.join(root_dir, 'Model')
    save_model_dir = os.path.join(model_dir, 'Saved_Model')
    # print("Model directory: ", model_dir)
    # print("Saved model directory: ", save_model_dir)
    model_name = 'ade_model_1s.pth'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = 'cuda:0'
    
    df, transformer_max_ids_per_frame, = load_and_preprocess_data(csv_folder=csv_dir)

    # 2. Create tensor from dataframe
    all_data_tensor, num_features = create_tensor_from_dataframe(df, transformer_max_ids_per_frame)

    # 3. Create input-output sequences
    X, Y = create_sequences(all_data_tensor, prediction_length=prediction_length)

    # 4. Create dataloaders for training and testing
    train_loader, test_loader = create_dataloaders(X, Y, num_features=num_features)
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
    print("done data loading")
    model_file_path = os.path.join(save_model_dir, model_name)  # Add a file name
    model = FrameTransformer(
        input_feature_size=cfg.NUM_INPUT_FEATURES, 
        num_ids=transformer_max_ids_per_frame, 
        sequence_length=X.size(1),  
        prediction_length=prediction_length,
        **model_kwargs
    )
    
    # Initialize the trainer
    trainScript = Trainer(
        model,
        train_prefetcher,
        test_prefetcher,
        model_name=model_name,
        model_path=model_file_path,
        device=device
    )  # Load the model from the specified path

    # Early stopping
    trainScript.earlyStop(enable=True, patience=30, delta=0.01)
    
    # Train the model
    train_losses, val_losses, _, _, epoch_times = trainScript.train(
        num_epochs=num_epochs, 
        learningRate=learning_rate, 
        criterion=loss_fn, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate, **optimizer_kwargs)
    )
    
    return model
tasks = [
    {
        'model_name': 'ade_model_1s.pth',
        'prediction_length' : 30,
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

if __name__ == '__main__':
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
        globals()['pool'] = mp.Pool(processes=min(num_gpus_to_use, len(tasks)))
        print(f"Process pool created with {min(num_gpus_to_use, len(tasks))} processes.")
        
        for i, kwargs in enumerate(tasks):
            printer.print(f"Starting task {i+1}/{len(tasks)} on GPU {kwargs['gpu_id']}", Colors.GREEN)
            # Ensure arguments are passed in the correct order to train_model
            task_args_tuple = (
                kwargs['model_name'],
                kwargs['prediction_length'],
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

            pool.apply_async(
                train_model,
                args=task_args_tuple,
                callback=lambda result, name=model_name_for_callback, gpu=gpu_id_for_task: print(f"Training for {name} completed on GPU {gpu}")
            )
            
        # Wait for all tasks to finish
        pool.close()
        pool.join()
        print("All tasks completed.")
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Terminating processes...")
        if pool:
            pool.terminate()
            pool.join()
        print("Processes terminated.")