from Essential import reverse_flip, tensor_to_cv2, map_to_original_coordinates, normalized_distance, print_table
from plot import save_2d_skeletons, save_visualization
from metrics import compute_pck, compute_accuracy, JointsMeanSquaredError
from torch.utils.data import DataLoader
from dataset import dataset_reader
from StackedHourGlass import StackedHourGlass
from os import path
import numpy as np
import torch
import json



def train(train_loader, val_loader, model, loss_func, optimizer, epoch, params, max_iterations=-1, use_gpu=False):
    # Set the model to training mode
    model.train()

    # Initialize lists to store loss and accuracy values
    loss_values = []
    accuracy_values = []

    # Loop through batches of data from the training DataLoader
    for i, batch_data in enumerate(train_loader):
        # Break the loop if the number of iterations exceeds max_iterations
        if i >= max_iterations and max_iterations > 0:
            break

        # Extract features and labels from the batch_data
        features = batch_data['pch'] #torch.Size([30, 1, 256, 256])
        labels = batch_data['hms']  #torch.Size([30, 14, 64, 64])
        label_weights = batch_data['joints_vis'] #torch.Size([30, 14, 1])

        # Move the data to GPU if use_gpu is True
        if use_gpu:
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            label_weights = label_weights.cuda(non_blocking=True)

        # Forward pass: Compute the model's output
        predictions = model(features) #predictions[0] shape: torch.Size([30, 14, 64, 64])

        # Compute loss using the loss function
        loss = 0
        if isinstance(predictions, list):
            for pred in predictions:
                loss += loss_func(pred, labels, label_weights)
        else: # If the model returns a single output
            loss = loss_func(predictions, labels, label_weights)

        # Zero out gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the accuracy of the model's predictions
        if isinstance(predictions, list):
            last_prediction = predictions[-1].detach().cpu().numpy()
        else:
            last_prediction = predictions.detach().cpu().numpy()

        _, avg_acc, _, _ = compute_accuracy(last_prediction, labels.detach().cpu().numpy())
        
        # Print training statistics every 'logging_frequency' iterations
        if i % params.logging_frequency == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss: {loss:.5f}\tAccuracy: {avg_acc:.3f}')

        # Store the computed loss and accuracy
        loss_values.append(loss.item())
        accuracy_values.append(avg_acc)

    # Return a dictionary containing lists of loss values and accuracy values
    return {'losses': loss_values, 'accs': accuracy_values}




def validate(loader, dataset_reader, model, loss_function, epoch, phase, params, max_iterations=-1, save_visuals=False, use_gpu=False):

  """
    Validate the model using the provided data loader and dataset reader.

    Args:
    - loader (DataLoader): Data loader for the validation set.
    - dataset_reader: Contains dataset-specific attributes and methods.
    - model (nn.Module): The neural network model to validate.
    - loss_function: The loss function to evaluate predictions.
    - epoch (int): Current training epoch.
    - phase (str): Training phase (e.g., 'train', 'val', etc.).
    - params: Contains training parameters and configurations.
    - max_iterations (int, optional): Max number of batches to validate. Default is -1, which processes all batches.
    - save_visuals (bool, optional): Flag to save visualization of predictions. Default is False.
    - use_gpu (bool, optional): Flag to use GPU for computations. Default is False.

    Returns:
    - dict: Results containing predicted coordinates, actual joint coordinates, errors, etc.
  """
  # Set model to evaluation mode
  model.eval()
    
  # Initialization
  total_samples = dataset_reader.n_smpl #540
  num_joints = dataset_reader.joint_num_ori #14

  # Lists to store results
  pred_heatmaps, bounding_boxes, actual_joints, joint_visibilities, std_errors = [], [], [], [], []
  losses, accuracies = [], []
    
  # Ensure no gradient computation for faster validation
  with torch.no_grad():
      for i, data in enumerate(loader):
          # Break loop if max_iterations is reached
          if max_iterations > 0 and i >= max_iterations:
            break

          input_image, heatmap_targets, joint_visibility, bbox, joints, std_err = (data['pch'], data['hms'], data['joints_vis'], data['bb'], data['joints_ori'], data['l_std_ori']) # Unpack data from the current batch
          #input_image: Size([30, 1, 256, 256]), 
          #heatmap_targets: Size([30, 14, 64, 64]), 
          #joint_visibility: Size([30, 14, 1]), 
          #bbox: Size([30, 4]), 
          #joints: Size([30, 14, 2]), 
          #std_err: Size([30])

          # Forward pass through the model
          predictions = model(input_image)
          final_predictions = predictions[-1] if isinstance(predictions, list) else predictions
            
          # Test using flipped images if specified
          if params.flip_test:
            # Flip input images horizontally
            flipped_input = input_image.flip(3).clone() #Size([30, 1, 256, 256])
            flipped_predictions = model(flipped_input)
            final_flipped_predictions = flipped_predictions[-1] if isinstance(flipped_predictions, list) else flipped_predictions
                
            # Adjust the predictions to original orientation
            flipped_predictions_adjusted = reverse_flip(final_flipped_predictions.cpu().numpy(), dataset_reader.flip_pairs).copy()
            final_flipped_predictions = torch.from_numpy(flipped_predictions_adjusted).cuda() if params.use_gpu else torch.from_numpy(flipped_predictions_adjusted)
                
            if True:  # Replacing 'if_shiftHM'
              final_flipped_predictions[:, :, :, 1:] = final_flipped_predictions.clone()[:, :, :, :-1]

            # Average with original prediction
            final_predictions = (final_predictions + final_flipped_predictions) * 0.5

          # Move targets to GPU if specified
          if use_gpu:
            heatmap_targets, joint_visibility = heatmap_targets.cuda(non_blocking=True), joint_visibility.cuda(non_blocking=True)

          # Compute loss and accuracy for the current batch
          loss = loss_function(final_predictions, heatmap_targets, joint_visibility)
          _, avg_acc, _, pred_heatmap = compute_accuracy(final_predictions.cpu().numpy(), heatmap_targets.cpu().numpy())
            
          # Append results to lists
          pred_heatmaps.append(pred_heatmap)
          bounding_boxes.append(bbox.numpy())
          actual_joints.append(joints.numpy())
          joint_visibilities.append(joint_visibility.cpu().numpy())
          std_errors.append(std_err.numpy())
          losses.append(loss)
          accuracies.append(avg_acc)
            
          # Save visuals if specified
          if save_visuals:
            save_visualization(params, dataset_reader, input_image, pred_heatmap, joints, phase, i, epoch)

          # Print progress for every specified frequency
          if i % params.logging_frequency == 0:
            print(f'Test: [{i}/{len(loader)}]\tLoss {loss:.5f}\tAccuracy {avg_acc:.3f}')
      # Compile results for all batches
      results = compile_results(pred_heatmaps, bounding_boxes, actual_joints, joint_visibilities, std_errors, params, losses, accuracies)
      # results['pred_coordinates']: (540, 14, 2)
      # results['actual_joints']: (540, 14, 2)
      # results['std_errors_all']: (540,)
      # results['normalized_err']: (540, 14)
      # results['pck']: (15, 11)
      # results['losses']: (18,)
      # results['accs']: (18,)
  return results


def initialize_checkpoint(pars, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint file if it exists.

    Args:
    - pars: Contains configurations and training parameters.
    - model (nn.Module): The neural network model to be loaded.
    - optimizer (Optimizer): The optimizer whose state needs to be loaded.

    Returns:
    - tuple: Various training states such as epoch, training/val losses and accuracies, performance, etc.
             If no checkpoint is found, returns defaults indicating the start of training.
    """

    # Define the path where the checkpoint is or would be saved
    checkpoint_path = path.join(pars.model_dir, 'checkpoint.pth')

    # If it's the start of training or no checkpoint is found, return default values
    if pars.start_epoch == 0 or not path.exists(checkpoint_path):
        return 0, [], [], [], [], 0.0, -1
    
    # Notify that the checkpoint is being loaded
    print(f"=> loading checkpoint '{checkpoint_path}'")

    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path)

    # Update model and optimizer with the saved states
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Notify that the checkpoint has been successfully loaded
    print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")

    # Return saved training states from the checkpoint
    return (checkpoint['epoch'], checkpoint['losses_train'], checkpoint['accs_train'],checkpoint['losses_val'], checkpoint['accs_val'], checkpoint['perf'], checkpoint['epoch'])



def update_performance_metrics(test_results, test_dataset):
    """
    Calculate and return performance metrics based on validation/testing results.
    """
    pck_all = test_results['pck']
    perf_indicator = pck_all[-1][-1]  # the last entry is the performance indicator
    pckh05 = np.array(pck_all)[:,-1]  # the last column represents PCKh@0.5
    
    # For better visual representation, you could print or store these in logs
    joint_names = list(test_dataset.joints_name[:test_dataset.joint_num_ori]) + ['total']
    print_table([pckh05], joint_names, ['pckh0.5'], print_function=print)
    
    return perf_indicator

def update_best_performance(current_perf, best_perf):
    """
    Compares the current performance indicator against the best performance so far.
    
    Parameters:
        current_perf (float): The current performance indicator.
        best_perf (float): The best performance indicator so far.
        
    Returns:
        float: The new best performance indicator.
        bool: Flag indicating whether the current model is the best one.
    """
    
    is_best = False
    if current_perf >= best_perf:
        best_perf = current_perf
        is_best = True
    
    return best_perf, is_best


def save_checkpoint(pars, model, optimizer, epoch, is_best, best_perf, train_losses, train_accs, val_losses, val_accs):
    """
    Save the current training state, including the epoch, model state, optimizer state, and performance metrics.
    
    Parameters:
        pars (object): Parameters object that includes settings and configurations.
        model (PyTorch model): The model currently being trained.
        optimizer (PyTorch optimizer): The optimizer for the model.
        epoch (int): The current training epoch.
        is_best (bool): Flag indicating whether the current model is the best one.
        best_perf (float): The best performance indicator so far.
        train_losses (list): List of training losses.
        train_accs (list): List of training accuracies.
        val_losses (list): List of validation losses.
        val_accs (list): List of validation accuracies.
    """
    
    checkpoint = {
        'epoch': epoch + 1,
        'model': pars.model,
        'state_dict': model.module.state_dict(),
        'best_state_dict': model.module.state_dict() if is_best else None,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
        'losses_train': train_losses,
        'accs_train': train_accs,
        'losses_val': val_losses,
        'accs_val': val_accs
    }
    
    checkpoint_path = path.join(pars.model_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_model_path = path.join(pars.model_dir, 'model_best.pth')
        torch.save(checkpoint, best_model_path)
    
    print(f'=> saving checkpoint to {checkpoint_path}')


def run_test_and_save_results(pars, dataset_reader, model, criterion, test_name, n_iter, phase='test', covers=None):
    # Update data path and cover list if specified
    if covers is not None:
        pars.cover_list = covers

    print(f'---------run final test {test_name}-----------')
    result_path = path.join(pars.rst_dir, f'SLP-{test_name}_exp.json')

    test_dataset = dataset_reader(pars, phase=phase, zero_base=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=pars.batch_size,
                             shuffle=False,
                             pin_memory=pars.pin_memory)

    test_results = validate(loader=test_loader,
                            dataset_reader=test_dataset,
                            model=model,
                            loss_function=criterion,
                            epoch=100,
                            phase='val',
                            params=pars,
                            max_iterations=n_iter,
                            save_visuals=True,
                            use_gpu=pars.use_gpu)

    pck_all = test_results['pck']
    pckh05 = np.array(pck_all)[:, -1]
    titles_c = list(test_dataset.joints_name[:test_dataset.joint_num_ori]) + ['total']

    print_table([pckh05], titles_c, ['pckh0.5'], print_function=print)
    #test_results['losses'] = [item.item() for item in test_results['losses']]

    with open(result_path, 'w') as f:
        json.dump(test_results, f)


def compile_results(pred_heatmaps, bounding_boxes, actual_joints, joint_visibilities, std_errors, params, losses, accuracies):
  """
    Compile results from the lists of predictions, ground truths, and other metrics into a consolidated dictionary.

    Args:
    - pred_heatmaps (list of numpy arrays): Predicted heatmaps for each batch.
    - bounding_boxes (list of numpy arrays): Ground truth bounding boxes for each batch.
    - actual_joints (list of numpy arrays): Ground truth joint coordinates for each batch.
    - joint_visibilities (list of numpy arrays): Visibility status of each joint for each batch.
    - std_errors (list of numpy arrays): Standard error values for each batch.
    - params: Contains configurations and training parameters.
    - losses (list of floats): Loss values for each batch.
    - accuracies (list of floats): Accuracy values for each batch.

    Returns:
    - dict: A dictionary containing consolidated results including predicted coordinates, actual joint coordinates, errors, etc.
  """
  def concatenate_arrays(array_list):
    """Utility function to concatenate a list of arrays along the first axis."""
    return np.concatenate(array_list, axis=0)
    
  # Concatenate all results for easier processing
  pred_heatmaps, bounding_boxes, actual_joints, joint_visibilities, std_errors_all = map(concatenate_arrays, [pred_heatmaps, bounding_boxes, actual_joints, joint_visibilities, std_errors])

  # Convert heatmap predictions to original image coordinates
  pred_coordinates = map_to_original_coordinates(pred_heatmaps, bounding_boxes, output_size=params.output_shape)

  # Compute the normalized distance error between predictions and actual joints
  normalized_err = normalized_distance(pred_coordinates, actual_joints, std_errors_all)
    
  # Compute Percentage of Correct Keypoints (PCK) for various thresholds
  thresholds = np.linspace(0, 0.5, 11)
  pck_all = compute_pck(normalized_err, joint_visibilities, thresholds)

  # Compile all results into a dictionary for easy access
  return {
        'pred_coordinates': pred_coordinates.tolist(),
        'actual_joints': actual_joints.tolist(),
        'std_errors_all': std_errors_all.tolist(),
        'normalized_err': normalized_err.tolist(),
        'pck': pck_all.tolist(),
        'losses': [loss.item() for loss in losses],
        'accs': accuracies
  }



def test_best_model_on_datasets(pars, criterion=JointsMeanSquaredError(use_target_weight=True).cuda()):

    model = load_best_model(pars, dataset_reader)  # Assume this function loads and returns the best model
    
    results = {}
    
    # Test on 'danaLab'
    print('---------run final test danaLab -----------')
    pars.data_path = 'SLP/danaLab'
    results['danaLab'] = run_test(pars,model, criterion, dataset_reader, 'test', 'SLP/danaLab')
    
    # Test on 'simLab'
    if not 'PM' in pars.source_modes:
        print('---------run final test simLab-----------')
        pars.data_path = 'SLP/simLab'
        results['simLab'] = run_test(pars,model, criterion, dataset_reader, 'test', 'SLP/simLab')
    
    # Test on 'danaLab' with all covers
    if not pars.cover_list == ['uncover', 'cover1', 'cover2']:
      print('---------run final test danaLab all covers-----------')
      pars.cover_list = ['uncover', 'cover1', 'cover2']
      pars.data_path = 'SLP/danaLab'
      results['danaLab_all'] = run_test(pars,model, criterion, dataset_reader, 'test', 'SLP/danaLab')
    
    # Test on 'simLab' with all covers
      if not 'PM' in pars.source_modes:
        print('---------run final test simLab all covers-----------')
        pars.data_path = 'SLP/simLab'
        results['simLab_all'] = run_test(pars,model, criterion, dataset_reader, 'test', 'SLP/simLab')
        
    return results

def run_test(pars,model, criterion, dataset_reader, phase, data_path):

  """
    Runs a testing phase on the given dataset using the specified model.

    Args:
    - pars: Configuration and hyperparameters for the testing process.
    - model (nn.Module): The neural network model to be tested.
    - criterion: The loss function used during the testing phase.
    - dataset_reader: Function to read and preprocess the dataset.
    - phase (str): The current phase (e.g., "train", "test").
    - data_path (str): Path to the dataset.

    Returns:
    - dict: A dictionary containing the results of the testing phase, including metrics like PCK.
  """

  # Read and preprocess the test dataset
  test_dataset = dataset_reader(pars, phase=phase, zero_base=True)

  # Create a DataLoader for the test dataset
  test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=pars.batch_size,
        shuffle=False,
        pin_memory=pars.pin_memory)
  
  # Perform validation (i.e., testing) on the test dataset using the provided model
  result = validate(
        loader=test_loader,
        dataset_reader=test_dataset,
        model=model,
        loss_function=criterion,
        epoch=100,
        params=pars,
        phase=phase,
        max_iterations=pars.trainIter,
        save_visuals=False,
        use_gpu=pars.use_gpu
    )
    
  # Extract the PCK results from the validation outputs
  pck_all = result['pck']
  # Get the PCK value at a threshold of 0.5
  pckh05 = np.array(pck_all)[:, -1]
  # Names of the joints for reporting in the results
  titles_c = list(test_dataset.joints_name[:test_dataset.joint_num_ori]) + ['total']
  # Print the PCK results in a table format
  print_table([pckh05], titles_c, ['pckh0.5'], print_function=print)
    
  return result


def load_best_model(pars, dataset_reader):
    # Initialize the model
    model = StackedHourGlass(
        channel_count=pars.nChannels,stack_count=pars.nStack, module_count=pars.nModules,
        reduction_count=pars.numReductions, input_channels=pars.num_channels,
        joint_count=dataset_reader.joint_num_ori, multi_loss=pars.mu_loss
    )

    # Move to GPU if available
    if pars.use_gpu:
        model = model.cuda()

    # Load the best model checkpoint
    checkpoint_path = path.join(pars.model_dir, 'model_best.pth')
    if path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['best_state_dict'])
    else:
        print("Checkpoint file does not exist. Please ensure you have a valid model_best.pth file.")

    # Wrap the model for multi-GPUs, if necessary, and move it to the right device
    model = torch.nn.DataParallel(model).cuda()
    
    return model