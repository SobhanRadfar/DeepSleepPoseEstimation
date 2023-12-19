from Essential import compute_normalized_distances, dist_accuracy, extract_predictions, print_table
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import os


class JointsMeanSquaredError(nn.Module):
    """
        Compute the mean squared error loss between predicted and target joint positions.

        Args:
            predictions (torch.Tensor): Predicted joint positions of shape (batch_size, num_joints, position_dim).
            targets (torch.Tensor): Target joint positions of shape (batch_size, num_joints, position_dim).
            weights (torch.Tensor): Weight values for each joint position of shape (batch_size, num_joints).

        Returns:
            torch.Tensor: Mean squared error loss averaged over all joints in the batch.
    """
    def __init__(self, use_target_weight=True):
        super(JointsMeanSquaredError, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')  # Initialize Mean Squared Error loss function
        self.use_target_weight = use_target_weight # Flag to determine whether to use target weights

    def forward(self, predictions, targets, weights):
        batch_size, num_joints = predictions.shape[:2]

        # Reshape predictions and targets for easier computation
        predictions = predictions.reshape(batch_size, num_joints, -1)
        targets = targets.reshape(batch_size, num_joints, -1)

        total_loss = 0 # Initialize total loss

        for joint_idx in range(num_joints):
            pred_joint = predictions[:, joint_idx]  # Predicted joint positions for the current joint
            target_joint = targets[:, joint_idx] # Target joint positions for the current joint

            if self.use_target_weight: # Compute loss with weighted predictions and targets
                total_loss += 0.5 * self.loss_fn(pred_joint * weights[:, joint_idx], target_joint * weights[:, joint_idx])
            else: # Compute loss without using target weights
                total_loss += 0.5 * self.loss_fn(pred_joint, target_joint)

        # Return the average loss across all joints in the batch
        return total_loss / num_joints



def compute_accuracy(predictions, ground_truth, heatmap_type='gaussian', threshold=0.5):
	"""
    Calculate accuracy according to PCK, using ground truth heatmap rather than x,y locations.
    :param predictions: Predicted heatmaps
    :param ground_truth: Ground truth heatmaps
    :param heatmap_type: Type of heatmap ('gaussian')
    :param threshold: Threshold for determining accuracy
    :return: Array of accuracies for each joint and the overall average, average accuracy, joint count, max predicted values
  """
	num_joints = predictions.shape[1]


  # Extract predicted and target coordinates from heatmaps
	predicted_coords, _ = extract_predictions(predictions)
	target_coords, _ = extract_predictions(ground_truth)

	height, width = predictions.shape[2:4]
	normalization_factor = np.ones((predicted_coords.shape[0], 2)) * np.array([height, width]) / 10

  # Compute normalized distances between predicted and target coordinates
	distances = compute_normalized_distances(predicted_coords, target_coords, normalization_factor)

	joint_accuracies = np.zeros(num_joints + 1)


	total_accuracy = 0
	valid_joint_count = 0
	for i in range(num_joints):
    # Calculate accuracy for each joint
		joint_accuracies[i + 1] = dist_accuracy(distances[i], threshold)
		if joint_accuracies[i + 1] >= 0:
			total_accuracy += joint_accuracies[i + 1]
			valid_joint_count += 1

	average_accuracy = total_accuracy / valid_joint_count if valid_joint_count != 0 else 0
	joint_accuracies[0] = average_accuracy
  # Return accuracy values and other relevant information
	return joint_accuracies, average_accuracy, valid_joint_count, predicted_coords


def compute_pck(errors, joint_visibility, thresholds):
    """
    Compute the Percentage of Correct Keypoints (PCK) for given thresholds.

    :param errors: Errors, preferably normalized. Shape: N x num_joints
    :param joint_visibility: Joint visibility. Set all to 1 if counting all. Shape: N x num_joints
    :param thresholds: Thresholds to evaluate
    :return: PCK values, shape: num_joints x len(thresholds)
    """

    joint_visibility = joint_visibility.squeeze()  # Remove singleton dimensions from joint_visibility
    joint_counts = np.sum(joint_visibility, axis=0)  # Count visible joints for each joint
    joint_ratios = joint_counts / np.sum(joint_counts).astype(np.float64) # Calculate ratios of visible joints

    pck_values = []

    for threshold in thresholds:
        pck_at_threshold = np.zeros(errors.shape[1] + 1) # Initialize PCK values for each joint and overall average
        hits = np.sum((errors <= threshold) * joint_visibility, axis=0) # Count hits within threshold for each joint
        pck_at_threshold[:-1] = hits / joint_counts  # Calculate PCK for each joint
        pck_at_threshold[-1] = np.sum(pck_at_threshold[:-1] * joint_ratios)  # Calculate average PCK using joint ratios
        pck_values.append(pck_at_threshold)

    # Convert PCK values to percentage and transpose for proper shape
    return (np.array(pck_values) * 100).T
