import matplotlib.pyplot as plt
from skimage import io
import torch
import os
import numpy as np
import cv2
import IPython
import random
import math




def read_image(address, modality, cover, sample, frame):
  """
    Reads an image or data file based on the provided parameters.
    
    Parameters:
    - address: Base directory path where the data is stored.
    - modality: Type of data to be read (e.g., 'depthRaw', 'PMarray').
    - cover: Specific sub-directory or category within the modality.
    - sample: Index of the sample.
    - frame: Frame number or specific data file identifier.
    
    Returns:
    - arr: Loaded image or data as a numpy array.
  """
  if modality in ['depthRaw', 'PMarray']:
    file_name_format = '{:06}.npy'
    loader = np.load
  else:
    file_name_format = 'image_{:06d}.png'
    loader = io.imread
  file_path = os.path.join(address, '{:05d}'.format(sample), modality, cover, file_name_format.format(frame))
  arr = np.array(loader(file_path))
  return arr


def generate_perspective_Transformers(path, samples, modalities):
  """
    Generates a dictionary containing perspective transformation matrices for 
    specified modalities based on the provided samples.
    
    Parameters:
    - path: Directory path where transformation matrices are stored.
    - samples: List of sample indices.
    - modalities: List of modalities for which the transformation matrices are to be loaded.
    
    Returns:
    - Perspective_ransformers_dct: A dictionary with modalities as keys and a list of perspective 
                                  transformation matrices as values.
  """

  Perspective_ransformers_dct = {modNm: [] for modNm in modalities}
  for i in samples:
    for mod in modalities:
      mod = uni_mod(mod)
      if 'PM' not in mod:
        pth_PTr = os.path.join(path, '{:05d}'.format(i + 1), 'align_PTr_{}.npy'.format(mod))
        PTr = np.load(pth_PTr)
      else:
        PTr = np.eye(3)
      Perspective_ransformers_dct[mod].append(PTr)
  return Perspective_ransformers_dct


def uni_mod(mod):
    recognized_modalities = ['depth', 'IR', 'PM']
    for recognized_modality in recognized_modalities:
        if recognized_modality in mod:
            return recognized_modality
    return mod



def load_physique_data(dsFd):
    phys_arr = np.load(os.path.join(dsFd, 'physiqueData.npy'))
    phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
    return phys_arr.astype(float)


def init_modality_lists(modalities):
    """
    Initializes a dictionary with lists as values for each modality.
    For each modality in the input list, three keys are added:
    1. Joints ground truth (li_joints_gt_{modality})
    2. Bounding boxes (li_bb_{modality})
    3. Squared bounding boxes (li_bb_sq_{modality})
    
    Parameters:
    - modalities: List of modalities for which the keys need to be created.
    
    Returns:
    - modality_dict: A dictionary with keys based on the provided modalities and empty lists as values.
    """
    modality_dict = {}
    for mod in modalities:
        modality_dict[f'li_joints_gt_{mod}'] = []
        modality_dict[f'li_bb_{mod}'] = []
        modality_dict[f'li_bb_sq_{mod}'] = []
    return modality_dict


def get_bounding_box(joint_img, margin_ratio=1.2, aspect_ratio=0):
    """
    Computes a bounding box for a set of 2D points. The bounding box can be adjusted for margin and aspect ratio.
    
    Parameters:
    - joint_img: An array of 2D points.
    - margin_ratio: Ratio by which to expand the bounding box from its center. Default is 1.2.
    - aspect_ratio: Desired aspect ratio (width / height) of the bounding box. Default is 0 (no adjustment).
    
    Returns:
    - bounding_box: A numpy array in the format [x, y, width, height].
    """

    # Find the minimum and maximum x and y values among the points
    x_min, y_min = np.min(joint_img[:,:2], axis=0)
    x_max, y_max = np.max(joint_img[:,:2], axis=0)

    # Compute the width and height based on the extremes
    width = x_max - x_min - 1
    height = y_max - y_min - 1

    # Initialize an array for the bounding box values
    bounding_box = np.zeros(4)

    if aspect_ratio: # If aspect_ratio is provided
        # Calculate the center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Adjust the width and height to meet the specified aspect ratio
        if width > aspect_ratio * height:
            height = width / aspect_ratio
        elif width < aspect_ratio * height:
            width = height * aspect_ratio


        # Compute bounding box with margins and adjusted aspect ratio
        bounding_box[2] = width * margin_ratio
        bounding_box[3] = height * margin_ratio
        bounding_box[0] = center_x - bounding_box[2] / 2
        bounding_box[1] = center_y - bounding_box[3] / 2
    else: # If no aspect_ratio is provided
        # Compute bounding box with margins
        bounding_box[0] = (x_min + x_max) / 2 - width / 2 * margin_ratio
        bounding_box[1] = (y_min + y_max) / 2 - height / 2 * margin_ratio
        bounding_box[2] = width * margin_ratio
        bounding_box[3] = height * margin_ratio

    return bounding_box



def adjust_bounding_box(bounding_box, aspect_ratio=1):
    """
    Adjust the provided bounding box to maintain the given aspect ratio.

    Parameters:
    - bounding_box: A list or array in the format [x, y, width, height]
    - aspect_ratio: Desired aspect ratio (width / height) of the bounding box

    Returns:
    - New bounding box that maintains the given aspect ratio
    """

    # Extract the coordinates and dimensions from the bounding box
    x, y, width, height = bounding_box

    # Compute the center of the bounding box
    center_x, center_y = x + width / 2, y + height / 2

    # Adjust the width or height of the bounding box to maintain the aspect ratio
    if width > aspect_ratio * height:
        height = width / aspect_ratio
    else:
        width = height * aspect_ratio

    # Calculate the top-left corner of the adjusted bounding box
    adjusted_bb = np.array([center_x - width / 2, center_y - height / 2, width, height])
    return adjusted_bb


def get_augmentation_config():
    """
    Generate configuration values for data augmentation.

    Returns:
    - scale: A scale factor for image resizing
    - rotation: A rotation angle for image rotation
    - flip_image: A boolean indicating whether to flip the image
    - color_adjustment: A list of scaling factors for RGB channels
    - apply_occlusion: A boolean indicating whether to apply occlusion
    """

    scale_deviation = 0.25
    rotation_deviation = 30
    color_deviation = 0.2

    # Calculate scale factor
    scale = 1.0 + np.clip(np.random.randn(), -1.0, 1.0) * scale_deviation

    # Calculate rotation angle
    rotation_chance = 0.6
    if np.random.rand() <= rotation_chance:
        rotation = np.clip(np.random.randn(), -2.0, 2.0) * rotation_deviation
    else:
        rotation = 0

    # Determine if image should be flipped
    flip_chance = 0.5
    flip_image = np.random.rand() <= flip_chance

    # Determine color adjustment factors for RGB channels
    color_range = [1.0 - color_deviation, 1.0 + color_deviation]
    color_adjustment = np.random.uniform(color_range[0], color_range[1], 3)

    # Determine if occlusion should be applied
    occlusion_chance = 0.5
    apply_occlusion = np.random.rand() <= occlusion_chance

    return scale, rotation, flip_image, color_adjustment, apply_occlusion

def generate_image_patch(image, bounding_box, flip, scale_factor, rotation, occlude, patch_size=(256, 256)):
    """
    Generate a transformed image patch from the given image based on the specified bounding box and transformations.
    
    Parameters:
    - image: Input image (numpy array) to extract the patch from.
    - bounding_box: A list or tuple specifying the [x, y, width, height] of the region of interest.
    - flip: Boolean flag to indicate if the image should be horizontally flipped.
    - scale_factor: Factor to scale the region of interest.
    - rotation: Angle (in degrees) for the rotation transformation.
    - occlude: Boolean flag to indicate if a random occlusion should be added to the image patch.
    - patch_size: Tuple indicating the desired size (height, width) of the output patch. Default is (256, 256).
    
    Returns:
    - image_patch: The transformed image patch.
    - transform: Transformation matrix used to generate the image patch.
    """

    patched_image = image.copy() # Create a copy of the original image to avoid modifying it directly.
    image_height, image_width, channels = patched_image.shape

    
    if occlude: # If occlusion is desired, generate a synthetic occlusion patch on the image.
        while True:

            # Define occlusion area ratios and calculate a random area for occlusion.
            min_area_ratio, max_area_ratio = 0.0, 0.7 
            synthetic_area = (random.random() * (max_area_ratio - min_area_ratio) + min_area_ratio) * bounding_box[2] * bounding_box[3]

            # Define aspect ratio boundaries and calculate a random aspect ratio.
            min_ratio, max_ratio = 0.3, 1 / 0.3
            synthetic_ratio = (random.random() * (max_ratio - min_ratio) + min_ratio)
            
            # Calculate the width and height of the synthetic occlusion.
            synthetic_height = math.sqrt(synthetic_area * synthetic_ratio)
            synthetic_width = math.sqrt(synthetic_area / synthetic_ratio)

            # Randomly determine the position (top-left corner) of the occlusion.
            x_offset = random.random() * (bounding_box[2] - synthetic_width - 1) + bounding_box[0]
            y_offset = random.random() * (bounding_box[3] - synthetic_height - 1) + bounding_box[1]

            # Check that the occlusion lies within the image boundaries.
            if 0 <= x_offset < image_width - synthetic_width and 0 <= y_offset < image_height - synthetic_height:

                # Determine exact pixel coordinates.
                x_start, y_start = int(x_offset), int(y_offset)
                width, height = int(synthetic_width), int(synthetic_height)

                # Replace the region in the image with random pixel values, simulating occlusion.
                patched_image[y_start:y_start + height, x_start:x_start + width, :] = np.random.rand(height, width, channels) * 255
                break

    # Calculate the center of the bounding box.
    center_x = bounding_box[0] + 0.5 * bounding_box[2]
    center_y = bounding_box[1] + 0.5 * bounding_box[3]
    box_width, box_height = bounding_box[2], bounding_box[3]

    # If flipping is desired, flip the image horizontally and adjust the bounding box center.
    if flip:
        patched_image = patched_image[:, ::-1, :]
        center_x = image_width - center_x - 1

    # Generate a transformation matrix to warp the image.
    transform = generate_patch_transform(center_x, center_y, box_width, box_height, patch_size[1], patch_size[0], scale_factor, rotation, inverse=False)

    # Apply the transformation to produce the image patch.
    image_patch = cv2.warpAffine(patched_image, transform, (int(patch_size[1]), int(patch_size[0])), flags=cv2.INTER_LINEAR).astype(np.float32)

    return image_patch, transform

def generate_patch_transform(center_x, center_y, original_width, original_height, target_width, target_height, scaling_factor, rotation_angle, inverse=False):
    """
    Generate a transformation matrix to warp a patch from source to target space.

    Parameters:
    - center_x, center_y: The center coordinates of the original patch.
    - original_width, original_height: Dimensions of the original patch.
    - target_width, target_height: Dimensions of the target space.
    - scaling_factor: Scaling factor for size augmentation.
    - rotation_angle: Rotation angle in degrees for rotation augmentation.
    - inverse: Whether to compute the inverse transformation.

    Returns:
    - A transformation matrix.
    """

    # Scale original dimensions
    scaled_width = original_width * scaling_factor
    scaled_height = original_height * scaling_factor

    # Convert rotation angle from degrees to radians
    rotation_rad = np.pi * rotation_angle / 180

    # Calculate rotated directions
    center = np.array([center_x, center_y])
    down_direction = rotate_2d_point(np.array([0, scaled_height * 0.5]), rotation_rad)
    right_direction = rotate_2d_point(np.array([scaled_width * 0.5, 0]), rotation_rad)

    # Set source and destination points
    source_points = np.float32([center, center + down_direction, center + right_direction])

    target_center = np.array([target_width * 0.5, target_height * 0.5])
    destination_points = np.float32([target_center, target_center + np.array([0, target_height * 0.5]), target_center + np.array([target_width * 0.5, 0])])

    # Compute affine transformation
    if inverse:
        transform = cv2.getAffineTransform(destination_points, source_points)
    else:
        transform = cv2.getAffineTransform(source_points, destination_points)

    return transform


def rotate_2d_point(point, angle_rad):
    """
    Rotate a point by a given angle.

    Parameters:
    - point: A 2D point to be rotated.
    - angle_rad: Rotation angle in radians.

    Returns:
    - Rotated 2D point.
    """
    # Extract the x and y coordinates from the point.
    x, y = point

    # Calculate the sine and cosine values of the given rotation angle.
    # These values will be used to compute the new rotated coordinates.
    s, c = np.sin(angle_rad), np.cos(angle_rad)

    # Compute the x-coordinate of the rotated point using the rotation matrix formula.
    rotated_x = x * c - y * s

    # Compute the y-coordinate of the rotated point using the rotation matrix formula.
    rotated_y = x * s + y * c

    return np.array([rotated_x, rotated_y], dtype=np.float32)



def transform_2d_point(point, transformation_matrix):
    """
   Transform a 2D point using a given transformation matrix.

    Parameters:
    - point: 2D point to be transformed.
    - transformation_matrix: Transformation matrix.

    Returns:
    - Transformed 2D point.
    """
    
    # Augment the 2D point to a 3D point by adding a 1 at the end.
    # This allows us to use a 3x3 transformation matrix for 2D transformations.
    augmented_point = np.array([*point, 1.])

    # Multiply the transformation matrix with the augmented point to get the transformed point.
    transformed_point = np.dot(transformation_matrix, augmented_point)

    # Return only the first two values of the transformed point as the result is 2D.
    return transformed_point[:2]


def generate_target(joints, joints_vis, sz_hm=[64, 64], sigma=2, gType='gaussian'):
	"""
    Generate target heatmaps for given joints and visibility information.

    Args:
        joints (list): List of joint coordinates.
        joints_vis (list): List of joint visibility values.
        sz_hm (list): Size of the generated heatmaps. Default: [64, 64].
        sigma (float): Standard deviation for Gaussian heatmap generation. Default: 2.
        gType (str): Type of heatmap generation. Currently, only 'gaussian' is supported.

    Returns:
        tuple: Target heatmaps and target weight values.
	"""
	n_jt = len(joints) # Number of joints
	target_weight = np.ones((n_jt, 1), dtype=np.float32)  # Initialize target weight values
	target_weight[:, 0] = joints_vis # Set target weights based on joint visibility

	if gType == 'gaussian':
		target = np.zeros((n_jt,sz_hm[1], sz_hm[0]), dtype=np.float32) # Initialize target heatmaps
		tmp_size = sigma * 3
		for joint_id in range(n_jt):
			mu_x = int(joints[joint_id][0] + 0.5)
			mu_y = int(joints[joint_id][1] + 0.5)

			ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # Upper-left corner of Gaussian region
			br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # Bottom-right corner of Gaussian region

      # If Gaussian region is completely outside heatmap boundaries, set target weight to 0 and continue
			if ul[0] >= sz_hm[0] or ul[1] >= sz_hm[1] or br[0] < 0 or br[1] < 0:
				target_weight[joint_id] = 0
				continue

			# Generate Gaussian heatmap
			size = 2 * tmp_size + 1
			x = np.arange(0, size, 1, np.float32)
			y = x[:, np.newaxis]
			x0 = y0 = size // 2
			g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

			# Usable Gaussian range within heatmap
			g_x = max(0, -ul[0]), min(br[0], sz_hm[0]) - ul[0]
			g_y = max(0, -ul[1]), min(br[1], sz_hm[1]) - ul[1]
			# Image range
			img_x = max(0, ul[0]), min(br[0], sz_hm[0])
			img_y = max(0, ul[1]), min(br[1], sz_hm[1])

			v = target_weight[joint_id]
			if v > 0.5:
				target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
	return target, target_weight


def normalize_image(img, convert_to_color=True):
    """
    Normalize an image to range [0, 255] and optionally convert to 3-channel color.

    Parameters:
    - img: Input image array
    - convert_to_color: Whether to convert the output to a 3-channel color image

    Returns:
    - Normalized and optionally color-converted image
    """

    value_max = img.max()
    value_min = img.min()

    normalized_img = ((img.astype(float) - value_min) / (value_max - value_min) * 255).astype(np.uint8)

    # Convert to 3-channel color if required
    if convert_to_color and img.ndim < 3:
        normalized_img = np.stack([normalized_img] * 3, axis=2)

    return normalized_img


def horizontal_concat_resize(images, use_max_height=True, interpolation=cv2.INTER_CUBIC):
    """
    Horizontally concatenate images after resizing them to have the same height.

    Parameters:
    - images: List of images to be concatenated
    - use_max_height: If True, resize images to the height of the tallest image,
                      otherwise to the height of the shortest image.
    - interpolation: Interpolation method used for resizing

    Returns:
    - Horizontally concatenated image
    """

    # Determine the target height for resizing
    if use_max_height:
        target_height = max(img.shape[0] for img in images)
    else:
        target_height = min(img.shape[0] for img in images)

    # Resize images to have the same target height while maintaining aspect ratio
    resized_images = [cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height), interpolation=interpolation)for img in images]

    # Horizontally concatenate the resized images
    return cv2.hconcat(resized_images)


def name_to_index(joints, name_list):
  """
    Convert joint names or pairs to their corresponding indices in the name_list.

    Args:
        joints (list): List of joint names or pairs.
        name_list (list): List of joint names.

    Returns:
        tuple: Tuple of indices corresponding to the provided joint names or pairs.
  """
  res = []
  if type(joints[0]) == tuple:
    for item in joints:
      res.append(tuple([name_list.index(item[0]), name_list.index(item[1])]))
  else:
    for item in joints:
      res.append(name_list.index(item))
  return tuple(res)


def extract_predictions(heatmaps):
    '''
    Extract predictions from heatmaps.
    :param heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    :return: Predicted coordinates and their corresponding maximum values
    '''
    assert isinstance(heatmaps, np.ndarray), 'Heatmaps should be a numpy.ndarray'
    assert heatmaps.ndim == 4, 'Heatmaps should be 4-dimensional'


    batch_size, num_joints, _, width = heatmaps.shape
    heatmaps_flattened = heatmaps.reshape((batch_size, num_joints, -1))

    # Find the indices of maximum values in the flattened heatmaps
    max_indices = np.argmax(heatmaps_flattened, axis=2).reshape((batch_size, num_joints, 1))
    max_values = np.amax(heatmaps_flattened, axis=2, keepdims=True)

     # Generate coordinates using max indices, considering heatmap width
    coordinates = np.tile(max_indices, (1, 1, 2)).astype(np.float32)
    coordinates[:, :, 0] %= width
    coordinates[:, :, 1] //= width

    # Create a mask for valid predictions based on non-zero max values
    valid_mask = (max_values > 0.0).astype(np.float32)
    valid_coordinates = coordinates * np.tile(valid_mask, (1, 1, 2))

    return valid_coordinates, max_values

def compute_normalized_distances(predictions, targets, normalization_factors):
    """
    Calculate normalized distances between predictions and targets.
    :param predictions: Predicted coordinates
    :param targets: Ground truth coordinates
    :param normalization_factors: Factors to normalize coordinates
    :return: Normalized distances
    """
    predictions = predictions.astype(np.float32) # Convert predictions to float32
    targets = targets.astype(np.float32) # Convert targets to float32

    num_samples, num_coords, _ = predictions.shape
    distances = np.zeros((num_coords, num_samples))

    for sample_idx in range(num_samples):
        for coord_idx in range(num_coords):
            if targets[sample_idx, coord_idx, 0] > 1 and targets[sample_idx, coord_idx, 1] > 1:

                # Normalize predictions and targets based on normalization factors
                normalized_predictions = predictions[sample_idx, coord_idx] / normalization_factors[sample_idx]
                normalized_targets = targets[sample_idx, coord_idx] / normalization_factors[sample_idx]

                 # Calculate the Euclidean distance between normalized predictions and targets
                distances[coord_idx, sample_idx] = np.linalg.norm(normalized_predictions - normalized_targets)
            else:
                distances[coord_idx, sample_idx] = -1 # Set distance to -1 for invalid targets

    return distances

def dist_accuracy(distance_values, threshold=0.5):
    """
    Compute accuracy of the distance values below a given threshold.
    Distances with value -1 are ignored.
    :param distance_values: Array of distance values
    :param threshold: Threshold for accuracy computation
    :return: Percentage of values below the threshold or -1 if no valid values are present
    """
    valid_distances = distance_values[distance_values != -1]  # Filter out distances with value -1

    if valid_distances.size > 0:
        return (valid_distances < threshold).mean() # Compute the proportion of values below the threshold
    else:
        return -1  # Return -1 if there are no valid values

def reverse_flip(output_data, joint_pairs):
    """
    Reverse the effect of flipping on the output data, specifically by reverting the mirror effect
    and switching joint pairs back to their original order.

    :param output_data: A numpy array of shape (batch_size, num_joints, height, width) representing the output data
    :param joint_pairs: List of tuples, where each tuple contains indices of two joints that need to be switched back
    :return: A numpy array with reversed flipping effect
    """

    # Mirror the data horizontally
    output_data = output_data[:, :, :, ::-1]

    # Switch joint pairs back to their original order
    for joint_1, joint_2 in joint_pairs:
        output_data[:, [joint_1, joint_2], :, :] = output_data[:, [joint_2, joint_1], :, :]

    return output_data


def make_folder(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)



def tensor_to_cv2(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Convert an image tensor to a CV2 formatted image (uint8) by applying mean and standard deviation.
    Assumes the original image is in the range [0, 1], and it also performs RGB to BGR conversion.

    :param image_tensor: Input tensor of shape (channels, height, width)
    :param mean: Tuple containing mean values for each channel
    :param std: Tuple containing standard deviation values for each channel
    :return: An image in CV2 format (uint8 and BGR)
    """

    num_channels = len(mean)
    assert num_channels == len(std), f'mean channel {num_channels} and std channel {len(std)} do not match.'

    if not isinstance(image_tensor, np.ndarray):  # If tensor, convert to numpy array
        image_array = image_tensor[:num_channels].cpu().numpy()
    else:
        image_array = image_tensor[:num_channels].copy()

    # Denormalize the image
    image_array = (image_array * np.array(std).reshape(num_channels, 1, 1)
                   + np.array(mean).reshape(num_channels, 1, 1))

    image_array = image_array.astype(np.uint8)

    # Convert from RGB to BGR and transpose from CHW to HWC
    return image_array[::-1].transpose(1, 2, 0)


def normalized_distance(predicted_pose, target_pose, normalization_factor):
    """
    Calculate the normalized distance between the predicted pose and the target pose.

    :param predicted_pose: Predicted pose, shape: nx2(3)
    :param target_pose: Target pose, shape: nx2(3)
    :param normalization_factor: Factor used for normalization
    :return: Normalized distance, shape: (N, num_joints)
    """

    error = predicted_pose - target_pose # Calculate error vector between predicted and target pose
    error_magnitude = np.linalg.norm(error, axis=2) # Calculate magnitude of error vector
    return error_magnitude / normalization_factor[..., None] # Normalize error by the provided factor



def map_to_original_coordinates(joints, bounding_boxes, output_size=[64, 64, 64]):
    """
    Map joint coordinates from a bounding box back to their positions in the original image.

    :param joints: Joint coordinates in the bounding box. Shape: (num_samples, num_joints, 2)
    :param bounding_boxes: Bounding boxes. Shape: num_samples x 4
    :param output_size: Output size used for normalization. Default: [64, 64, 64]
    :return: Mapped joint coordinates. Same shape as input joints.
    """

    mapped_joints = np.zeros_like(joints)
    mapped_joints[..., 0] = joints[..., 0] / output_size[0] * bounding_boxes[..., 2, None] + bounding_boxes[..., 0, None]  # Map x-coordinates of joints to original image space using bounding box information
    mapped_joints[..., 1] = joints[..., 1] / output_size[1] * bounding_boxes[..., 3, None] + bounding_boxes[..., 1, None] # Map y-coordinates of joints to original image space using bounding box information

    return mapped_joints

def print_table(data, column_headers, row_headers, cell_width=10, precision=1, print_function=print):
    '''
    Print a formatted table with given data, column and row headers.

    Parameters:
    - data: The 2D list containing the table's data.
    - column_headers: List containing the column headers.
    - row_headers: List containing the row headers.
    - cell_width: The width of each cell in the table.
    - precision: Precision for floating point numbers.
    - print_function: Function to use for printing (e.g., print or logger.info).

    Returns:
    None
    '''

    num_rows = len(data)
    num_cols = len(data[0])

    if not (len(column_headers) == num_cols and len(row_headers) == num_rows):
        raise ValueError('Mismatch between data dimensions and header lengths.')

    # Format for the header row, centered in each cell with vertical separators
    header_format = "|{:^" + "{}".format(cell_width) + "}|"
    print_function(header_format.format("") + "".join(header_format.format(header) for header in column_headers))

    # Format for each data row, with headers and values separated by vertical separators
    row_format = "|{:<" + "{}".format(cell_width) + "}" + ("|{:^" + "{}.{}f".format(cell_width, precision) + "}|") * num_cols
    
    # Print the row with row header and data values
    for i in range(num_rows):
        print_function(row_format.format(row_headers[i], *data[i]))


# def print_table(data, column_headers, row_headers, cell_width=10, precision=1, print_function=print):
#     '''
#     Print a formatted table with given data, column and row headers.

#     Parameters:
#     - data: The 2D list containing the table's data.
#     - column_headers: List containing the column headers.
#     - row_headers: List containing the row headers.
#     - cell_width: The width of each cell in the table.
#     - precision: Precision for floating point numbers.
#     - print_function: Function to use for printing (e.g., print or logger.info).

#     Returns:
#     None
#     '''

#     num_rows = len(data)
#     num_cols = len(data[0])

#     if not (len(column_headers) == num_cols and len(row_headers) == num_rows):
#         raise ValueError('Mismatch between data dimensions and header lengths.')

#     # Format for the header row, centered in each cell
#     header_format = ("{:^" + "{}".format(cell_width) + "}") * (num_cols + 1)
#     print_function(header_format.format("", *column_headers))

#     # Format for each data row, with headers and values
#     row_format = "{:^" + "{}".format(cell_width) + "}" + ("{:^" + "{}.{}f".format(cell_width, precision) + "}") * num_cols
    
#     # Print the row with row header and data values
#     for i in range(num_rows):
#         print_function(row_format.format(row_headers[i], *data[i]))




