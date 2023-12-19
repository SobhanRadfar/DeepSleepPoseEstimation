from Essential import tensor_to_cv2
import matplotlib.pyplot as plt
import IPython
import cv2
import numpy as np
import os


def visualize_keypoints(image, keypoints, keypoint_links, visibility_threshold=0.4, overlay_opacity=1):
    """
    The `visualize_keypoints` function overlays keypoints and their linkages on a given image.

    Parameters:
    - image: Input image on which keypoints are visualized. shape = (256, 256, 1)
    - keypoints: Coordinates of the keypoints. shape = (14, 3)
    - keypoint_links: Linkage info between keypoints for drawing lines. shape = 6 * (2, )tuple
    - visibility_threshold: The threshold above which a keypoint is considered visible/valid.
    - overlay_opacity: The transparency level of the overlaid keypoints and linkages.

    Uses a rainbow color map to colorize the keypoints and their linkages.
    Checks the visibility of each keypoint before drawing.

    Returns:
    - Image with keypoints and their linkages overlaid.
    """
    
    # Transpose keypoints for easier indexing
    keypoints = keypoints.T

    # Generate a rainbow color map and derive colors for each keypoint/link
    color_map = plt.get_cmap('rainbow')
    color_map = plt.get_cmap('rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, len(keypoint_links) + 2)]
    colors = [(color[2] * 255, color[1] * 255, color[0] * 255) for color in colors]

    overlay = np.copy(image)

    for idx, link in enumerate(keypoint_links):  
        start_idx, end_idx = link
        start_point = tuple(keypoints[:2, start_idx].astype(np.int32))
        end_point = tuple(keypoints[:2, end_idx].astype(np.int32))

        if keypoints[2, start_idx] > visibility_threshold and keypoints[2, end_idx] > visibility_threshold:
            cv2.line(overlay, start_point, end_point, color=colors[idx], thickness=2, lineType=cv2.LINE_AA)  

        if keypoints[2, start_idx] > visibility_threshold:
            cv2.circle(overlay, start_point, radius=3, color=colors[idx], thickness=-1, lineType=cv2.LINE_AA)  

        if keypoints[2, end_idx] > visibility_threshold:
            cv2.circle(overlay, end_point, radius=3, color=colors[idx], thickness=-1, lineType=cv2.LINE_AA)  

    # Blend the original image and the overlay to produce the final output
    return cv2.addWeighted(image, 1.0 - overlay_opacity, overlay, overlay_opacity, 0)


def ipyth_imshow(img):
	_, ret = cv2.imencode('.jpg', img)
	i = IPython.display.Image(data=ret)
	IPython.display.display(i)


def save_2d_skeletons(image_patch, predicted_2d, skeleton, save_dir, epoch, identifier='tmp', suffix=''):
    """
    Annotate an image with 2D skeleton predictions and save it to a specified directory.

    :param image_patch: RGB image in the format (channels, width, height) as a numpy array
    :param predicted_2d: Skeleton predictions with shape (x, y, score) for each joint
    :param save_dir: Directory to save the annotated image
    :param identifier: Identifier for the saved image's filename
    :param suffix: Suffix for the directory name
    :return: None
    """

    save_path = os.path.join(save_dir, '2d' + suffix)
    os.makedirs(save_path, exist_ok=True)

    annotated_image = visualize_keypoints(image_patch, predicted_2d, skeleton)
    cv2.imwrite(os.path.join(save_path, str(identifier) +'_epoch_'+str(epoch)+ '.jpg'), annotated_image)



def save_visualization(params, dataset_reader, input_image, pred_heatmap, joints, phase, i, epoch):
    """
    Visualizes and saves the given input image with the predicted heatmaps and joints.

    Args:
    - params: Configuration and hyperparameters for visualization.
    - dataset_reader: Provides necessary dataset information like mean, std, and color maps.
    - input_image (torch.Tensor): The input image tensor to be visualized.
    - pred_heatmap (numpy array): Predicted heatmap for joints/keypoints.
    - joints: Ground truth joint coordinates (not used in this function but might be needed for future enhancements).
    - phase (str): Current phase, e.g., 'test' or 'validation'.
    - i (int): Current batch index.
    - epoch (int): Current epoch number.

    """

    # Determine the directory to save the visualization based on the phase
    save_dir = params.vis_test_dir if phase == 'test' else params.vis_val_dir
    # Extract the modality (e.g., 'RGB', 'Depth') from parameters
    modality = params.source_modes[0]  
    # Retrieve the mean and standard deviation for the current modality
    mean = dataset_reader.means[modality]
    std = dataset_reader.stds[modality]
    
    # Convert the input image tensor to a CV2 compatible format
    img_patch_visual = tensor_to_cv2(input_image[0], mean, std)  # Assumes a tensor_to_cv2 function exists
    
    # If the modality is not RGB, apply the respective color map to the image
    if modality != 'RGB':
        color_map = getattr(cv2, dataset_reader.dct_clrMap[modality])
        img_patch_visual = cv2.applyColorMap(img_patch_visual, color_map)

    # Determine the index of the test sample being visualized
    test_index = i * params.batch_size

    # Get the indices forming the skeleton structure of the joints for visualization
    skeleton_indices = dataset_reader.skeletons_idx

    # Convert predicted heatmap values to 2D joint coordinates on the input patch
    pred_2d_patch = np.ones((dataset_reader.joint_num_ori, 3))
    pred_2d_patch[:, :2] = pred_heatmap[0] / params.output_shape[0] * params.patch_size[1]
    
    # Save the visual representation of the 2D skeleton prediction on the image
    save_2d_skeletons(img_patch_visual, pred_2d_patch, skeleton_indices, save_dir, epoch=epoch, suffix='-' + modality, identifier=test_index)

