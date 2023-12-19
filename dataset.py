from Essential import name_to_index, generate_perspective_Transformers, load_physique_data, init_modality_lists
from Essential import get_bounding_box, read_image, uni_mod, adjust_bounding_box, generate_image_patch, get_augmentation_config
from Essential import transform_2d_point, generate_target, normalize_image, horizontal_concat_resize
from plot import visualize_keypoints
import torchvision.transforms as transforms
import torch
import scipy.io as sio
import numpy as np
import cv2
import os



cover_dict={'uncover': 0, 'cover1': 1, 'cover2': 2}
class dataset_reader:
  joints_number = 17
  joint_num_ori = 14

  # Define the names for joints, skeletons and their flip pairs
  joints_name = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow',
                  'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'Thorax', 'Head', 'Pelvis', 'Torso', 'Neck')

  skeletons_name = (('Thorax', 'Head'), ('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'), ('Thorax', 'L_Shoulder'),
                    ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'), ('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'), ('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'))

  flip_pairs_name = (('R_Hip', 'L_Hip'), ('R_Knee', 'L_Knee'), ('R_Ankle', 'L_Ankle'),
                       ('R_Shoulder', 'L_Shoulder'), ('R_Elbow', 'L_Elbow'), ('R_Wrist', 'L_Wrist'))

  # Color map dictionary
  dct_clrMap = {'depth': 'COLORMAP_BONE',
               'depthRaw': 'COLORMAP_BONE',
               'IR': 'COLORMAP_HOT',
               'IRraw': 'COLORMAP_HOT',
               'PM': 'COLORMAP_JET',
               'PMraw': 'COLORMAP_JET'}

  # Convert joint names to indices
  skeletons_idx = name_to_index(joints=skeletons_name, name_list=joints_name)
  flip_pairs = name_to_index(joints=flip_pairs_name, name_list=joints_name)

  def __init__(self, pars, phase='train', zero_base=True):
    self.pars = pars
    self.data_path = pars.data_path
    frames_number = 45

    # Settings based on data path
    lookup_table = {True: {"n_subj": 7, "d_bed": 2.264, "n_split": 0},
                    False: {"n_subj": 102, "d_bed": 2.101, "n_split": 90}}

    settings = lookup_table['simLab' in self.data_path]
    self.n_subj = settings["n_subj"]
    self.d_bed = settings["d_bed"]
    self.n_split = settings["n_split"]

    # Define sizes for each modality
    self.sizes={'RGB': [576, 1024],
                'PM': [84, 192],
                'IR': [120, 160],
                'depth': [424, 512]}

    self.sz_depth = [424, 512]
    self.sz_PM = [84, 192]
    self.sz_IR = [120, 160]
    self.sz_RGB = [576, 1024]

    # Camera parameters
    self.c_d = [208.1, 259.7]
    self.f_d = [367.8, 367.8]
    self.PM_max = 94
    self.phase = phase
    self.sz_pch = pars.patch_size
    self.fc_depth = pars.fc_depth

    # Define the mean and standard deviation for normalization
    self.means={
			'RGB': [0.3875689, 0.39156103, 0.37614644],
			'depth': [0.7302197],
			'depthRaw': [2190.869],
			'IR': [0.1924838],
			'PM': [0.009072126],
		}

    self.stds = {
			'RGB': [0.21462509, 0.22602762, 0.21271782],
			'depth': [0.25182092],
			'depthRaw': [756.1536],
			'IR': [0.077975444],
			'PM': [0.038837425],
		}

    # Define train/test phases
    phase_dict = {'train': range(self.n_split),
                  'test': range(self.n_split, self.n_subj),
                  'all': range(self.n_subj)}

    idxs_subj_all = range(self.n_subj)
    idxs_subj = idxs_subj_all if 'simLab' in self.data_path else phase_dict.get(phase, idxs_subj_all)
    self.idxs_subj = idxs_subj


    # Generate perspective transformers
    self.perspective_tarns = generate_perspective_Transformers(path=self.data_path, samples=idxs_subj_all, modalities=['RGB', 'IR', 'depth', 'PM'])

    # Load physique data
    self.phys_arr = load_physique_data(self.data_path)
    modalities = ['RGB', 'IR', 'depth', 'PM']

    # Initialize modality lists
    modality_lists = init_modality_lists(['RGB', 'IR', 'depth', 'PM'])
    for key, value in modality_lists.items():
      setattr(self, key, value)

    # Calibration data and ground truth loading
    self.li_caliPM = []
    print('==> loading Ground Truth')
    for i in idxs_subj_all:
      # Load joints data for RGB modality.
      joints_gt_RGB_t = sio.loadmat(os.path.join(self.data_path, '{:05d}'.format(i + 1), 'joints_gt_RGB.mat'))['joints_gt'].transpose([2, 1, 0]) 
      # Load joints data for IR modality.
      joints_gt_IR_t = sio.loadmat(os.path.join(self.data_path, '{:05d}'.format(i + 1), 'joints_gt_IR.mat'))['joints_gt'].transpose([2, 1, 0])

      # Adjust joint data by 1 if zero_base is true.
      if zero_base:
        joints_gt_RGB_t -= 1
        joints_gt_IR_t -= 1

      # Obtain the perspective transformation matrices for RGB and depth modalities.
      PTr_RGB = self.perspective_tarns['RGB'][i]
      PTr_depth = self.perspective_tarns['depth'][i]

      # Compute the perspective transformation matrix from RGB to depth.
      PTr_RGB2depth = np.dot(np.linalg.inv(PTr_depth), PTr_RGB)
      PTr_RGB2depth /= np.linalg.norm(PTr_RGB2depth)

       # Transform joints data from RGB space to depth space.
      joints_gt_depth_t = cv2.perspectiveTransform(joints_gt_RGB_t[:, :, :2], PTr_RGB2depth)[0]

      # Map the transformation on every joint.
      joints_gt_depth_t = np.array(list(map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB2depth)[0], joints_gt_RGB_t[:, :, :2])))
      joints_gt_depth_t = np.concatenate([joints_gt_depth_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)

      # Transform joints data from RGB space to PM space.
      joints_gt_PM_t = np.array(list(map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB)[0], joints_gt_RGB_t[:, :, :2])))
      joints_gt_PM_t = np.concatenate([joints_gt_PM_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)

      # Load calibration data for pressure modality.
      pth_cali = os.path.join(self.data_path, '{:05d}'.format(i + 1), 'PMcali.npy')

      if not 'simLab' in self.data_path:
        self.li_caliPM.append(np.load(pth_cali))

      # Append loaded joint data to their respective lists.
      self.li_joints_gt_RGB.append(joints_gt_RGB_t)
      self.li_joints_gt_IR.append(joints_gt_IR_t)
      self.li_joints_gt_depth.append(joints_gt_depth_t)
      self.li_joints_gt_PM.append(joints_gt_PM_t)

      # Calculate and append bounding boxes for all modalities.
      self.li_bb_RGB.append(np.array(list(map(get_bounding_box, joints_gt_RGB_t))))
      self.li_bb_IR.append(np.array(list(map(get_bounding_box, joints_gt_IR_t))))
      self.li_bb_depth.append(np.array(list(map(get_bounding_box, joints_gt_depth_t))))
      self.li_bb_PM.append(np.array(list(map(get_bounding_box, joints_gt_PM_t))))

      # Calculate and append square bounding boxes for all modalities.
      self.li_bb_sq_RGB.append(np.array(list(map(lambda x: get_bounding_box(x, aspect_ratio=1), joints_gt_RGB_t))))
      self.li_bb_sq_IR.append(np.array(list(map(lambda x: get_bounding_box(x, aspect_ratio=1), joints_gt_IR_t))))
      self.li_bb_sq_depth.append(np.array(list(map(lambda x: get_bounding_box(x, aspect_ratio=1), joints_gt_depth_t))))
      self.li_bb_sq_PM.append(np.array(list(map(lambda x: get_bounding_box(x, aspect_ratio=1), joints_gt_PM_t))))
    print('==> loading Ground Truth: done!')
    # Create a description list containing subject ID, coverage type, and frame ID.
    self.pthDesc_li = [[i + 1, cov, j + 1] for i in idxs_subj for cov in pars.cover_list for j in range(frames_number)]
    self.n_smpl = len(self.pthDesc_li)

  def joint_data(self, sample_idx=0, modality='depthRaw', square_bb=True):
    """
    This function fetches the joint data based on the provided sample index and modality. It can also adjust
    the joint data using real pressure scale if required and can return a square bounding box around the joints.

    Parameters:
    - sample_idx (int): The index of the sample to be fetched.
    - modality (str): The modality of the data to be fetched.
    - square_bb (bool): Whether to return a square bounding box or a regular one.

    Returns:
    - array: The data array for the specified modality.
    - joints: The ground truth joint locations.
    - bounding_box: The bounding box around the joints.
    """

    # Extract the subject ID, coverage type, and frame ID from the description list.
    subject_id, cover, frame_id = self.pthDesc_li[sample_idx]
    is_real_pressure = False

    # Determine if the modality is real pressure.
    if modality == 'PMreal':
      is_real_pressure = True
      modality = 'PMarray'

    # Read the data array from the image.
    array = read_image(address=self.data_path, sample=subject_id, modality=modality, cover=cover, frame=frame_id)

    # If the data is real pressure, adjust it using the pressure scale.
    if is_real_pressure:
      pressure_scale = self.li_caliPM[subject_id - 1][cover_dict[cover], frame_id - 1]
      array = array * pressure_scale

    # Standardize the modality names.
    if 'depth' in modality:
      modality = 'depth'
    if 'IR' in modality:
      modality = 'IR'

    # Get the ground truth joints for the specified modality and sample.
    modality = uni_mod(modality)
    joints_gt = getattr(self, f'li_joints_gt_{modality}')
    joints = joints_gt[subject_id-1][frame_id-1]

    # Fetch the bounding box. Use square bounding box if specified.
    if square_bb:
      bounding_box = getattr(self, f'li_bb_sq_{modality}')[subject_id - 1][frame_id - 1]
    else:
      bounding_box = getattr(self, f'li_bb_{modality}')[subject_id-1][frame_id - 1]

    return array, joints, bounding_box



  def get_PTr_A2B(self, idx=0, modA='IR', modB='depthRaw'):

    """
    Computes the perspective transformation matrix to convert from one modality to another.

    Parameters:
    - idx (int): The index of the sample.
    - modA (str): Source modality.
    - modB (str): Target modality.

    Returns:
    - PTr_A2B: Perspective transformation matrix from modA to modB.
    """

    # Extract the subject ID from the description list.
    id_subj, _, _ = self.pthDesc_li[idx]

    # Standardize the modality names.
    modA = uni_mod(modA)
    modB = uni_mod(modB)

    # Fetch the perspective transformation matrices for the two modalities.
    PTrA = self.perspective_tarns[modA][id_subj - 1]
    PTrB = self.perspective_tarns[modB][id_subj - 1]

    # Compute the transformation matrix from modA to modB.
    PTr_A2B = np.linalg.solve(PTrB, PTrA)
    PTr_A2B /= np.linalg.norm(PTr_A2B)

    return PTr_A2B


  def get_array_A2B(self, idx=0, modA='IR', modB='depthRaw'):
    """
    Transforms the array data from one modality to another using the computed perspective transformation matrix.

    Parameters:
    - idx (int): The index of the sample.
    - modA (str): Source modality.
    - modB (str): Target modality.

    Returns:
    - transformed_array: The data array transformed from modA to modB.
    """
    # Extract the subject ID, coverage type, and frame ID from the description list.
    id_subj, cov, id_frm = self.pthDesc_li[idx]

    # Read the source data array.
    arr = read_image(address=self.data_path, sample=id_subj, modality=modA, cover=cov, frame=id_frm)

    # Get the perspective transformation matrix from modA to modB.
    PTr_A2B = self.get_PTr_A2B(idx=idx, modA=modA, modB=modB)
    # Standardize the target modality name.
    modB = uni_mod(modB)

    # Get the size of the target modality.
    sz_B = getattr(self, f'sz_{uni_mod(modB)}')
    # Transform the source array to the target modality.
    return cv2.warpPerspective(arr, PTr_A2B, tuple(sz_B))



  # def get_phy(self, idx=0):
  #   n_subj, cov, n_frm = self.pthDesc_li[idx]
  #   phyVec = self.phys_arr[n_subj -1]
  #   return phyVec


  def joints_heatmap(self, idx):
    """
        Generates a heatmap for joints based on given data.
        
        Parameters:
        - idx: index of the sample to generate heatmap for.
        
        Returns:
        - dictionary containing processed image patch, heatmap, joint visibility and other related information.
    """

    # Extract relevant parameters
    mods = self.pars.source_modes # modalities (IR, PM, etc.)
    n_jt = self.joint_num_ori # 14
    sz_pch = self.pars.patch_size # (256, 256) size of the image patch
    output_shape = self.pars.output_shape[:2] # (64, 64) output shape
    mod0 = mods[0]
    li_img = []
    li_mean =[]
    li_std = []

    # Get the joint data and bounding box
    img, joints_ori, bb = self.joint_data(sample_idx=idx, modality=mod0, square_bb=True) #img: IR = (160, 120), joints_ori: (14,3), bb:(4, )
    joints_ori = joints_ori[:n_jt, :2]  # joints_ori: (14,2)
    img_height, img_width = img.shape[:2]  # limiting to 2D joints
    if not self.pars.use_bounding_box:
      mod_unm = uni_mod(mod0)
      sz_ori = self.sizes[mod_unm]
      bb = [0, 0, sz_ori[0], sz_ori[1]] #(4, )
      bb = adjust_bounding_box(bounding_box=bb, aspect_ratio=1) #(4, )

    li_mean, li_std, li_img = self.process_img(img, mod0, li_mean, li_std, li_img)

    # Process each image modality
    for mod in mods[1:]:
      img = self.get_array_A2B(idx=idx, modA=mod, modB=mod0) #shape based on mode
      li_mean, li_std, li_img = self.process_img(img, mod, li_mean, li_std, li_img)

    # Combine images from different modalities
    img_cb = np.concatenate(li_img, axis=-1) # For mods (IR, PM, depth): shape: (160,120,3)

    # Data augmentation for training phase
    if self.phase=='train' and self.pars.robust:
      scale, rot, do_flip, color_scale, do_occlusion = get_augmentation_config()
    else:
      scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False

    # Generate the image patch
    img_patch, trans = generate_image_patch(img_cb, bb, do_flip, scale, rot, do_occlusion, patch_size=self.pars.patch_size[::-1])
    
    # Adjustments for the image patches
    if img_patch.ndim<3:
      img_channels = 1
      img_patch = img_patch[..., None]
    else:
      img_channels = img_patch.shape[2]

    # Transform the joint coordinates based on the image patch transformations
    for i in range(img_channels):
      img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

    joints_pch = joints_ori.copy()
    if do_flip:
      joints_pch[:, 0] = img_width - joints_ori[:, 0] - 1
      # Flip joint pairs (for example: left arm with right arm)
      for pair in self.flip_pairs:
        joints_pch[pair[0], :], joints_pch[pair[1], :] = joints_pch[pair[1], :].copy(), joints_pch[pair[0], :].copy()

    # Apply 2D transformation to the joints using the computed transform 'trans'
    for i in range(len(joints_pch)):
      joints_pch[i, 0:2] = transform_2d_point(joints_pch[i, 0:2], trans)

    # Compute the scaling factor between the original image and the output heatmap size
    stride = sz_pch[0]/output_shape[1]
    stride = sz_pch[0]/output_shape[1]
    joints_hm = joints_pch/stride
    joints_vis = np.ones(n_jt)
    
    # Determine the visibility of each joint based on its position within the image patch boundaries
    for i in range(len(joints_pch)):
      joints_vis[i] *= ((joints_pch[i, 0] >= 0) & (joints_pch[i, 0] < self.pars.patch_size[0]) & (joints_pch[i, 1] >= 0) & (joints_pch[i, 1] < self.pars.patch_size[1]))

    # Generate target heatmaps for the joints
    hms, jt_wt = generate_target(joints_hm, joints_vis, sz_hm=output_shape[::-1])

    # Find the index of the joints 'Thorax' and 'Head'
    idx_t, idx_h = name_to_index(('Thorax', 'Head'), self.joints_name)

    # Calculate the distances between 'Thorax' and 'Head' in both heatmap and original space
    l_std_hm = np.linalg.norm(joints_hm[idx_h] - joints_hm[idx_t])
    l_std_ori = np.linalg.norm(joints_ori[idx_h] - joints_ori[idx_t])

    if_vis = False
    if if_vis:
      # Save the image patch with visualized joints
      print('saving feeder data out to rstT')
      tmpimg = img_patch.copy().astype(np.uint8)
      tmpkps = np.ones((n_jt, 3))
      tmpkps[:, :2] = joints_pch[:, :2]
      tmpkps[:, 2] = joints_vis
      tmpimg = visualize_keypoints(tmpimg, tmpkps, self.skeletons_idx)
      cv2.imwrite(os.path.join('rstT', str(idx) + '_pch.jpg'), tmpimg)

      # Save the heatmap image
      hmImg = hms.sum(axis=0)
      hm_nmd = normalize_image(hmImg)
      cv2.imwrite(os.path.join('rstT', str(idx) + '_hm.jpg'), hm_nmd)

      # Concatenate and save the image patch and heatmap side by side
      tmpimg = normalize_image(tmpimg)
      img_cb = horizontal_concat_resize([tmpimg, hm_nmd])
      cv2.imwrite(os.path.join('rstT', str(idx) + '_cb.jpg'), img_cb)
    
    # Convert the image and heatmap data to tensors and normalize
    trans_tch = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=li_mean, std=li_std)])
    pch_tch = trans_tch(img_patch)
    hms_tch = torch.from_numpy(hms)

    rst = {
			'pch':pch_tch,
			'hms': hms_tch,
			'joints_vis': jt_wt,
			'joints_pch': joints_pch.astype(np.float32),
			'l_std_hm':l_std_hm.astype(np.float32),
			'l_std_ori':l_std_ori.astype(np.float32),
			'joints_ori': joints_ori.astype(np.float32),
			'bb': bb.astype(np.float32)
		}

    return rst

  def process_img(self, img, mod, li_mean, li_std, li_img):
    li_mean += self.means[mod]
    li_std += self.stds[mod]
    if 'RGB' != mod:
        img = img[..., None]
    li_img.append(img)
    return li_mean, li_std, li_img

  def __getitem__(self, index):
    rst = self.joints_heatmap(index)
    return rst

  def __len__(self):
    return self.n_smpl
