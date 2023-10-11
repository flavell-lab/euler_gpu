from .preprocess import initialize
from .gncc import calculate_gncc
import numpy as np
import torch
import torch.nn.functional as F


def transform_image(images_repeated, dx_gpu, dy_gpu, angles_rad, memory_dict):
    """
    Rotate the image by a list of angles.

    Arguments:

    - images_repeated: PyTorch tensor (N x C x H x W)
        * N := number of batches
        * C := number of channel (should be 1)
        * H := height of the image
        * W := width of the image
    - dx_gpu, dy_gpu: lists of translations in the x and y directions, respectively, as PyTorch tensors
    - angles_rad: list of angles (in radians) to rotate the image, as a PyTorch tensor
    - memory_dict: dictionary of preallocated tensors

    Returns:
    - a tensor of rotated images with size (N x 1 x H x W)
    """    
    H, W = images_repeated.shape[2], images_repeated.shape[3]

    batch_size = dx_gpu.shape[0]

    # Initialize variables
    images_repeated = images_repeated[:batch_size]
    cos_vals = memory_dict['cos_vals'][:batch_size]
    sin_vals = memory_dict['sin_vals'][:batch_size]
    rotation_matrices = memory_dict['rotation_matrices'][:batch_size]
    output_tensor = memory_dict['output_tensor'][:batch_size]
    grid = memory_dict['grid'][:batch_size]

    # Rotation and translation
    cos_vals[:] = torch.cos(angles_rad)
    sin_vals[:] = torch.sin(angles_rad)

    # Directly assign values to the preallocated tensor
    rotation_matrices[:, 0, 0] = cos_vals
    rotation_matrices[:, 0, 1] = -sin_vals * H / W
    rotation_matrices[:, 0, 2] = dy_gpu
    rotation_matrices[:, 1, 0] = sin_vals * W / H
    rotation_matrices[:, 1, 1] = cos_vals
    rotation_matrices[:, 1, 2] = dx_gpu

    # Grid sample expects input in (N x C x H x W) format

    grid[:] = F.affine_grid(rotation_matrices, images_repeated.size(), align_corners=False)
    output_tensor[:] = F.grid_sample(images_repeated, grid, align_corners=False)

    return output_tensor


def transform_image_3d(resized_moving_image_xyz,
                      memory_dict,
                      best_transformation,
                      device):

    z_dimension = resized_moving_image_xyz.shape[2]
    moving_image_xyz_tensor = torch.tensor(
                resized_moving_image_xyz.astype(np.float32).transpose(2, 0, 1),
                device=device,
                dtype=torch.float32).unsqueeze(1).repeat(1, 1, 1, 1)

    transformed_moving_image_xyz = transform_image(
                moving_image_xyz_tensor,
                best_transformation[0].repeat(z_dimension),
                best_transformation[1].repeat(z_dimension),
                best_transformation[2].repeat(z_dimension),
                memory_dict)
    return np.transpose(np.squeeze(
                 transformed_moving_image_xyz.cpu().numpy(),
                 axis=1),
                 (1, 2, 0))


def transform_image_3d_v0(resized_fixed_image_xyz,
                       resized_moving_image_xyz,
                       best_transformation,
                       device):

    z_dim = resized_fixed_image_xyz.shape[2]
    moving_image_xyz_tensor = torch.tensor(
                resized_moving_image_xyz.astype(np.float32).transpose(2, 0, 1),
                device=device,
                dtype=torch.float32)
    moving_image_xyz_tensor = moving_image_xyz_tensor.unsqueeze(1).repeat(1, 1, 1, 1)

    memory_dict_3d = initialize(
                resized_fixed_image_xyz.astype(np.float32)[:, :, 0],
                resized_moving_image_xyz.astype(np.float32)[:, :, 0],
                torch.zeros(z_dim, device=device),
                torch.zeros(z_dim, device=device),
                torch.zeros(z_dim, device=device),
                z_dim,
                device)

    transformed_moving_image_xyz = transform_image(
                moving_image_xyz_tensor,
                best_transformation[0].repeat(z_dim),
                best_transformation[1].repeat(z_dim),
                best_transformation[2].repeat(z_dim),
                memory_dict_3d)

    return np.transpose(np.squeeze(
                 transformed_moving_image_xyz.cpu().numpy(),
                 axis=1),
                 (1, 2, 0))

def translate_along_z(shift_range,
                      resized_fixed_image_xyz,
                      transformed_moving_image_xyz,
                      moving_image_median):

    final_moving_image_xyz = np.full(
            transformed_moving_image_xyz.shape,
            moving_image_median)

    dz, gncc = search_for_z(shift_range,
                      resized_fixed_image_xyz,
                      transformed_moving_image_xyz,
                      moving_image_median)
    if dz < 0:
        final_moving_image_xyz[:, :, :dz] = \
            transformed_moving_image_xyz[:, :, -dz:]
    elif dz > 0:
        final_moving_image_xyz[:, :, dz:] = \
            transformed_moving_image_xyz[:, :, :-dz]
    elif dz == 0:
        final_moving_image_xyz = transformed_moving_image_xyz

    return dz, gncc, final_moving_image_xyz


def search_for_z(shift_range, resized_fixed_image_xyz,
        transformed_moving_image_xyz, moving_image_median):

    new_moving_image_xyz = np.full(
            transformed_moving_image_xyz.shape,
            moving_image_median)

    gncc = calculate_gncc(resized_fixed_image_xyz,
            transformed_moving_image_xyz)
    dz = 0
    for shift in shift_range:
        if shift < 0:
            new_moving_image_xyz[:, :, :shift] = \
                transformed_moving_image_xyz[:, :, -shift:]
        elif shift > 0:
            new_moving_image_xyz[:, :, shift:] = \
                transformed_moving_image_xyz[:, :, :-shift]

        new_gncc = calculate_gncc(resized_fixed_image_xyz, new_moving_image_xyz)
        if new_gncc > gncc:
            gncc = new_gncc
            dz = shift

    return dz, gncc
