import torch
import torch.nn.functional as F

def transform_image(images_repeated, dx_gpu, dy_gpu, angles_rad, memory_dict):
    """
    Rotate the image by a list of angles. 

    Arguments:

    - images_repeated: PyTorch tensor (N x H x W)
    - dx_gpu, dy_gpu: lists of translations in the x and y directions, respectively, as PyTorch tensors
    - angles_rad: list of angles (in radians) to rotate the image, as a PyTorch tensor
    - memory_dict: dictionary of preallocated tensors

    Returns:
    - a tensor of rotated images with size (N x H x W)
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

    grid[:] = F.affine_grid(rotation_matrices, images_repeated.size(), align_corners=True)
    output_tensor[:] = F.grid_sample(images_repeated, grid, align_corners=True)
    
    return output_tensor
