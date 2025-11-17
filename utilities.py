"""
Pirog's Nodes Utilities
Common utility functions for image processing and conversion.
Used by multiple nodes to avoid code duplication.
"""

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


def tensor_to_pil(tensor):
    """
    Convert ComfyUI tensor to PIL Image

    Args:
        tensor: ComfyUI image tensor in BHWC format (Batch, Height, Width, Channels)

    Returns:
        PIL Image object
    """
    # Ensure tensor is in CPU and float
    tensor = tensor.cpu().float()

    # Handle batch dimension - take first image if batched
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and then to uint8
    np_image = (tensor.numpy() * 255).astype(np.uint8)

    # Create PIL image based on channel count
    if np_image.shape[-1] == 3:
        return Image.fromarray(np_image, mode='RGB')
    elif np_image.shape[-1] == 4:
        return Image.fromarray(np_image, mode='RGBA')
    elif np_image.shape[-1] == 1:
        return Image.fromarray(np_image.squeeze(-1), mode='L')
    else:
        raise ValueError(f"Unsupported channel count: {np_image.shape[-1]}")


def pil_to_tensor(pil_image):
    """
    Convert PIL Image to ComfyUI tensor format

    Args:
        pil_image: PIL Image object

    Returns:
        ComfyUI tensor in BHWC format (Batch, Height, Width, Channels)
    """
    # Convert to numpy array and normalize to [0, 1]
    np_image = np.array(pil_image).astype(np.float32) / 255.0

    # Handle grayscale images - add channel dimension if needed
    if len(np_image.shape) == 2:
        np_image = np_image[..., np.newaxis]

    # Convert to torch tensor
    tensor = torch.from_numpy(np_image)
    return tensor.unsqueeze(0)


def srgb_to_linear(img):
    """
    Converts sRGB image data to linear RGB.

    Args:
        img: numpy array with sRGB values in [0, 1] range

    Returns:
        numpy array with linear RGB values
    """
    mask = img <= 0.04045
    return np.where(mask, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img):
    """
    Converts linear RGB image data to sRGB.

    Args:
        img: numpy array with linear RGB values

    Returns:
        numpy array with sRGB values in [0, 1] range
    """
    mask = img <= 0.0031308
    srgb = np.where(mask, img * 12.92, 1.055 * (img ** (1.0 / 2.4)) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def tensor_to_numpy(tensor):
    """
    Convert ComfyUI tensor to numpy array

    Args:
        tensor: ComfyUI image tensor in BHWC format (Batch, Height, Width, Channels)

    Returns:
        numpy array in BHWC format
    """
    return tensor.cpu().numpy()


def numpy_to_tensor(np_array):
    """
    Convert numpy array to ComfyUI tensor

    Args:
        np_array: numpy array in BHWC format (Batch, Height, Width, Channels)

    Returns:
        ComfyUI tensor in BHWC format
    """
    return torch.from_numpy(np_array)


def safe_poisson(lam):
    """
    Generate Poisson samples that handles large lambda values by using normal approximation.

    Args:
        lam: lambda parameter (mean) for Poisson distribution

    Returns:
        Poisson samples, using normal approximation for large lambda values
    """
    # Threshold for switching to normal approximation (numpy's practical limit)
    MAX_POISSON_LAM = 2**30  # Conservative threshold

    # For large lambda, use normal approximation: N(lambda, sqrt(lambda))
    large_lam_mask = lam > MAX_POISSON_LAM

    if np.any(large_lam_mask):
        result = np.zeros_like(lam, dtype=np.float64)
        # Normal approximation for large lambda
        result[large_lam_mask] = np.random.normal(lam[large_lam_mask], np.sqrt(lam[large_lam_mask]))
        # Exact Poisson for small lambda
        result[~large_lam_mask] = np.random.poisson(lam[~large_lam_mask])
        return result
    else:
        return np.random.poisson(lam)


def apply_noise_blur(noise, blur_sigma):
    """
    Applies a Gaussian blur to the generated noise field.

    Args:
        noise: numpy array representing noise
        blur_sigma: sigma value for Gaussian blur

    Returns:
        Blurred noise array
    """
    if blur_sigma > 0:
        for c in range(noise.shape[2]):
            noise[:, :, c] = gaussian_filter(noise[:, :, c], sigma=blur_sigma)
    return noise
