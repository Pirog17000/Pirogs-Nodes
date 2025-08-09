"""
Pirog's Nodes Implementation
A custom node pack for ComfyUI that provides enhanced sampling functionality.
Author: pirog
"""

import os
import sys
import torch
import json
import random
import re
import logging
import importlib.util
import numpy as np
from PIL import Image, ImageChops, ImageOps

# Try to import scipy for HQ blur, fallback to torch-based implementation
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Add the ComfyUI root directory to Python path to access the main nodes
comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)

# Import the common_ksampler function and MAX_RESOLUTION directly from the main nodes.py
# This ensures we always use the latest version from ComfyUI
try:
    from nodes import common_ksampler, MAX_RESOLUTION
except ImportError:
    # Fallback: try to import from nodes module in current directory
    spec = importlib.util.spec_from_file_location("nodes", os.path.join(comfy_dir, "nodes.py"))
    nodes_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes_module)
    common_ksampler = nodes_module.common_ksampler
    MAX_RESOLUTION = nodes_module.MAX_RESOLUTION

# Import required ComfyUI modules
import comfy.samplers
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict


class KSamplerMultiSeed:
    """
    KSampler Multi-Seed Node
    
    This node is extracted from the main ComfyUI nodes.py and provides
    multi-seed sampling functionality. It automatically uses the latest
    common_ksampler function from ComfyUI, ensuring compatibility with updates.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "positive": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "The amount of denoising applied."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Classifier-Free Guidance scale."}),
                "seed_count": ("INT", {"default": 1, "min": 1, "max": 1000,
                                       "tooltip": "The number of seeds to generate images with."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The starting random seed. It will be incremented for each image in the batch."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"tooltip": "The scheduler controls how noise is gradually removed."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("A batch of denoised latents, one for each seed.",)
    FUNCTION = "sample"

    CATEGORY = "pirog/sampling"
    DESCRIPTION = "Generates multiple images by incrementing the seed for each generation for each latent in the input batch."

    def sample(self, model, positive, negative, latent_image, denoise, steps, cfg, seed_count, seed, sampler_name,
               scheduler):
        output_latents = []
        input_latents = latent_image["samples"]

        total_generations = input_latents.shape[0] * seed_count
        pbar = comfy.utils.ProgressBar(total_generations)

        for latent_sample in input_latents:
            for i in range(seed_count):
                current_seed = seed + i

                # Reshape the single sample to a batch of 1 for the sampler
                latent_for_sampler = {"samples": latent_sample.unsqueeze(0)}

                # Call the common ksampler function from the main ComfyUI nodes
                # This ensures we always use the latest version
                result_latent, = common_ksampler(model, current_seed, steps, cfg, sampler_name, scheduler, positive,
                                                 negative, latent_for_sampler, denoise=denoise)

                output_latents.append(result_latent["samples"])
                pbar.update(1)

        # Combine all the generated latents into a single batch tensor
        final_samples = torch.cat(output_latents, dim=0)

        # Create the final output dictionary
        final_latent = {"samples": final_samples}

        return (final_latent,)


class KSamplerMultiSeedPlus:
    """
    KSamplerMultiSeed+ Node - Integrated Sampling Pipeline
    
    This node combines the KSamplerMultiSeed functionality with integrated VAE encode/decode
    and empty latent image generation. It provides a complete pipeline in one node:
    - For denoise=1.0: Creates empty latent → samples → decodes to image
    - For denoise<1.0: Encodes input image → samples → decodes to image
    
    Uses minimal approach by calling vanilla ComfyUI functions directly.
    """
    
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for encoding/decoding."}),
                "positive": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "The amount of denoising applied. 1.0=new image, <1.0=img2img."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Classifier-Free Guidance scale."}),
                "seed_count": ("INT", {"default": 1, "min": 1, "max": 1000,
                                       "tooltip": "The number of seeds to generate images with."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The starting random seed. It will be incremented for each image in the batch."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"tooltip": "The scheduler controls how noise is gradually removed."}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, 
                                  "tooltip": "The width of the generated image in pixels (used when denoise=1.0)."}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, 
                                   "tooltip": "The height of the generated image in pixels (used when denoise=1.0)."}),
            },
            "optional": {
                "input_image": ("IMAGE", {"tooltip": "Input image for img2img (used when denoise<1.0)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Generated images, one for each seed.",)
    FUNCTION = "sample_integrated"

    CATEGORY = "pirog/sampling"
    DESCRIPTION = "Integrated sampling pipeline: Creates or encodes latent → multi-seed sampling → decodes to images. For denoise=1.0 uses width/height to create empty latent. For denoise<1.0 uses input_image."

    def sample_integrated(self, model, vae, positive, negative, denoise, steps, cfg, seed_count, seed, 
                         sampler_name, scheduler, width, height, input_image=None):
        
        # Determine if we're doing txt2img (denoise=1.0) or img2img (denoise<1.0)
        is_txt2img = abs(denoise - 1.0) < 0.001
        
        if is_txt2img:
            # Create empty latent image (txt2img pipeline)
            if input_image is not None:
                print("Warning: input_image provided but denoise=1.0, ignoring input_image and using width/height")
            latent = torch.zeros([1, 4, height // 8, width // 8], device=self.device)
            latent_image = {"samples": latent}
        else:
            # Encode input image (img2img pipeline)
            if input_image is None:
                raise ValueError("input_image is required when denoise < 1.0 (img2img mode)")
            # Use vanilla VAE encode approach
            encoded_tensor = vae.encode(input_image[:,:,:,:3])
            latent_image = {"samples": encoded_tensor}

        # Multi-seed sampling using the same minimal approach as KSamplerMultiSeed
        output_latents = []
        input_latents = latent_image["samples"]

        total_generations = input_latents.shape[0] * seed_count
        pbar = comfy.utils.ProgressBar(total_generations)

        for latent_sample in input_latents:
            for i in range(seed_count):
                current_seed = seed + i

                # Reshape the single sample to a batch of 1 for the sampler
                latent_for_sampler = {"samples": latent_sample.unsqueeze(0)}

                # Call the common ksampler function from the main ComfyUI nodes
                # This ensures we always use the latest version
                result_latent, = common_ksampler(model, current_seed, steps, cfg, sampler_name, scheduler, positive,
                                                 negative, latent_for_sampler, denoise=denoise)

                output_latents.append(result_latent["samples"])
                pbar.update(1)

        # Combine all the generated latents into a single batch tensor
        final_samples = torch.cat(output_latents, dim=0)

        # Decode latents to images using vanilla VAE decode approach
        images = vae.decode(final_samples)
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        return (images,)


class StringCombine:
    """
    Text Concatenation Node with Fancy Boolean Toggle
    
    This node provides two multiline text input fields and a boolean toggle
    to control whether to concatenate both strings or output only the first one.
    Features a sleek toggle button following KJ/Essentials styling patterns.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text1": ("STRING", {
                    "multiline": True, 
                    "default": "First text...", 
                    "tooltip": "Primary text input. Always included in output."
                }),
                "text2": ("STRING", {
                    "multiline": True, 
                    "default": "Second text...", 
                    "tooltip": "Secondary text input. Concatenated when toggle is enabled."
                }),
                "concat_enabled": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Enable to concatenate both texts. Disable to output only first text."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("The concatenated or single text output based on toggle state.",)
    FUNCTION = "concatenate_text"

    CATEGORY = "pirog/text"
    DESCRIPTION = "Concatenates two multiline text inputs based on a boolean toggle. When enabled, outputs text1 + text2. When disabled, outputs only text1."

    def concatenate_text(self, text1, text2, concat_enabled):
        if concat_enabled:
            # Concatenate both texts
            result = text1 + text2
        else:
            # Output only the first text
            result = text1
        
        return (result,)


class Watermark:
    """
    Advanced Watermark Node
    
    Applies watermarks to images with multiple blend modes, positioning options,
    and advanced masking capabilities. Supports batch processing with proper
    error handling and logging.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Base image(s) to apply watermark to"}),
                "watermark": ("IMAGE", {"tooltip": "Watermark image to overlay"}),
                "scale": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.001, 
                    "max": 1.0, 
                    "step": 0.001,
                    "tooltip": "Scale factor for watermark size (0.001-1.0)"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.001, 
                    "max": 1.0, 
                    "step": 0.001,
                    "tooltip": "Watermark opacity (0.001-1.0)"
                }),
                "blend_mode": ([
                    "normal", "overlay", "soft_light", "hard_light", "difference", 
                    "exclusion", "darken", "lighten", "color_burn", "color_dodge", "add"
                ], {"tooltip": "Blending mode for watermark application"}),
                "position": ([
                    "random", "center", "top-left", "top-center", "top-right",
                    "middle-left", "middle-right", "bottom-left", "bottom-center", "bottom-right"
                ], {"tooltip": "Watermark positioning on the image"}),
                "invert_watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert watermark colors before applying"
                }),
                "use_black_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use black pixels as transparent mask"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Image(s) with watermark applied",)
    FUNCTION = "apply_watermark"
    CATEGORY = "pirog/image"
    DESCRIPTION = "Applies watermarks to images with advanced blending modes, positioning, and masking options."

    def apply_watermark(self, image, watermark, scale, opacity, blend_mode, position, invert_watermark, use_black_mask):
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Input image tensor shape: {image.shape}")
            logger.info(f"Input watermark tensor shape: {watermark.shape}")

            # Ensure tensors are in (B, C, H, W) format
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.shape[1] != 3 and image.shape[1] != 4:
                image = image.permute(0, 3, 1, 2)
            
            if watermark.dim() == 3:
                watermark = watermark.unsqueeze(0)
            if watermark.shape[1] != 3 and watermark.shape[1] != 4:
                watermark = watermark.permute(0, 3, 1, 2)

            batch_size, channels, height, width = image.shape
            processed_images = []

            for i in range(batch_size):
                img_pil = self.tensor_to_pil(image[i])
                watermark_pil = self.tensor_to_pil(watermark[0] if watermark.shape[0] > 1 else watermark)
                
                result = self.process_single_image(img_pil, watermark_pil, scale, opacity, blend_mode, position, invert_watermark, use_black_mask, logger)
                processed_images.append(self.pil_to_tensor(result))

            # Stack processed images and return
            result = torch.stack(processed_images)
            
            # Ensure the output is in the format expected by ComfyUI (B, H, W, C)
            result = result.permute(0, 2, 3, 1)
            
            return (result,)
        except Exception as e:
            logger.error(f"Error in watermark application: {str(e)}")
            return (image,)  # Return original image in case of error

    def tensor_to_pil(self, tensor):
        tensor = tensor.cpu().float()
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3 dimensions (C,H,W), got tensor with shape {tensor.shape}")
        tensor = tensor.clamp(0, 1)
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        if array.shape[2] == 1:
            return Image.fromarray(array.squeeze(), mode='L')
        elif array.shape[2] == 3:
            return Image.fromarray(array, mode='RGB')
        elif array.shape[2] == 4:
            return Image.fromarray(array, mode='RGBA')
        else:
            raise ValueError(f"Unexpected number of channels: {array.shape[2]}")

    def pil_to_tensor(self, pil_image):
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        if len(np_image.shape) == 2:
            np_image = np_image[..., np.newaxis]
        tensor = torch.from_numpy(np_image).permute(2, 0, 1)
        return tensor

    def process_single_image(self, image, watermark, scale, opacity, blend_mode, position, invert_watermark, use_black_mask, logger):
        watermark, mask = self.prepare_watermark(watermark, scale, invert_watermark, use_black_mask, logger)
        pos = self.get_watermark_position(position, image.size, watermark.size)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        watermark_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        watermark_layer.paste(watermark, pos, watermark)
        
        blended = self.apply_blend_mode(image, watermark_layer, blend_mode, logger)
        
        # Combine the black pixel mask with the alpha channel mask
        if use_black_mask and mask:
            mask_layer = Image.new('L', image.size, 0)
            mask_layer.paste(mask, pos)
            combined_mask = ImageChops.multiply(watermark_layer.split()[3], mask_layer)
        else:
            combined_mask = watermark_layer.split()[3]
        
        # Convert opacity to integer value
        opacity_int = self.float_to_int(opacity)
        
        # Use the combined mask to blend between the original image and the blended result
        result = Image.composite(blended, image, ImageChops.multiply(combined_mask, Image.new('L', combined_mask.size, opacity_int)))
        
        logger.info(f"Watermark applied at position {pos} with opacity {opacity} (int: {opacity_int}), "
                    f"blend mode {blend_mode}, and use_black_mask={use_black_mask}")
        return result

    def apply_blend_mode(self, base, blend, mode, logger):
        if mode == "normal":
            return blend
        elif mode == "overlay":
            return ImageChops.overlay(base, blend)
        elif mode == "soft_light":
            return ImageChops.soft_light(base, blend)
        elif mode == "hard_light":
            return ImageChops.hard_light(base, blend)
        elif mode == "difference":
            return ImageChops.difference(base, blend)
        elif mode == "exclusion":
            return ImageChops.invert(ImageChops.difference(ImageChops.invert(base), blend))
        elif mode == "darken":
            return ImageChops.darker(base, blend)
        elif mode == "lighten":
            return ImageChops.lighter(base, blend)
        elif mode == "color_burn":
            return ImageChops.subtract(base, ImageChops.invert(blend))
        elif mode == "color_dodge":
            return ImageChops.invert(ImageChops.subtract(ImageChops.invert(base), blend))
        elif mode == "add":
            return ImageChops.add(base, blend, scale=1.0, offset=0)
        else:
            logger.warning(f"Unknown blend mode: {mode}. Falling back to normal blend.")
            return blend

    def prepare_watermark(self, watermark, scale, invert, use_black_mask, logger):
        # Resize the watermark
        w, h = watermark.size
        new_w, new_h = int(w * scale), int(h * scale)
        watermark = watermark.resize((new_w, new_h), Image.LANCZOS)
        
        # Invert the watermark if requested
        if invert:
            watermark = ImageOps.invert(watermark)
        
        # Ensure the watermark is in RGBA mode
        if watermark.mode != 'RGBA':
            watermark = watermark.convert('RGBA')
        
        # Create mask from black pixels after possible inversion
        if use_black_mask:
            # Convert to grayscale
            gray_watermark = watermark.convert('L')
            # Create mask: white for non-black pixels, black for black pixels
            # Using a small threshold (5) to account for near-black pixels
            mask = gray_watermark.point(lambda x: 255 if x > 5 else 0, mode='1')
            
            # Log the number of pixels in the mask for debugging
            black_pixel_count = sum(1 for pixel in mask.getdata() if pixel == 0)
            logger.info(f"Black pixel mask created with {black_pixel_count} black pixels")
        else:
            mask = None
        
        return watermark, mask

    def get_watermark_position(self, position, image_size, watermark_size):
        if position == "center":
            return ((image_size[0] - watermark_size[0]) // 2,
                    (image_size[1] - watermark_size[1]) // 2)
        elif position == "top-left":
            return (0, 0)
        elif position == "top-center":
            return ((image_size[0] - watermark_size[0]) // 2, 0)
        elif position == "top-right":
            return (image_size[0] - watermark_size[0], 0)
        elif position == "middle-left":
            return (0, (image_size[1] - watermark_size[1]) // 2)
        elif position == "middle-right":
            return (image_size[0] - watermark_size[0], (image_size[1] - watermark_size[1]) // 2)
        elif position == "bottom-left":
            return (0, image_size[1] - watermark_size[1])
        elif position == "bottom-center":
            return ((image_size[0] - watermark_size[0]) // 2, image_size[1] - watermark_size[1])
        elif position == "bottom-right":
            return (image_size[0] - watermark_size[0],
                    image_size[1] - watermark_size[1])
        else:  # random
            return (np.random.randint(0, max(1, image_size[0] - watermark_size[0])),
                    np.random.randint(0, max(1, image_size[1] - watermark_size[1])))

    def float_to_int(self, value):
        """Convert float (0-1) to int (0-255)"""
        return int(value * 255)


class ImageScalePro:
    """
    Professional Image Scaling Tool
    
    Advanced proportional image scaling with resolution limits, step alignment,
    and professional-grade controls. Features MX-style slider compatibility
    for enhanced user experience.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image(s) to scale"}),
                "scale_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Scale factor - 1.0 = original size, 2.0 = double size, 0.5 = half size"
                }),
                "resolution_step": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Resolution alignment step (dimensions will be rounded to multiples of this value)"
                }),
                "enable_limits": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable resolution limits to prevent extremely large or small images"
                }),
                "min_resolution": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Minimum resolution for the longest side when limits are enabled"
                }),
                "max_resolution": ("INT", {
                    "default": 2560,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Maximum resolution for the longest side when limits are enabled"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Scaled image(s) with proper aspect ratio preservation",)
    FUNCTION = "scale_image"
    CATEGORY = "pirog/transform"
    DESCRIPTION = "Professional image scaling with proportional resize, resolution limits, and step alignment for optimal results."

    def scale_image(self, image, scale_multiplier, resolution_step, enable_limits, min_resolution, max_resolution):
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")

            batch_size, height, width, channels = image.shape
            scaled_images = []

            for i in range(batch_size):
                pil_image = self.tensor_to_pil(image[i])
                logger.info(f"Original PIL Image size: {pil_image.size}")

                new_width = int(width * scale_multiplier)
                new_height = int(height * scale_multiplier)

                if enable_limits:
                    new_width, new_height = self.apply_resolution_limits(new_width, new_height, min_resolution, max_resolution)

                new_width = max((new_width // resolution_step) * resolution_step, resolution_step)
                new_height = max((new_height // resolution_step) * resolution_step, resolution_step)

                logger.info(f"Calculated new dimensions: {new_width}x{new_height}")

                resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                logger.info(f"Resized PIL Image size: {resized_image.size}")

                scaled_images.append(self.tensor_to_pil_tensor(resized_image))

            output_tensor = torch.cat(scaled_images, dim=0)
            logger.info(f"Output tensor shape: {output_tensor.shape}")

            return (output_tensor,)

        except Exception as e:
            logger.error(f"Error in scale_image: {str(e)}")
            logger.error(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")
            return (image,)

    def tensor_to_pil(self, tensor):
        tensor = tensor.cpu().float()
        tensor = torch.clamp(tensor, 0, 1)
        array = (tensor * 255).byte().numpy()
        
        if array.shape[-1] == 3:
            return Image.fromarray(array, mode='RGB')
        elif array.shape[-1] == 4:
            return Image.fromarray(array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {array.shape[-1]}")

    def tensor_to_pil_tensor(self, pil_image):
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def apply_resolution_limits(self, width, height, min_resolution, max_resolution):
        longest_side = max(width, height)
        scale_factor = 1.0

        if longest_side < min_resolution:
            scale_factor = min_resolution / longest_side
        elif longest_side > max_resolution:
            scale_factor = max_resolution / longest_side

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        return new_width, new_height


class PromptRandomizer:
    """
    Advanced Prompt Randomization Tool
    
    Randomizes text prompts using pattern replacement with dictionary support.
    Features relative path handling, validation, fallback dictionaries, and
    configurable text processing options.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "?color? ?dog|cat? with ?orange|black? dots on the ?couch|floor?",
                    "tooltip": "Use pattern ?word1|word2|metaword? to randomize. Encapsulate groups with question marks. Meta-words are looked up in the dictionary."
                }),
                "dictionary_filename": ("STRING", {
                    "default": "prompt_dictionary.json",
                    "tooltip": "Dictionary filename (relative to this node pack). Will create a default if not found."
                }),
                "seed": ("INT", {
                    "default": -1, 
                    "min": -1, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Set to -1 for random seed each run, or any positive number for reproducible results."
                }),
                "preserve_newlines": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep newlines in output. When disabled, newlines are converted to commas."
                }),
                "clean_text": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply text cleaning (remove extra spaces, fix punctuation, convert anime terms)."
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("modified_prompt", "selections", "used_seed")
    FUNCTION = "randomize_prompt"
    CATEGORY = "pirog/text"
    DESCRIPTION = "Randomizes text prompts using pattern replacement with dictionary support, validation, and flexible text processing options."

    def randomize_prompt(self, prompt, dictionary_filename, seed, preserve_newlines, clean_text):
        all_selections = []
        
        # Set the seed
        used_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
        random.seed(used_seed)

        # Get dictionary path relative to this node pack
        dictionary_path = self.get_dictionary_path(dictionary_filename)

        # Load or create the dictionary
        random_dictionary = self.load_or_create_dictionary(dictionary_path)

        # Process randomization
        for i in range(10):  # Max 10 iterations to prevent infinite loops
            # Find all tags in the prompt string
            tags = re.findall(r"\?\s*[^\?]+\s*\?", prompt)
            
            if not tags:  # No more tags to process
                break

            for tag in tags:
                # Get the options
                options = tag[1:-1].split("|")
                options = [option.strip() for option in options]

                # Choose a random option
                random_option = random.choice(options)

                # If the option is in the dictionary, use a random word from that category
                # Otherwise, use the option as-is
                if random_option in random_dictionary and random_dictionary[random_option]:
                    word = random.choice(random_dictionary[random_option])
                else:
                    word = random_option

                # Replace the tag with the chosen word
                prompt = prompt.replace(tag, word, 1)

                # Add only non-empty selections to array for later display
                if word and word.strip():
                    all_selections.append(word.strip())

        # Join selections into a single string
        selections_string = ", ".join(all_selections)
        
        # Apply text processing based on settings
        if clean_text:
            prompt = self.clean_prompt(prompt)
        
        if not preserve_newlines:
            prompt = self.remove_newlines(prompt)

        return (prompt, selections_string, used_seed)

    def get_dictionary_path(self, filename):
        # Get the directory where this node pack is located
        script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Ensure filename is just a filename, not a path
        filename = os.path.basename(filename)
        
        # Create full path
        dictionary_path = os.path.join(script_directory, filename)
        
        return dictionary_path

    def load_or_create_dictionary(self, dictionary_path):
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as dict_file:
                random_dictionary = json.load(dict_file)
                return random_dictionary
        except FileNotFoundError:
            # Create a default dictionary
            default_dict = self.create_default_dictionary()
            try:
                with open(dictionary_path, 'w', encoding='utf-8') as dict_file:
                    json.dump(default_dict, dict_file, indent=2, ensure_ascii=False)
                print(f"Created default dictionary at: {dictionary_path}")
            except Exception as e:
                print(f"Could not create dictionary file: {e}")
            return default_dict
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in dictionary file: {dictionary_path}. Using default dictionary. Error: {e}")
            return self.create_default_dictionary()
        except Exception as e:
            print(f"Error reading dictionary file: {dictionary_path}. Using default dictionary. Error: {e}")
            return self.create_default_dictionary()

    def create_default_dictionary(self):
        """Create a default dictionary with common prompt categories"""
        return {
            "color": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "gray"],
            "animal": ["cat", "dog", "bird", "rabbit", "fox", "wolf", "lion", "tiger", "bear", "deer"],
            "style": ["photorealistic", "artistic", "anime", "cartoon", "realistic", "stylized", "abstract"],
            "lighting": ["soft lighting", "dramatic lighting", "natural lighting", "studio lighting", "golden hour"],
            "mood": ["happy", "serene", "mysterious", "energetic", "calm", "dramatic", "peaceful", "intense"],
            "quality": ["high quality", "masterpiece", "detailed", "sharp", "crisp", "professional", "stunning"],
            "background": ["forest", "city", "beach", "mountains", "garden", "studio", "indoor", "outdoor"],
            "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "stormy", "clear", "overcast"]
        }

    def clean_prompt(self, string_to_fix):
        """Clean and normalize the prompt text"""
        fixed = string_to_fix
        
        # Multiple iterations to catch nested issues
        for i in range(5):
            # Normalize whitespace
            fixed = fixed.replace('\n', ', ')
            fixed = re.sub(r'\s+', ' ', fixed)  # Replace multiple spaces with single space
            
            # Fix punctuation
            fixed = fixed.replace(' , ,', ',')
            fixed = fixed.replace(' ,', ',')
            fixed = fixed.replace(' ,;', ';')
            fixed = fixed.replace(';;', ';')
            fixed = fixed.replace('; ;', ';')
            fixed = fixed.replace(';,', '; ')
            fixed = fixed.replace('; ,', ';')
            fixed = fixed.replace(', ;', ';')
            fixed = fixed.replace(' ,, ', ', ')
            fixed = fixed.replace(', ,', ', ')
            fixed = fixed.replace(',,', ',')
            fixed = fixed.replace(', , , ', ', ')
            fixed = fixed.replace(', , ,', ', ')
            
            # Remove empty brackets
            fixed = fixed.replace('[]', '')
            fixed = fixed.replace('()', '')
            fixed = fixed.replace('(()', '')
            
            # Anime/manga term normalization (optional - can be disabled)
            fixed = fixed.replace('1girl', 'one woman')
            fixed = fixed.replace('2girls', 'two women')
            fixed = fixed.replace('3girls', 'three women')
            fixed = fixed.replace('1boy', 'one man')
            fixed = fixed.replace('2boys', 'two men')
            fixed = fixed.replace('3boys', 'three men')
            
        return fixed.strip()

    def remove_newlines(self, text):
        """Convert newlines to commas for single-line output"""
        # Replace newlines with commas, but avoid double commas
        text = text.replace('\n', ', ')
        text = re.sub(r',\s*,', ',', text)  # Remove double commas
        text = re.sub(r',\s*$', '', text)   # Remove trailing comma
        return text.strip()


class DSLRNoise(ComfyNodeABC):
    """
    DSLR Camera Noise Simulation Node
    
    Scientifically accurate camera sensor noise simulation based on real DSLR noise research.
    Implements the Poisson-Gaussian noise model with shot noise, read noise, thermal noise,
    and PRNU (Pixel Response Non-Uniformity) as described in camera sensor literature.
    
    References:
    - "Noise, Dynamic Range and Bit Depth in Digital SLRs" (Emil Martinec)
    - "Digital Camera Image Noise" (Cambridge in Colour)
    - ISO 15739 noise measurement standards
    """
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "image": (IO.IMAGE, {
                    "tooltip": "📸 Input image to apply DSLR camera noise simulation. Works with any resolution and batch size."
                }),
                
                "iso": (IO.INT, {
                    "default": 800, "min": 100, "max": 25600, "step": 100,
                    "tooltip": "📊 ISO Sensitivity (100-25600): Camera's light sensitivity setting.\n\n"
                              "• 100-400: Clean, minimal noise (bright daylight)\n"
                              "• 800-1600: Moderate noise (indoor/evening)\n"
                              "• 3200-6400: High noise (low light conditions)\n"
                              "• 12800+: Very noisy (extreme low light)\n\n"
                              "📋 Default: 800 (realistic indoor photography)"
                }),
                
                "camera_model": (["modern_fullframe", "modern_apsc", "older_sensor", "high_end"], {
                    "default": "modern_fullframe",
                    "tooltip": "📷 Camera Sensor Type: Different sensor technologies have unique noise characteristics.\n\n"
                              "• Modern Full-Frame: Clean, low noise (e.g., Canon R5, Sony A7R)\n"
                              "• Modern APS-C: Slightly more noise, good performance (e.g., Fuji X-T5)\n"
                              "• Older Sensor: Higher noise, pattern issues (e.g., Canon 20D era)\n"
                              "• High-End: Extremely clean, professional (e.g., Phase One, Canon R3)\n\n"
                              "📋 Default: modern_fullframe"
                }),
                
                "temperature": (IO.FLOAT, {
                    "default": 20.0, "min": -10.0, "max": 60.0, "step": 1.0,
                    "tooltip": "🌡️ Sensor Temperature (°C): Heat significantly affects digital noise.\n\n"
                              "• -10°C to 10°C: Minimal thermal noise (cold weather)\n"
                              "• 20°C: Room temperature baseline\n"
                              "• 30°C+: Increased thermal noise (hot environments)\n"
                              "• 50°C+: Significant noise increase (overheated sensor)\n\n"
                              "📋 Default: 20°C (room temperature)"
                }),
                
                "exposure_time": (IO.FLOAT, {
                    "default": 1.0, "min": 0.001, "max": 30.0, "step": 0.1,
                    "tooltip": "⏱️ Exposure Time (seconds): Longer exposures accumulate more thermal noise.\n\n"
                              "• 0.001-0.1s: Fast shutter, minimal thermal impact\n"
                              "• 1-5s: Moderate exposure, some thermal buildup\n"
                              "• 10-30s: Long exposure, significant thermal noise\n\n"
                              "📋 Default: 1.0s (standard exposure)"
                }),
                
                "strength": (IO.FLOAT, {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "💪 Overall Noise Strength: Master intensity control for all noise types.\n\n"
                              "• 0.0: No noise applied\n"
                              "• 0.5: Subtle, realistic noise\n"
                              "• 1.0: Standard DSLR noise levels\n"
                              "• 1.5-2.0: Enhanced noise for artistic effect\n"
                              "• 2.5-3.0: Heavy noise for dramatic/vintage looks\n\n"
                              "📋 Default: 1.0 (realistic levels)"
                }),
                
                "seed": (IO.INT, {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff, 
                    "control_after_generate": True,
                    "tooltip": "🎲 Random Seed: Controls noise pattern reproducibility.\n\n"
                              "• Same seed = identical noise pattern\n"
                              "• Different seed = different noise distribution\n"
                              "• Auto-increment after generation for variation\n\n"
                              "📋 Default: 0 (click 🎲 to randomize)"
                }),
            },
            
            "optional": {
                "shot_noise_strength": (IO.FLOAT, {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "📸 Shot Noise Strength: Photon noise from light quantization (main noise source).\n\n"
                              "• 0.0: Disable shot noise (unrealistic)\n"
                              "• 0.5: Reduced shot noise\n"
                              "• 1.0: Accurate shot noise levels\n"
                              "• 1.5-2.0: Enhanced for artistic effect\n\n"
                              "This is the primary noise in well-lit areas. Follows Poisson statistics.\n"
                              "📋 Default: 1.0 (scientifically accurate)"
                }),
                
                "read_noise_strength": (IO.FLOAT, {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "🔌 Read Noise Strength: Electronic noise from sensor readout circuitry.\n\n"
                              "• 0.0: Perfect sensor (unrealistic)\n"
                              "• 0.5: High-end sensor performance\n"
                              "• 1.0: Typical DSLR read noise\n"
                              "• 1.5-2.0: Older/budget sensor simulation\n\n"
                              "Most visible in shadows and dark areas. Constant regardless of signal.\n"
                              "📋 Default: 1.0 (typical DSLR)"
                }),
                
                "thermal_noise_strength": (IO.FLOAT, {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "🔥 Thermal Noise Strength: Heat-generated electron noise.\n\n"
                              "• 0.0: No thermal effects\n"
                              "• 0.5: Well-cooled sensor\n"
                              "• 1.0: Normal thermal noise\n"
                              "• 1.5-2.0: Hot sensor conditions\n\n"
                              "Increases exponentially with temperature and exposure time.\n"
                              "📋 Default: 1.0 (normal conditions)"
                }),
                
                "pattern_noise_strength": (IO.FLOAT, {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "📊 Pattern Noise Strength: Fixed pattern and banding noise.\n\n"
                              "• 0.0: Modern sensor (no visible patterns)\n"
                              "• 0.3: Slight banding (some older sensors)\n"
                              "• 0.5-0.7: Noticeable patterns (older DSLRs)\n"
                              "• 1.0: Strong patterns (vintage/damaged sensors)\n\n"
                              "Creates horizontal/vertical banding. Mostly historical interest.\n"
                              "📋 Default: 0.0 (modern sensors are clean)"
                }),
                
                "prnu_strength": (IO.FLOAT, {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "🎯 PRNU Strength: Pixel Response Non-Uniformity.\n\n"
                              "• 0.0: Perfect pixel uniformity (unrealistic)\n"
                              "• 0.5: High-end sensor quality\n"
                              "• 1.0: Typical DSLR variation (~0.5-1%)\n"
                              "• 1.5-2.0: Enhanced variation for effect\n\n"
                              "Each pixel responds slightly differently. Most visible in bright areas.\n"
                              "📋 Default: 1.0 (realistic variation)"
                }),
                
                "color_noise_ratio": (IO.FLOAT, {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "🌈 Color vs Luminance Noise: Controls noise correlation between RGB channels.\n\n"
                              "• 0.0: Pure luminance noise (correlated, grayscale-like)\n"
                              "• 0.25: Realistic DSLR balance (mostly correlated)\n"
                              "• 0.5: Balanced color/luminance noise\n"
                              "• 1.0: Full color noise (independent RGB channels)\n\n"
                              "Lower values = more film-like, Higher values = more digital artifacts.\n"
                              "📋 Default: 0.25 (realistic DSLR behavior)"
                }),
                
                "noise_blur": (IO.FLOAT, {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "🌫️ Noise Blur (HQ): Applies high-quality Gaussian blur to NOISE PATTERNS before adding to image.\n\n"
                              "• 0.0: Sharp, crisp noise (unrealistic for high ISO)\n"
                              "• 0.2-0.5: Subtle noise softening (realistic high ISO)\n"
                              "• 0.6-1.0: Moderate noise blur (very high ISO, low light)\n"
                              "• 1.2-2.0: Strong noise blur (extreme conditions)\n\n"
                              "Real DSLR noise has natural softness due to:\n"
                              "• Sensor physical limitations at pixel level\n"
                              "• Anti-aliasing and optical effects\n"
                              "• Thermal electron diffusion\n\n"
                              "⚠️ IMPORTANT: Blurs NOISE ONLY, not the final image!\n"
                              "📋 Default: 0.0 (sharp noise, set 0.3+ for realism)"
                }),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    OUTPUT_TOOLTIPS = ("📸 Image with scientifically accurate DSLR camera noise applied using real sensor physics.",)
    FUNCTION = "add_dslr_noise"
    CATEGORY = "pirog/image"
    DESCRIPTION = "🔬 Adds scientifically accurate DSLR camera sensor noise based on real sensor physics and measurements. Includes shot noise (Poisson), read noise (electronic), thermal noise (heat), PRNU (pixel variation), and pattern noise (banding). Each component can be controlled independently for realistic or artistic effects."

    def add_dslr_noise(self, image, iso, camera_model, temperature, exposure_time, strength, seed,
                       shot_noise_strength=1.0, read_noise_strength=1.0, thermal_noise_strength=1.0,
                       pattern_noise_strength=0.0, prnu_strength=1.0, color_noise_ratio=0.25, noise_blur=0.0):
        
        # Handle potential string values from ComfyUI's control_after_generate feature
        def safe_float(value, default):
            """Safely convert value to float, using default if conversion fails"""
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return default
        
        # Sanitize all float parameters
        shot_noise_strength = safe_float(shot_noise_strength, 1.0)
        read_noise_strength = safe_float(read_noise_strength, 1.0)
        thermal_noise_strength = safe_float(thermal_noise_strength, 1.0)
        pattern_noise_strength = safe_float(pattern_noise_strength, 0.0)
        prnu_strength = safe_float(prnu_strength, 1.0)
        color_noise_ratio = safe_float(color_noise_ratio, 0.25)
        noise_blur = safe_float(noise_blur, 0.0)
        
        # Set random seeds for reproducibility
        # Ensure NumPy seed is within valid range (0 to 2^32 - 1)
        np_seed = seed % (2**32)
        torch.manual_seed(seed)
        np.random.seed(np_seed)
        
        # Convert to numpy for processing
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape
        
        # Camera sensor parameters based on research data
        camera_params = {
            "modern_fullframe": {"read_noise_e": 2.5, "full_well": 90000, "gain_factor": 1.0, "prnu_factor": 0.006},
            "modern_apsc": {"read_noise_e": 3.2, "full_well": 60000, "gain_factor": 1.1, "prnu_factor": 0.008},
            "older_sensor": {"read_noise_e": 5.5, "full_well": 45000, "gain_factor": 1.3, "prnu_factor": 0.012},
            "high_end": {"read_noise_e": 1.8, "full_well": 120000, "gain_factor": 0.9, "prnu_factor": 0.004}
        }
        
        params = camera_params[camera_model]
        
        # ISO-dependent parameters (based on camera research)
        # Read noise increases with ISO due to amplification
        iso_factor = iso / 100.0
        read_noise_adu = params["read_noise_e"] * np.sqrt(iso_factor) * params["gain_factor"]
        
        # Thermal noise (doubles every ~6-8°C above 20°C)
        temp_factor = 2.0 ** ((temperature - 20.0) / 7.0) if temperature > 20 else 1.0
        thermal_noise_rate = 0.1 * temp_factor * exposure_time  # electrons per second per pixel
        
        # Convert images to linear sensor space (critical for accurate noise modeling)
        # Most images are in sRGB, convert to linear
        linear_images = self.srgb_to_linear(img_np)
        
        # Scale to ADU space (typical DSLR: 0-4095 or 0-16383)
        max_adu = 4095.0  # 12-bit sensor
        linear_adu = linear_images * max_adu
        
        output_images = []
        
        for b in range(batch_size):
            img_adu = linear_adu[b].copy()
            
            # 1. SHOT NOISE (Poisson distributed, variance = signal)
            # Research: shot noise variance is EQUAL to the signal level
            if shot_noise_strength > 0:
                # Ensure positive values for Poisson statistics
                signal = np.maximum(img_adu, 0.1)
                
                # Shot noise standard deviation = sqrt(signal)
                shot_std = np.sqrt(signal)
                shot_noise = np.random.normal(0, shot_std, img_adu.shape)
                
                # Apply blur to shot noise if requested
                if noise_blur > 0.0:
                    shot_noise = self.apply_noise_blur(shot_noise, noise_blur)
                
                img_adu += shot_noise * shot_noise_strength * strength
            
            # 2. READ NOISE (Gaussian, constant standard deviation)
            # Research: independent of signal level, increases with ISO
            if read_noise_strength > 0:
                read_std = read_noise_adu
                if color_noise_ratio < 1.0 and channels >= 3:
                    # Generate correlated read noise (more luminance-like)
                    base_noise = np.random.normal(0, read_std, img_adu.shape[:2])
                    luma_noise = np.stack([base_noise] * channels, axis=2)
                    color_noise = np.random.normal(0, read_std, img_adu.shape)
                    
                    # Apply blur to noise components if requested
                    if noise_blur > 0.0:
                        luma_noise = self.apply_noise_blur(luma_noise, noise_blur)
                        color_noise = self.apply_noise_blur(color_noise, noise_blur)
                    
                    read_noise = (luma_noise * (1.0 - color_noise_ratio) + 
                                 color_noise * color_noise_ratio)
                else:
                    # Full color noise (independent per channel)
                    read_noise = np.random.normal(0, read_std, img_adu.shape)
                    
                    # Apply blur to read noise if requested
                    if noise_blur > 0.0:
                        read_noise = self.apply_noise_blur(read_noise, noise_blur)
                
                img_adu += read_noise * read_noise_strength * strength
            
            # 3. THERMAL NOISE (Gaussian, depends on temperature and time)
            # Research: increases exponentially with temperature
            if thermal_noise_strength > 0:
                thermal_std = np.sqrt(thermal_noise_rate * iso_factor)
                if color_noise_ratio < 1.0 and channels >= 3:
                    # Generate correlated thermal noise (more luminance-like)
                    base_noise = np.random.normal(0, thermal_std, img_adu.shape[:2])
                    luma_noise = np.stack([base_noise] * channels, axis=2)
                    color_noise = np.random.normal(0, thermal_std, img_adu.shape)
                    
                    # Apply blur to thermal noise components if requested
                    if noise_blur > 0.0:
                        luma_noise = self.apply_noise_blur(luma_noise, noise_blur)
                        color_noise = self.apply_noise_blur(color_noise, noise_blur)
                    
                    thermal_noise = (luma_noise * (1.0 - color_noise_ratio) + 
                                   color_noise * color_noise_ratio)
                else:
                    # Full color noise (independent per channel)
                    thermal_noise = np.random.normal(0, thermal_std, img_adu.shape)
                    
                    # Apply blur to thermal noise if requested
                    if noise_blur > 0.0:
                        thermal_noise = self.apply_noise_blur(thermal_noise, noise_blur)
                
                img_adu += thermal_noise * thermal_noise_strength * strength
            
            # 4. PRNU (Pixel Response Non-Uniformity)
            # Research: noise proportional to signal, ~0.5-1% variation
            if prnu_strength > 0:
                prnu_factor = params["prnu_factor"]
                prnu_variation = 1.0 + np.random.normal(0, prnu_factor, img_adu.shape)
                img_adu = img_adu * prnu_variation * prnu_strength
            
            # 5. PATTERN NOISE (Fixed pattern + variable banding)
            # Research: visible in older sensors, row/column correlated
            if pattern_noise_strength > 0:
                # Fixed row pattern
                row_pattern = np.random.normal(0, read_noise_adu * 0.3, (height, 1, channels))
                row_pattern = np.tile(row_pattern, (1, width, 1))
                
                # Fixed column pattern  
                col_pattern = np.random.normal(0, read_noise_adu * 0.2, (1, width, channels))
                col_pattern = np.tile(col_pattern, (height, 1, 1))
                
                # Apply blur to pattern noise if requested
                pattern_noise = row_pattern + col_pattern
                if noise_blur > 0.0:
                    pattern_noise = self.apply_noise_blur(pattern_noise, noise_blur)
                
                img_adu += pattern_noise * pattern_noise_strength * strength
            
            # Color noise handling is now done during noise generation above
            # (read noise and thermal noise sections) to preserve image colors
            
            # Convert back to 0-1 range
            img_linear = np.clip(img_adu / max_adu, 0.0, 1.0)
            
            # Convert back to sRGB space
            img_final = self.linear_to_srgb(img_linear)
            
            output_images.append(img_final)
        
        result = np.stack(output_images, axis=0)
        
        return (torch.from_numpy(result).float(),)
    
    def srgb_to_linear(self, srgb):
        """Convert sRGB to linear RGB space"""
        # sRGB to linear conversion (proper, not gamma 2.2)
        linear = np.where(srgb <= 0.04045,
                         srgb / 12.92,
                         np.power((srgb + 0.055) / 1.055, 2.4))
        return linear
    
    def linear_to_srgb(self, linear):
        """Convert linear RGB to sRGB space"""
        # Linear to sRGB conversion
        srgb = np.where(linear <= 0.0031308,
                       linear * 12.92,
                       1.055 * np.power(linear, 1.0/2.4) - 0.055)
        return np.clip(srgb, 0.0, 1.0)
    
    def apply_noise_blur(self, noise, blur_strength):
        """Apply blur specifically to noise patterns before adding to image"""
        if blur_strength <= 0.0:
            return noise
        
        # Convert blur strength to sigma (reasonable range: 0.1 to 2.0)
        sigma = np.clip(blur_strength, 0.1, 2.0)
        
        # Apply blur to noise using existing blur implementation
        return self.apply_gaussian_blur(noise, sigma)
    
    def apply_gaussian_blur(self, image, sigma):
        """Apply high-quality Gaussian blur using scipy if available, otherwise torch fallback"""
        if SCIPY_AVAILABLE:
            # Use scipy for highest quality blur
            blurred = image.copy()
            height, width, channels = image.shape
            for c in range(channels):
                blurred[:,:,c] = gaussian_filter(image[:,:,c], sigma=sigma, mode='reflect')
            return np.clip(blurred, 0.0, 1.0)
        else:
            # Fallback to torch-based blur implementation
            return self.torch_gaussian_blur(image, sigma)
    
    def torch_gaussian_blur(self, image, sigma):
        """Torch-based Gaussian blur implementation as fallback"""
        # Convert numpy to torch
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # BHWC -> BCHW
        
        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Generate 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(img_tensor.size(1), 1, kernel_size, kernel_size)
        
        # Apply blur with padding
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(
            img_tensor, 
            kernel_2d, 
            padding=padding, 
            groups=img_tensor.size(1)
        )
        
        # Convert back to numpy
        blurred = blurred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # BCHW -> HWC
        return np.clip(blurred, 0.0, 1.0)


class TestResetButton:
    """
    Test Reset Button Node
    
    A simple test node that demonstrates custom JavaScript reset button functionality.
    Features an integer input field with a default value of 5 and a reset button that
    restores the value to default when clicked.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "test_value": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Test integer value with reset button functionality"
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    OUTPUT_TOOLTIPS = ("The test integer value",)
    FUNCTION = "get_value"
    CATEGORY = "pirog/image"
    DESCRIPTION = "Test node for demonstrating custom reset button functionality. Features an integer field with reset button that restores to default value (5)."

    def get_value(self, test_value):
        """Returns the test value as output"""
        return (test_value,)