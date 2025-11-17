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
from scipy.ndimage import gaussian_filter
import requests
import base64
from io import BytesIO
from urllib.parse import urlparse
from .utilities import tensor_to_pil, pil_to_tensor, srgb_to_linear, linear_to_srgb, apply_noise_blur, tensor_to_numpy, numpy_to_tensor, safe_poisson

# Try to import LM Studio SDK for model unloading
try:
    import lmstudio
    LMSTUDIO_SDK_AVAILABLE = True
except ImportError:
    LMSTUDIO_SDK_AVAILABLE = False
    print("LM Studio SDK not available. Model unloading will not work. Install with: pip install lmstudio")

logger = logging.getLogger(__name__)


# Add the ComfyUI root directory to Python path to access the main nodes
comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)

# Import the common_ksampler function, MAX_RESOLUTION, and SaveImage directly from the main nodes.py
# This ensures we always use the latest version from ComfyUI
try:
    from nodes import common_ksampler, MAX_RESOLUTION, SaveImage
except ImportError:
    # Fallback: try to import from nodes module in current directory
    spec = importlib.util.spec_from_file_location("nodes", os.path.join(comfy_dir, "nodes.py"))
    nodes_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes_module)
    common_ksampler = nodes_module.common_ksampler
    MAX_RESOLUTION = nodes_module.MAX_RESOLUTION
    SaveImage = nodes_module.SaveImage

# Import required ComfyUI modules
# import comfy.samplers
from comfy.comfy_types import IO, ComfyNodeABC #, InputTypeDict
try:
    import folder_paths
    from PIL.PngImagePlugin import PngInfo
    from comfy.cli_args import args
except ImportError:
    # These will be available at runtime in ComfyUI
    folder_paths = None
    PngInfo = None
    args = None




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

            # Ensure tensors are in BHWC format (ComfyUI standard)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            # ComfyUI tensors are already in BHWC format, no permute needed

            if watermark.dim() == 3:
                watermark = watermark.unsqueeze(0)
            # ComfyUI tensors are already in BHWC format, no permute needed

            batch_size, height, width, channels = image.shape
            processed_images = []

            for i in range(batch_size):
                img_pil = tensor_to_pil(image[i])
                watermark_pil = tensor_to_pil(watermark[0] if watermark.shape[0] > 1 else watermark)

                result = self.process_single_image(img_pil, watermark_pil, scale, opacity, blend_mode, position, invert_watermark, use_black_mask, logger)
                processed_images.append(pil_to_tensor(result))

            # Concatenate processed images along the batch dimension
            result = torch.cat(processed_images, dim=0)
            # Result is already in BHWC format as expected by ComfyUI
            
            return (result,)
        except Exception as e:
            logger.error(f"Error in watermark application: {str(e)}")
            return (image,)  # Return original image in case of error


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
    Image Scaling Tool
    
    Advanced proportional image scaling with resolution limits, step alignment,
    and professional-grade controls. Features MX-style slider compatibility
    for enhanced user experience. Includes selectable resampling methods with an
    automatic mode that chooses the best filter for upscaling or downscaling.
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
                "resampling_method": (["auto", "LANCZOS", "NEAREST", "BILINEAR", "BICUBIC"], {
                    "default": "auto",
                    "tooltip": "Resampling method. 'auto' selects NEAREST for upscaling, LANCZOS for downscaling."
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

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("scaled_image", "width", "height")
    OUTPUT_TOOLTIPS = ("Scaled image(s) with proper aspect ratio preservation", "Width of scaled image", "Height of scaled image")
    FUNCTION = "scale_image"
    CATEGORY = "pirog/transform"
    DESCRIPTION = "Professional image scaling with proportional resize, resolution limits, step alignment, and selectable resampling methods for optimal results."

    def scale_image(self, image, scale_multiplier, resampling_method, resolution_step, enable_limits, min_resolution, max_resolution):
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")

            batch_size, height, width, channels = image.shape
            scaled_images = []

            # Map string names to PIL resampling filters
            resample_filters = {
                "LANCZOS": Image.LANCZOS,
                "NEAREST": Image.NEAREST,
                "BILINEAR": Image.BILINEAR,
                "BICUBIC": Image.BICUBIC,
            }

            for i in range(batch_size):
                pil_image = tensor_to_pil(image[i])
                logger.info(f"Original PIL Image size: {pil_image.size}")

                new_width = int(width * scale_multiplier)
                new_height = int(height * scale_multiplier)

                if enable_limits:
                    new_width, new_height = self.apply_resolution_limits(new_width, new_height, min_resolution, max_resolution)

                new_width = max((new_width // resolution_step) * resolution_step, resolution_step)
                new_height = max((new_height // resolution_step) * resolution_step, resolution_step)

                logger.info(f"Calculated new dimensions: {new_width}x{new_height}")

                # Determine resampling method
                if resampling_method == "auto":
                    # Downscaling: LANCZOS is better. Upscaling: NEAREST is sharp and fast.
                    resample_filter = Image.LANCZOS if scale_multiplier < 1.0 else Image.NEAREST
                else:
                    resample_filter = resample_filters.get(resampling_method, Image.LANCZOS)
                
                logger.info(f"Using resampling method: {resampling_method} -> {resample_filter}")

                resized_image = pil_image.resize((new_width, new_height), resample_filter)
                logger.info(f"Resized PIL Image size: {resized_image.size}")

                scaled_images.append(pil_to_tensor(resized_image))

            output_tensor = torch.cat(scaled_images, dim=0)
            logger.info(f"Output tensor shape: {output_tensor.shape}")

            return (output_tensor, new_width, new_height)

        except Exception as e:
            logger.error(f"Error in scale_image: {str(e)}")
            logger.error(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")
            return (image, width, height)


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


class GetImageSize:
    """
    Get Image Dimensions

    Extracts width and height dimensions from an image tensor.
    Returns integer values for width and height.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to get dimensions from"})
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    OUTPUT_TOOLTIPS = ("Width of the image", "Height of the image")
    FUNCTION = "get_size"
    CATEGORY = "pirog/utility"
    DESCRIPTION = "Extract width and height dimensions from an image."

    def get_size(self, image):
        """
        Extract width and height from image tensor.

        Args:
            image: Image tensor in ComfyUI format, can be BHWC or HWC

        Returns:
            tuple: (width, height) as integers
        """
        # Image tensor shape is either (batch_size, height, width, channels) or (height, width, channels).
        # We can reliably get height and width from the second and third last dimensions.
        height = image.shape[-3]
        width = image.shape[-2]

        return (width, height)


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
                }),
                "prompts_count": ("INT", {
                    "default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of randomized prompts to generate. Useful for batch processing with samplers that support multiple prompts."
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("modified_prompts", "selections", "used_seed")
    FUNCTION = "randomize_prompt"
    CATEGORY = "pirog/text"
    DESCRIPTION = "Randomizes text prompts using pattern replacement with dictionary support, validation, and flexible text processing options. Can generate multiple prompts for batch processing."

    def randomize_prompt(self, prompt, dictionary_filename, seed, preserve_newlines, clean_text, prompts_count):
        # Handle input processing - convert arrays to strings if needed
        if isinstance(prompt, list):
            prompt = " ".join(str(x) for x in prompt)

        # Get dictionary path relative to this node pack
        dictionary_path = self.get_dictionary_path(dictionary_filename)

        # Load or create the dictionary
        random_dictionary = self.load_or_create_dictionary(dictionary_path)

        generated_prompts = []
        all_selections_list = []

        # Generate multiple prompts with different seeds
        for prompt_idx in range(prompts_count):
            # Set the seed for this prompt (increment from base seed)
            current_seed = seed + prompt_idx if seed != -1 else random.randint(0, 0xffffffffffffffff) + prompt_idx
            random.seed(current_seed)

            # Work on a copy of the original prompt
            current_prompt = prompt
            current_selections = []

            # Process randomization for this prompt
            for i in range(10):  # Max 10 iterations to prevent infinite loops
                # Find all tags in the prompt string
                tags = re.findall(r"\?\s*[^\?]+\s*\?", current_prompt)

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
                    current_prompt = current_prompt.replace(tag, word, 1)

                    # Add only non-empty selections to array for later display
                    if word and word.strip():
                        current_selections.append(word.strip())

            # Apply text processing based on settings
            if clean_text:
                current_prompt = self.clean_prompt(current_prompt)

            if not preserve_newlines:
                current_prompt = self.remove_newlines(current_prompt)

            generated_prompts.append(current_prompt)
            all_selections_list.append(", ".join(current_selections))

        # Use the first seed as the "used_seed" for compatibility
        used_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)

        # Return single prompt/string when count=1, list when count>1
        if prompts_count == 1:
            return (generated_prompts[0], all_selections_list[0], used_seed)
        else:
            return (generated_prompts, all_selections_list, used_seed)

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
                logging.info(f"Created default dictionary at: {dictionary_path}")
            except Exception as e:
                logging.warning(f"Could not create dictionary file: {e}")
            return default_dict
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON in dictionary file: {dictionary_path}. Using default dictionary. Error: {e}")
            return self.create_default_dictionary()
        except Exception as e:
            logging.error(f"Error reading dictionary file: {dictionary_path}. Using default dictionary. Error: {e}")
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


# --- Camera Sensor Profiles ---
# Based on research into camera sensor performance. These profiles dictate the
# physical characteristics of the simulated sensor.
CAMERA_PROFILES = {
    "Pro Full-Frame": {
        "read_noise_base_e": 1.5,      # Electrons of read noise at base ISO (e.g., ISO 100)
        "read_noise_iso_scaling": 0.15,# How much read noise increases with analog gain
        "dark_current_base": 0.1,     # Electrons/sec at 20°C - a measure of thermal noise
        "prnu_factor": 0.004,          # Photo Response Non-Uniformity (multiplicative noise)
        "dsnu_factor": 0.002,          # Dark Signal Non-Uniformity (additive fixed pattern noise)
        "quantization_levels": 16384,  # 14-bit ADC (Analog-to-Digital Converter)
        "full_well_capacity": 90000,   # Maximum electrons a pixel can hold
        "chroma_noise_factor": 0.4,    # Balance between luminance and chrominance noise
    },
    "APS-C Enthusiast": {
        "read_noise_base_e": 2.5,
        "read_noise_iso_scaling": 0.25,
        "dark_current_base": 1.5,
        "prnu_factor": 0.007,
        "dsnu_factor": 0.005,
        "quantization_levels": 4096,   # 12-bit ADC
        "full_well_capacity": 60000,
        "chroma_noise_factor": 0.5,
    },
    "Smartphone": {
        "read_noise_base_e": 4.0,
        "read_noise_iso_scaling": 0.35,
        "dark_current_base": 1.2,
        "prnu_factor": 0.01,
        "dsnu_factor": 0.008,
        "quantization_levels": 1024,   # 10-bit ADC
        "full_well_capacity": 15000,
        "chroma_noise_factor": 0.6,
    },
    "Vintage DSLR": {
        "read_noise_base_e": 5.5,
        "read_noise_iso_scaling": 0.4,
        "dark_current_base": 4.5,
        "prnu_factor": 0.012,
        "dsnu_factor": 0.01,
        "quantization_levels": 4096,   # 12-bit ADC
        "full_well_capacity": 45000,
        "chroma_noise_factor": 0.7,
    }
}


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
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The input image to apply noise to."
                }),
                "iso": ("INT", {
                    "default": 800, "min": 50, "max": 12800, "step": 50,
                    "tooltip": "Higher ISO increases noise, simulating higher light sensitivity."
                }),
                "profile": (list(CAMERA_PROFILES.keys()), {
                    "tooltip": "Select a camera sensor profile. Each profile has distinct noise characteristics."
                }),
                "temperature": ("FLOAT", {
                    "default": 25.0, "min": -20.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Sensor temperature (Celsius). Higher temperatures increase thermal noise."
                }),
                "exposure_time": ("FLOAT", {
                    "default": 1/125.0, "min": 0.0, "max": 30.0, "step": 0.001,
                    "tooltip": "Shutter speed (seconds). Longer exposures increase thermal noise and affect shot noise."
                }),
                "overall_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Artistic multiplier for the final calculated noise. 1.0 is physically accurate."
                }),
                "noise_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Applies a Gaussian blur to the final noise pattern to simulate softness."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "The random seed for the noise generation, allowing for reproducible patterns."
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    OUTPUT_TOOLTIPS = ("📸 Image with scientifically accurate DSLR camera noise applied using real sensor physics.",)
    FUNCTION = "add_camera_noise"
    CATEGORY = "pirog/image"
    DESCRIPTION = "🔬 Adds scientifically accurate DSLR camera sensor noise based on real sensor physics and measurements. Includes shot noise (Poisson), read noise (electronic), thermal noise (heat), PRNU (pixel variation), and pattern noise (banding). Each component can be controlled independently for realistic or artistic effects."

    def add_camera_noise(self, image, iso, profile, temperature, exposure_time, overall_strength, noise_blur, seed):
        
        np_seed = seed % (2**32)
        torch.manual_seed(seed)
        np.random.seed(np_seed)
        
        params = CAMERA_PROFILES[profile]
        
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape
        
        linear_images = srgb_to_linear(img_np)
        signal_e = linear_images * params["full_well_capacity"]
        
        output_images = []
        
        for b in range(batch_size):
            current_signal_e = signal_e[b].copy()

            # 1. PRNU (Photo Response Non-Uniformity) - Multiplicative signal noise
            prnu_noise = np.random.normal(1.0, params["prnu_factor"], current_signal_e.shape)
            current_signal_e *= prnu_noise

            ### MODIFIED ###
            # To make exposure_time meaningful, we simulate the lighting conditions. A long exposure
            # implies a dark scene (fewer photons per second). We scale the signal down for the
            # purpose of noise calculation to reflect this, which correctly reduces shot noise
            # and makes read noise more prominent, as it would be in a real long exposure.
            reference_exposure = 1.0 / 125.0  # A typical baseline shutter speed
            safe_exposure_time = max(exposure_time, 1e-6) # Avoid division by zero
            signal_scaler = reference_exposure / safe_exposure_time
            scaled_signal_for_noise_calc = current_signal_e * signal_scaler

            # 2. Shot Noise (Pre-gain)
            # We now use the scaled signal to calculate shot noise. The original signal is used
            # later to preserve the image's actual brightness.
            shot_noise_e = safe_poisson(np.maximum(scaled_signal_for_noise_calc, 0)) - scaled_signal_for_noise_calc

            # 3. Thermal Noise (Dark Current & DSNU) (Pre-gain)
            # This part correctly uses exposure_time already, so no changes are needed.
            temp_factor = 2.0 ** ((temperature - 20.0) / 7.0)
            dark_current_e = params["dark_current_base"] * temp_factor * exposure_time
            dsnu_pattern = np.random.normal(1.0, params["dsnu_factor"], current_signal_e.shape)
            dark_current_with_dsnu = dark_current_e * dsnu_pattern
            thermal_noise_e = safe_poisson(np.maximum(dark_current_with_dsnu, 0)) - dark_current_with_dsnu

            # 4. ISO Gain and Read Noise (Post-gain)
            gain = max(iso / 100.0, 1.0)
            read_noise_std = params["read_noise_base_e"] + (gain - 1.0) * params["read_noise_iso_scaling"]

            base_luma_noise = np.random.normal(0, read_noise_std, (height, width, 1))

            # Create low-resolution chroma noise and upsample to match image dimensions
            chroma_h, chroma_w = max(1, height // 2), max(1, width // 2)
            lowres_chroma_noise = np.random.normal(0, read_noise_std, (chroma_h, chroma_w, channels))

            # Upsample by repeating, ensuring we get at least the target dimensions
            repeat_h = int(np.ceil(height / chroma_h))
            repeat_w = int(np.ceil(width / chroma_w))
            chroma_noise = np.repeat(np.repeat(lowres_chroma_noise, repeat_h, axis=0), repeat_w, axis=1)
            # Crop to exact dimensions
            chroma_noise = chroma_noise[:height, :width, :]

            chroma_factor = params["chroma_noise_factor"]
            read_noise = base_luma_noise * (1.0 - chroma_factor) + chroma_noise * chroma_factor
            
            # Note: The blur application was removed from here.

            # 5. Combine All Noise Components
            # The original signal is NOT amplified. Pre-gain noise is amplified. Post-gain noise is not.
            total_noise = (shot_noise_e + thermal_noise_e) * gain + read_noise

            ### MODIFIED ###
            # The blur is now applied to the FINAL combined noise map. This is more efficient
            # and better simulates how the entire camera pipeline (demosaicing, etc.) can
            # soften the appearance of noise.
            if noise_blur > 0.0:
                # Assuming apply_noise_blur takes the noise map and a sigma value for the blur.
                # A direct mapping is used here, but you could scale it, e.g., blur_sigma = noise_blur * 0.8
                blur_sigma = noise_blur
                total_noise = apply_noise_blur(total_noise, blur_sigma)

            # 6. Add Noise to Signal and Apply Overall Strength
            # The base signal is the original `current_signal_e` to preserve brightness.
            signal_with_all_noise = current_signal_e + (total_noise * overall_strength)

            # 7. Quantize and Convert Back
            conversion_gain = params["full_well_capacity"] / params["quantization_levels"]
            final_adu = signal_with_all_noise / conversion_gain
            clipped_adu = np.clip(final_adu, 0, params["quantization_levels"])
            img_linear = clipped_adu / params["quantization_levels"]
            img_final = linear_to_srgb(img_linear)
            
            output_images.append(img_final)
            
        result = np.stack(output_images, axis=0)
        return (torch.from_numpy(result).float(),)


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


class LensSimulatedBloom:
    """
    Lens-simulated Bloom Node
    
    Applies a high-quality, hybrid bloom effect to input images using a combination of
    multi-pass downsampling blur (haze) and FFT convolution with custom kernels (spikes).
    This creates realistic camera lens bloom effects with soft atmospheric glow and 
    sharp structured diffraction artifacts.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Luminosity threshold to isolate highlights (0.0-1.0)"
                }),
                "iterations": ("INT", {
                    "default": 4, 
                    "min": 1, 
                    "max": 16, 
                    "step": 1,
                    "tooltip": "Number of blur layers to generate"
                }),
                "amount": ("FLOAT", {
                    "default": 10.0, 
                    "min": 0.1, 
                    "max": 100.0, 
                    "step": 0.1,
                    "tooltip": "Initial blur strength, which is halved for each subsequent iteration"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Opacity of the bloom effect. 0.0 is original image, 1.0 is full bloom."
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_bloom"
    CATEGORY = "Pirog/image"
    DESCRIPTION = "Applies a bloom effect based on a Photoshop workflow using luminosity masking and layer blending."

    def apply_bloom(self, image, threshold, iterations, amount, opacity):
        """Apply the bloom effect to input images"""
        try:
            import cv2
            import numpy as np

        except ImportError:
            raise RuntimeError("Missing required libraries for Bloom node. Please install numpy and opencv-python.")

        img_np_in = tensor_to_numpy(image)

        output_images = []
        for single_img_in in img_np_in:
            
            img_uint8 = (single_img_in * 255).astype(np.uint8)
            
            lab_image = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            
            l_channel = lab_image[:,:,0]
            
            threshold_value = threshold * 255
            
            _, mask = cv2.threshold(l_channel, threshold_value, 255, cv2.THRESH_BINARY)
            
            mask_3d = (np.expand_dims(mask, axis=2) > 0).astype(float)
            
            processing_canvas = single_img_in * mask_3d
            
            blurred_layers = []
            current_blur_amount = amount
            for _ in range(iterations):
                if current_blur_amount > 0:
                    # Kernel size must be an odd number, derived from sigma (amount)
                    ksize = int(current_blur_amount * 3) 
                    if ksize % 2 == 0:
                        ksize += 1
                    
                    blurred_layer = cv2.GaussianBlur(processing_canvas, (ksize, ksize), current_blur_amount)
                    blurred_layers.append(blurred_layer)
                else:
                    blurred_layers.append(processing_canvas.copy())
                
                current_blur_amount /= 2.0
            
            bloom_result_image = single_img_in.copy()
            
            for blurred_layer in blurred_layers:
                bloom_result_image = np.maximum(bloom_result_image, blurred_layer)

            # Blend with original image based on opacity
            final_image = (single_img_in * (1.0 - opacity)) + (bloom_result_image * opacity)
                
            final_image = np.clip(final_image, 0.0, 1.0)
            
            output_images.append(final_image)

        result_array = np.stack(output_images, axis=0)
        result_tensor = numpy_to_tensor(result_array)
        
        return (result_tensor,)



# ====================================
# CROP-UNCROP NODES INTEGRATION
# ====================================

class CropImage:
    """
    Crops images from specified sides (left, right, top, bottom)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "pirog/image"

    def crop(self, images, left, right, top, bottom):
        result_images = []
        
        for img in images:
            height, width = img.shape[:2]
            
            # Calculate remaining dimensions after cropping
            new_width = width - left - right
            new_height = height - top - bottom
            
            # Ensure dimensions don't become negative
            if new_width <= 0 or new_height <= 0:
                logging.warning(f"Crop values too large for image size {width}x{height}")
                result_images.append(img)
                continue
                
            # Crop image sequentially
            result = img
            if left > 0:
                result = result[:, left:]
            if right > 0:
                result = result[:, :-right]
            if top > 0:
                result = result[top:]
            if bottom > 0:
                result = result[:-bottom]
                
            result_images.append(result)
            
        return (torch.stack(result_images),)


class BatchCropFromMaskSimple:
    """
    Crops images based on mask boundaries with expansion
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "expansion": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "use_binary_mask": ("BOOLEAN", {"default": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOX")
    RETURN_NAMES = ("cropped_images", "bboxes")
    FUNCTION = "crop"
    CATEGORY = "pirog/image"

    def get_bbox_from_mask(self, mask_tensor, expansion, use_binary_mask, threshold):
        # Apply binary thresholding if requested
        if use_binary_mask:
            # Convert to binary: values >= threshold become 1.0, values < threshold become 0.0
            mask = (mask_tensor >= threshold).cpu().numpy()
        else:
            # Use original grayscale behavior (anything > 0.5 is considered active)
            mask = (mask_tensor > 0.5).cpu().numpy()
        
        y_indices, x_indices = np.nonzero(mask)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
            
        min_x = max(0, np.min(x_indices) - expansion)
        max_x = min(mask.shape[1], np.max(x_indices) + expansion)
        min_y = max(0, np.min(y_indices) - expansion)
        max_y = min(mask.shape[0], np.max(y_indices) + expansion)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def crop(self, images, masks, expansion, use_binary_mask, threshold):
        bboxes = []
        cropped_images = []

        for img, mask in zip(images, masks):
            bbox = self.get_bbox_from_mask(mask, expansion, use_binary_mask, threshold)
            if bbox is None:
                continue
                
            min_x, min_y, width, height = bbox
            cropped = img[min_y:min_y+height, min_x:min_x+width]
            cropped_images.append(cropped)
            bboxes.append(bbox)

        if not cropped_images:
            return (images, [(0, 0, images[0].shape[1], images[0].shape[0])])

        return (torch.stack(cropped_images), bboxes)


class BatchUncropSimple:
    """
    Uncrops images back to their original size with blending
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "bboxes": ("BBOX",),
                "blend_width": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "uncrop"
    CATEGORY = "pirog/image"

    def create_blend_mask(self, size, blend_width):
        mask = np.ones(size)
        if blend_width > 0:
            for i in range(blend_width):
                alpha = i / blend_width
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
        return torch.from_numpy(mask).float()

    def uncrop(self, original_images, cropped_images, bboxes, blend_width):
        num_outputs = max(len(original_images), len(cropped_images))
        result_images = []
        device = original_images.device

        for i in range(num_outputs):
            # Get original image (cycle if needed)
            orig_img = original_images[i % len(original_images)].clone()
            # Get cropped image and bbox
            crop_img = cropped_images[i % len(cropped_images)]
            bbox = bboxes[i % len(bboxes)]

            x, y, w, h = bbox
            
            # Ensure bbox coordinates are within original image bounds
            orig_h, orig_w = orig_img.shape[:2]
            x = max(0, min(x, orig_w - 1))
            y = max(0, min(y, orig_h - 1))
            w = min(w, orig_w - x)
            h = min(h, orig_h - y)
            
            # Get actual crop dimensions
            crop_h, crop_w = crop_img.shape[:2]
            
            # Always resize crop to match the actual region we're going to fill
            # This ensures dimensional consistency
            crop_img_resized = torch.nn.functional.interpolate(
                crop_img.unsqueeze(0).permute(0, 3, 1, 2),  # Add batch dim and convert to BCHW
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)  # Remove batch dim and convert back to HWC
            
            # Create blend mask with the exact same dimensions as the target region
            target_region_shape = orig_img[y:y+h, x:x+w].shape[:2]
            blend_mask = self.create_blend_mask(target_region_shape, blend_width).to(device)
            
            # Ensure the mask dimensions match exactly
            if blend_mask.shape != target_region_shape:
                logging.warning(f"Warning: Adjusting blend mask from {blend_mask.shape} to {target_region_shape}")
                blend_mask = torch.nn.functional.interpolate(
                    blend_mask.unsqueeze(0).unsqueeze(0),
                    size=target_region_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            for c in range(3):
                orig_img[y:y+h, x:x+w, c] = (
                    crop_img_resized[:, :, c] * blend_mask + 
                    orig_img[y:y+h, x:x+w, c] * (1 - blend_mask)
                )
            
            result_images.append(orig_img)

        return (torch.stack(result_images),)


class CropMaskByBBox:
    """
    Crops masks using bounding box coordinates
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "bboxes": ("BBOX",),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "crop"
    CATEGORY = "pirog/image"

    def crop(self, masks, bboxes):
        cropped_masks = []

        for mask, bbox in zip(masks, bboxes):
            x, y, width, height = bbox
            
            # Crop the mask using bbox coordinates
            cropped = mask[y:y+height, x:x+width]
            cropped_masks.append(cropped)

        # Stack the cropped masks
        return (torch.stack(cropped_masks),)


# ====================================
# MASK PROCESSING NODES
# ====================================

class BlurMask:
    """
    Blurs masks with smart edge handling to prevent feathering at image boundaries
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "blur_mask"
    CATEGORY = "pirog/mask"

    def blur_mask(self, mask, blur_radius, tapered_corners):
        if blur_radius == 0:
            return (mask,)
        
        # Convert mask to numpy for processing
        mask_np = mask.cpu().numpy()
        batch_size = mask_np.shape[0]
        
        result_masks = []
        
        for i in range(batch_size):
            current_mask = mask_np[i]
            height, width = current_mask.shape
            
            if tapered_corners:
                # Create a distance map from image boundaries
                # This identifies how far each pixel is from the nearest image edge
                y_coords, x_coords = np.mgrid[:height, :width]
                
                # Distance from each edge
                dist_from_top = y_coords.astype(np.float32)
                dist_from_bottom = (height - 1 - y_coords).astype(np.float32)
                dist_from_left = x_coords.astype(np.float32)
                dist_from_right = (width - 1 - x_coords).astype(np.float32)
                
                # Minimum distance to any edge - using stacking and min for compatibility
                edge_distance = np.minimum(
                    np.minimum(dist_from_top, dist_from_bottom),
                    np.minimum(dist_from_left, dist_from_right)
                )
                
                # Calculate blur protection zone
                # Pixels within blur_radius distance from edge get protection
                protection_radius = max(1, int(blur_radius * 2))
                boundary_mask = edge_distance < protection_radius
                
                if np.any(boundary_mask):
                    # For pixels near boundaries, use reflection padding to prevent feathering
                    from scipy.ndimage import gaussian_filter
                    blurred_reflected = gaussian_filter(current_mask, sigma=blur_radius, mode='reflect')
                    
                    # For interior pixels, use normal blur
                    blurred_normal = gaussian_filter(current_mask, sigma=blur_radius, mode='constant')
                    
                    # Create smooth transition between boundary and interior regions
                    # Distance-based blending with smooth falloff
                    normalized_distance = np.clip(edge_distance / protection_radius, 0, 1)
                    # Use smoothstep function for natural transition
                    blend_factor = normalized_distance * normalized_distance * (3 - 2 * normalized_distance)
                    
                    # Blend: near boundaries use reflected blur, far from boundaries use normal blur
                    result = blurred_reflected * (1 - blend_factor) + blurred_normal * blend_factor
                else:
                    # If no pixels are near boundaries, just use normal blur
                    from scipy.ndimage import gaussian_filter
                    result = gaussian_filter(current_mask, sigma=blur_radius, mode='constant')
            else:
                # Simple gaussian blur without edge protection
                from scipy.ndimage import gaussian_filter
                result = gaussian_filter(current_mask, sigma=blur_radius, mode='constant')
            
            result_masks.append(torch.from_numpy(result))
        
        return (torch.stack(result_masks),)


class InvertMask:
    """
    Inverts mask values (0 becomes 1, 1 becomes 0)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "invert_mask"
    CATEGORY = "pirog/mask"

    def invert_mask(self, mask):
        return (1.0 - mask,)


class GradientMaskGenerator:
    """
    Generates linear gradient masks with directional control, balance, and contrast adjustment
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "direction": (["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"], {"default": "left_to_right"}),
                "balance": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "soften": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_gradient"
    CATEGORY = "pirog/mask"

    def generate_gradient(self, width, height, direction, balance, contrast, soften):
        # Create base coordinate array (0 to 1)
        if direction == "left_to_right":
            coords = np.linspace(0, 1, width, dtype=np.float32)
            mask = np.tile(coords, (height, 1))
        elif direction == "right_to_left":
            coords = np.linspace(1, 0, width, dtype=np.float32)
            mask = np.tile(coords, (height, 1))
        elif direction == "top_to_bottom":
            coords = np.linspace(0, 1, height, dtype=np.float32)
            mask = np.tile(coords.reshape(-1, 1), (1, width))
        elif direction == "bottom_to_top":
            coords = np.linspace(1, 0, height, dtype=np.float32)
            mask = np.tile(coords.reshape(-1, 1), (1, width))
        
        # Apply balance adjustment - shift the midpoint (0.5 gray) position
        if balance != 0.5:
            # Create piecewise linear gradient with shifted midpoint
            # balance determines where the 0.5 gray value appears in the gradient
            
            # For each pixel position, calculate what the gradient value should be
            mask_balanced = np.zeros_like(mask)
            
            # First segment: 0 to balance position maps to 0.0 to 0.5
            first_segment = mask <= balance
            if np.any(first_segment):
                # Linear interpolation from 0 to 0.5 over 0 to balance range
                mask_balanced[first_segment] = 0.5 * (mask[first_segment] / balance)
            
            # Second segment: balance to 1.0 position maps to 0.5 to 1.0  
            second_segment = mask > balance
            if np.any(second_segment):
                # Linear interpolation from 0.5 to 1.0 over balance to 1.0 range
                mask_balanced[second_segment] = 0.5 + 0.5 * ((mask[second_segment] - balance) / (1.0 - balance))
            
            mask = mask_balanced
        
        # Apply contrast adjustment - expand around 0.5 midpoint toward black/white
        if contrast > 0:
            # Contrast pushes values away from 0.5 toward 0.0 and 1.0
            # Higher contrast = more separation, sharper transitions
            
            # Calculate distance from midpoint (0.5)
            distance_from_mid = np.abs(mask - 0.5)
            
            # Determine which side of midpoint each pixel is on
            above_mid = mask >= 0.5
            # Apply contrast expansion
            # contrast=0: no change, contrast=1: maximum separation
            expanded_distance = distance_from_mid + contrast * (0.5 - distance_from_mid)
            # Reconstruct values on correct side of midpoint
            mask = np.where(above_mid, 
                          0.5 + expanded_distance,  # Above midpoint: push toward 1.0
                          0.5 - expanded_distance)  # Below midpoint: push toward 0.0
            
            # Ensure we stay in valid range
            mask = np.clip(mask, 0.0, 1.0)
        
        # Apply directional softening (blur along gradient direction only)
        if soften > 0:
            mask = self.apply_directional_blur(mask, direction, soften)
        
        # Convert to tensor and add batch dimension
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return (mask_tensor,)

    def apply_directional_blur(self, mask, direction, sigma):
        """
        Apply 1D Gaussian blur only along the gradient direction to soften transitions
        without bleeding to perpendicular sides
        """
        if sigma <= 0:
            return mask
            
        # Calculate kernel size (should be odd)
        kernel_size = int(sigma * 6 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 1D Gaussian kernel
        kernel_1d = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / sigma ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply directional blur based on gradient direction
        if direction in ["left_to_right", "right_to_left"]:
            # Horizontal gradient - blur horizontally only
            # Apply 1D convolution along axis=1 (width/horizontal)
            from scipy.ndimage import convolve1d
            blurred = convolve1d(mask, kernel_1d, axis=1, mode='reflect')
        else:  # top_to_bottom or bottom_to_top
            # Vertical gradient - blur vertically only  
            # Apply 1D convolution along axis=0 (height/vertical)
            from scipy.ndimage import convolve1d
            blurred = convolve1d(mask, kernel_1d, axis=0, mode='reflect')
            
        return blurred


# ====================================
# IMAGE PROCESSING WITH MASKS
# ====================================

class ImageBlendByMask:
    """
    Blends two images using a mask to control the blend regions and amount
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "blend_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"
    CATEGORY = "pirog/image"

    def blend_images(self, source_image, target_image, mask, invert_mask, blend_amount):
        # Ensure all inputs are on the same device
        device = source_image.device
        target_image = target_image.to(device)
        mask = mask.to(device)
        
        # Determine output batch size - use the maximum to process all images
        # When counts differ, we'll cycle through the smaller batch
        batch_size = max(source_image.shape[0], target_image.shape[0], mask.shape[0])
        
        results = []
        
        for i in range(batch_size):
            # Cycle through batches if they have different sizes
            source_idx = i % source_image.shape[0]
            target_idx = i % target_image.shape[0]
            mask_idx = i % mask.shape[0]
            
            source_single = source_image[source_idx:source_idx+1]
            target_single = target_image[target_idx:target_idx+1]
            mask_single = mask[mask_idx:mask_idx+1]
            
            # Resize images to match if they have different dimensions
            if source_single.shape[1:3] != target_single.shape[1:3]:
                # Resize target to match source dimensions
                target_single = torch.nn.functional.interpolate(
                    target_single.permute(0, 3, 1, 2), 
                    size=(source_single.shape[1], source_single.shape[2]), 
                    mode='bilinear', 
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Resize mask to match image dimensions if needed
            if mask_single.shape[1:3] != source_single.shape[1:3]:
                mask_single = torch.nn.functional.interpolate(
                    mask_single.unsqueeze(1), 
                    size=(source_single.shape[1], source_single.shape[2]), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            # Expand mask to match image channels (add channel dimension)
            mask_expanded = mask_single.unsqueeze(-1).expand(-1, -1, -1, source_single.shape[-1])
            
            # Invert mask if requested
            if invert_mask:
                mask_expanded = 1.0 - mask_expanded
            
            # Apply blend amount to mask
            blend_mask = mask_expanded * blend_amount
            
            # Perform the blend: result = source * (1 - blend_mask) + target * blend_mask
            result_single = source_single * (1.0 - blend_mask) + target_single * blend_mask
            
            results.append(result_single)
        
        # Stack all results
        final_result = torch.cat(results, dim=0)
        
        return (final_result,)


class BlurByMask:
    """
    Applies selective blur to an image based on mask values
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "blur_amount": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur_by_mask"
    CATEGORY = "pirog/image"

    def gaussian_blur_torch(self, image, sigma):
        """
        Fast Gaussian blur using PyTorch operations
        """
        if sigma <= 0:
            return image
            
        # Calculate kernel size (should be odd)
        kernel_size = int(sigma * 6 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel_1d = torch.exp(-0.5 * (torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2) ** 2 / sigma ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Reshape for conv2d (out_channels, in_channels, height, width)
        kernel_x = kernel_1d.view(1, 1, 1, kernel_size).expand(image.shape[-1], 1, 1, kernel_size)
        kernel_y = kernel_1d.view(1, 1, kernel_size, 1).expand(image.shape[-1], 1, kernel_size, 1)
        
        # Convert image to (batch, channels, height, width) for conv2d
        img_conv = image.permute(0, 3, 1, 2)
        
        # Apply separable Gaussian blur (horizontal then vertical)
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(img_conv, kernel_x, padding=(0, padding), groups=image.shape[-1])
        blurred = torch.nn.functional.conv2d(blurred, kernel_y, padding=(padding, 0), groups=image.shape[-1])
        
        # Convert back to (batch, height, width, channels)
        return blurred.permute(0, 2, 3, 1)

    def blur_by_mask(self, image, mask, invert_mask, blur_amount):
        # Ensure all inputs are on the same device
        device = image.device
        mask = mask.to(device)
        
        # Handle batch dimensions
        batch_size = min(image.shape[0], mask.shape[0])
        
        # Crop to matching batch size
        image_batch = image[:batch_size]
        mask_batch = mask[:batch_size]
        
        # Resize mask to match image dimensions if needed
        if mask_batch.shape[1:3] != image_batch.shape[1:3]:
            mask_batch = torch.nn.functional.interpolate(
                mask_batch.unsqueeze(1), 
                size=(image_batch.shape[1], image_batch.shape[2]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # Apply blur to the image
        if blur_amount > 0:
            blurred_image = self.gaussian_blur_torch(image_batch, blur_amount)
        else:
            blurred_image = image_batch
        
        # Expand mask to match image channels
        mask_expanded = mask_batch.unsqueeze(-1).expand(-1, -1, -1, image_batch.shape[-1])
        
        # Invert mask if requested
        if invert_mask:
            mask_expanded = 1.0 - mask_expanded
        
        # Blend: where mask is 1.0, use blurred image; where mask is 0.0, use original
        result = image_batch * (1.0 - mask_expanded) + blurred_image * mask_expanded
        
        return (result,)


class PreviewImageQueue(SaveImage):
    """
    PreviewImage node with a queue button
    """
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    CATEGORY = "pirog/image"


class LMStudioQuery(ComfyNodeABC):
    """
    LM Studio Query Node

    Universal node for querying LM Studio server with text and optional image inputs.
    Supports model loading/unloading, multi-prompt generation with different seeds,
    and batch processing of images for descriptions.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "System prompt to set the AI's behavior and role"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                    "tooltip": "User prompt for the query. Use with images for descriptions or standalone for text generation."
                }),
                "model_name": ("STRING", {
                    "default": "",
                    "tooltip": "Model name or partial name to search for. Leave empty to use currently loaded model."
                }),
                "server_url": ("STRING", {
                    "default": "http://localhost:1234",
                    "tooltip": "LM Studio server URL"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Sampling temperature (0.0 = deterministic, 2.0 = very random)"
                }),
                "max_tokens": ("INT", {
                    "default": 256, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "Maximum number of tokens to generate"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling parameter"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible results. 0 = random each time."
                }),
                "prompts_number": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of prompts to generate with different seeds"
                }),
                "unload_after_use": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after processing to free memory"
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Optional images for visual queries (batch processing supported)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("generated_texts", "model_ref")
    OUTPUT_TOOLTIPS = ("Array of generated text responses", "Reference to the loaded model name")
    FUNCTION = "query_lm_studio"
    CATEGORY = "pirog/ai"
    DESCRIPTION = "Query LM Studio for text generation with optional image input, model management, and multi-seed prompt generation."

    def query_lm_studio(self, system_prompt, user_prompt, model_name, server_url, temperature, max_tokens, top_p, seed, prompts_number, unload_after_use, images=None):
        logger = logging.getLogger(__name__)

        # Handle input processing - convert arrays to strings if needed
        if isinstance(system_prompt, list):
            system_prompt = " ".join(str(x) for x in system_prompt)
        if isinstance(user_prompt, list):
            user_prompt = " ".join(str(x) for x in user_prompt)

        try:
            # Get models list and check server in one request
            models_data = self._get_models_and_check_server(server_url)
            if not models_data:
                raise RuntimeError(f"Cannot connect to LM Studio server at {server_url} or no models available")

            models = models_data['models']
            loaded_model_name = models_data.get('loaded_model')

            # Determine which model to use
            if model_name:
                # User specified a model - find it or use loaded one
                target_model = self._find_model(model_name, models)
                if not target_model:
                    raise RuntimeError(f"Model '{model_name}' not found in available models")
                # Use the target model name for API calls
                api_model = target_model
            else:
                # No specific model - use currently loaded
                if not loaded_model_name:
                    raise RuntimeError("No model specified and no model currently loaded")
                api_model = loaded_model_name

            # Prepare images if provided
            image_data_list = []
            if images is not None:
                for i in range(images.shape[0]):
                    img_pil = self._tensor_to_pil(images[i])
                    if img_pil is None:
                        raise RuntimeError(f"Failed to convert image {i} to PIL")
                    img_base64 = self._pil_to_base64(img_pil)
                    image_data_list.append(img_base64)

            # Generate prompts
            generated_texts = []
            for prompt_idx in range(prompts_number):
                current_seed = seed + prompt_idx if seed != 0 else random.randint(0, 0xffffffffffffffff)

                if image_data_list:
                    # Process each image with the current seed
                    for img_base64 in image_data_list:
                        response = self._send_vision_query(
                            server_url, system_prompt, user_prompt, img_base64,
                            temperature, max_tokens, top_p, current_seed, api_model
                        )
                        generated_texts.append(response)
                else:
                    # Text-only query
                    response = self._send_text_query(
                        server_url, system_prompt, user_prompt,
                        temperature, max_tokens, top_p, current_seed, api_model
                    )
                    generated_texts.append(response)

            # Unload model if requested (optional, since LM Studio may auto-manage)
            if unload_after_use and api_model:
                self._unload_model(server_url, api_model)

            return (generated_texts, api_model)

        except Exception as e:
            logger.error(f"Error in LM Studio query: {str(e)}")
            return ([f"Error: {str(e)}"], "")

    def _get_models_and_check_server(self, server_url):
        """Get models list and check server connectivity in one fast request"""
        try:
            response = requests.get(f"{server_url}/v1/models", timeout=2)  # Fast timeout
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                loaded_model = None
                for model in data.get('data', []):
                    if model.get('loaded', False):
                        loaded_model = model['id']
                        break
                return {'models': models, 'loaded_model': loaded_model}
        except Exception as e:
            logging.warning(f"Failed to get models: {e}")
        return None

    def _find_model(self, model_name, available_models):
        """Find model by name (exact or partial match)"""
        # Find exact match first
        for model in available_models:
            if model == model_name:
                return model

        # Find partial match
        for model in available_models:
            if model_name.lower() in model.lower():
                return model

        return None

    def _load_model(self, server_url, model_name):
        """Load a model on LM Studio"""
        try:
            response = requests.post(
                f"{server_url}/v1/models/load",
                json={"model": model_name},
                timeout=60  # Model loading can take time
            )
            if response.status_code == 200:
                return model_name
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
        return None

    def _unload_model(self, server_url, model_name):
        """Unload a model from LM Studio using SDK only"""
        if not LMSTUDIO_SDK_AVAILABLE:
            print(f"Cannot unload model {model_name}: LM Studio SDK not available. Install with: pip install lmstudio")
            return False

        try:
            # Extract host and port from server_url (remove http:// prefix)
            parsed_url = urlparse(server_url)
            api_host = f"{parsed_url.hostname}:{parsed_url.port}"

            # Use SDK for proper unloading - direct unload by model key (recommended approach)
            client = lmstudio.Client(api_host=api_host)
            client.llm.unload(model_name)
            print(f"Successfully unloaded model via SDK: {model_name}")
            return True
        except Exception as e:
            print(f"SDK unload failed for {model_name}: {e}")
            return False

    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        tensor = tensor.cpu().float()

        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)

        if tensor.dim() == 3 and tensor.shape[2] in [1, 3, 4]:  # RGB/RGBA image (H, W, C)
            tensor = torch.clamp(tensor, 0, 1)
            array = (tensor.numpy() * 255).astype('uint8')  # Already (H, W, C)
            if array.shape[2] == 1:
                return Image.fromarray(array.squeeze(axis=2), mode='L')
            elif array.shape[2] == 3:
                return Image.fromarray(array, mode='RGB')
            elif array.shape[2] == 4:
                return Image.fromarray(array, mode='RGBA')
        elif tensor.dim() == 2:  # Grayscale mask (H, W)
            tensor = torch.clamp(tensor, 0, 1)
            array = (tensor.numpy() * 255).astype('uint8')
            return Image.fromarray(array, mode='L')
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}, dim: {tensor.dim()}")

    def _pil_to_base64(self, pil_image):
        """Convert PIL Image to base64 string"""
        if pil_image is None:
            raise ValueError("PIL image is None")
        buffer = BytesIO()
        # Try JPEG first, fallback to PNG if needed
        try:
            pil_image.save(buffer, format='JPEG', quality=95)
        except Exception:
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _sanitize_text_output(self, text):
        """Sanitize LLM output to escape unsafe characters for ComfyUI processing"""
        if not isinstance(text, str):
            return text

        # Escape backslashes first (must be done before other replacements)
        text = text.replace('\\', '\\\\')

        # Escape parentheses that can break CLIP tokenization
        text = text.replace('(', '\\(').replace(')', '\\)')

        # Escape brackets that can cause parsing issues
        text = text.replace('[', '\\[').replace(']', '\\]')

        # Escape curly braces
        text = text.replace('{', '\\{').replace('}', '\\}')

        # Escape quotes that might break string processing
        text = text.replace('"', '\\"').replace("'", "\\'")

        # Convert newlines to spaces with separator
        text = text.replace('\n', ' -- ')

        # Remove control characters and non-printable characters (but keep spaces and tabs)
        text = ''.join(char for char in text if ord(char) >= 32 or char in ' \t\n')

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _send_text_query(self, server_url, system_prompt, user_prompt, temperature, max_tokens, top_p, seed, model):
        """Send text-only query to LM Studio"""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
            "stream": False
        }

        response = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            return self._sanitize_text_output(content)
        else:
            raise RuntimeError(f"Query failed: {response.status_code} - {response.text}")

    def _send_vision_query(self, server_url, system_prompt, user_prompt, image_base64, temperature, max_tokens, top_p, seed, model):
        """Send vision query with image to LM Studio"""
        # Try OpenAI-compatible format first
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
            "stream": False
        }

        response = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            return self._sanitize_text_output(content)
        else:
            # Try alternative format if OpenAI format fails
            logging.warning(f"OpenAI format failed, trying alternative. Status: {response.status_code}")
            alt_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "images": [image_base64],  # Alternative format
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "seed": seed,
                "stream": False
            }

            response = requests.post(f"{server_url}/v1/chat/completions", json=alt_payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                return self._sanitize_text_output(content)
            else:
                raise RuntimeError(f"Vision query failed: {response.status_code} - {response.text}")


class CLIPTextEncodeMultiple(ComfyNodeABC):
    """
    CLIP Text Encode Multiple Node

    Encodes multiple text prompts using a CLIP model into conditioning embeddings.
    Takes a list of strings and returns a list of CONDITIONING for batch processing.
    Useful for tile-based workflows where each tile needs its own prompt.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "texts": ("STRING", {"tooltip": "List of text prompts to encode. Each text will be encoded separately."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("List of conditioning embeddings, one for each input text.",)
    FUNCTION = "encode_multiple"
    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes multiple text prompts into conditioning embeddings for batch processing workflows."

    def encode_multiple(self, clip, texts):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        # Ensure texts is a list
        if isinstance(texts, str):
            # Try to parse string as list (e.g., "['text1', 'text2']")
            try:
                import ast
                parsed = ast.literal_eval(texts)
                if isinstance(parsed, list):
                    texts = parsed
                else:
                    texts = [texts]
            except (ValueError, SyntaxError):
                texts = [texts]
        elif not isinstance(texts, list):
            texts = [str(texts)]

        conditionings = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                tokens = clip.tokenize(text)
                conditioning = clip.encode_from_tokens_scheduled(tokens)
                conditionings.append(conditioning)

        # Return as a list of conditionings for tile processing
        return (conditionings,)


class CLIPTextEncodeFluxMultiple(ComfyNodeABC):
    """
    CLIP Text Encode Flux Multiple Node

    Encodes multiple text prompt pairs using a CLIP model into conditioning embeddings for Flux models.
    Takes lists of strings for clip_l and t5xxl and returns a list of CONDITIONING for batch processing.
    Useful for tile-based workflows where each tile needs its own prompt pair.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "clip_l_texts": ("STRING", {"tooltip": "List of CLIP-L text prompts to encode. Each text will be encoded separately."}),
                "t5xxl_texts": ("STRING", {"tooltip": "List of T5-XXL text prompts to encode. Each text will be paired with corresponding clip_l_texts."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Guidance scale for Flux models."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("List of conditioning embeddings, one for each input text pair.",)
    FUNCTION = "encode_flux_multiple"
    CATEGORY = "advanced/conditioning/flux"
    DESCRIPTION = "Encodes multiple text prompt pairs into conditioning embeddings for Flux models in batch processing workflows."

    def encode_flux_multiple(self, clip, clip_l_texts, t5xxl_texts, guidance):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        # Ensure inputs are lists
        def parse_input(input_val):
            if isinstance(input_val, str):
                # Try to parse string as list (e.g., "['text1', 'text2']")
                try:
                    import ast
                    parsed = ast.literal_eval(input_val)
                    if isinstance(parsed, list):
                        return parsed
                    else:
                        return [input_val]
                except (ValueError, SyntaxError):
                    return [input_val]
            elif not isinstance(input_val, list):
                return [str(input_val)]
            return input_val

        clip_l_texts = parse_input(clip_l_texts)
        t5xxl_texts = parse_input(t5xxl_texts)

        # Ensure both lists have the same length
        max_len = max(len(clip_l_texts), len(t5xxl_texts))
        if len(clip_l_texts) < max_len:
            clip_l_texts.extend([clip_l_texts[-1]] * (max_len - len(clip_l_texts)))
        if len(t5xxl_texts) < max_len:
            t5xxl_texts.extend([t5xxl_texts[-1]] * (max_len - len(t5xxl_texts)))

        conditionings = []
        for clip_l_text, t5xxl_text in zip(clip_l_texts, t5xxl_texts):
            if isinstance(clip_l_text, str) and isinstance(t5xxl_text, str) and (clip_l_text.strip() or t5xxl_text.strip()):
                tokens = clip.tokenize(clip_l_text)
                tokens["t5xxl"] = clip.tokenize(t5xxl_text)["t5xxl"]
                conditioning = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})
                conditionings.append(conditioning)

        # Return as a list of conditionings for tile processing
        return (conditionings,)


class BatchLoadImages(ComfyNodeABC):
    """
    Batch Load Images Node

    Loads all images from specified directory.
    Supports JPG, PNG, WEBP formats.
    Returns image batch and corresponding filenames.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "./input_images",
                    "tooltip": "Path to directory containing images to load"
                }),
                "recursive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Search recursively in subdirectories"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filenames")
    OUTPUT_TOOLTIPS = ("Batch of loaded images", "List of corresponding filenames")
    FUNCTION = "load_images"
    CATEGORY = "pirog/image"
    DESCRIPTION = "Loads all JPG, PNG, WEBP images from specified directory. Returns image batch and filename list for batch processing."

    def load_images(self, directory_path, recursive):
        """Load all supported images from directory"""
        import os
        from PIL import Image

        supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
        images = []
        filenames = []

        try:
            # Normalize path
            directory_path = os.path.expanduser(directory_path)

            if recursive:
                # Recursive search
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        if file.lower().endswith(supported_formats):
                            full_path = os.path.join(root, file)
                            try:
                                # Load image
                                pil_image = Image.open(full_path)
                                # Convert to RGB if needed
                                if pil_image.mode != 'RGB':
                                    pil_image = pil_image.convert('RGB')

                                # Convert to tensor
                                img_tensor = pil_to_tensor(pil_image)
                                images.append(img_tensor)
                                filenames.append(file)
                            except Exception as e:
                                print(f"Error loading {full_path}: {e}")
                                continue
            else:
                # Non-recursive search
                if os.path.exists(directory_path):
                    for file in os.listdir(directory_path):
                        if file.lower().endswith(supported_formats):
                            full_path = os.path.join(directory_path, file)
                            try:
                                # Load image
                                pil_image = Image.open(full_path)
                                # Convert to RGB if needed
                                if pil_image.mode != 'RGB':
                                    pil_image = pil_image.convert('RGB')

                                # Convert to tensor
                                img_tensor = pil_to_tensor(pil_image)
                                images.append(img_tensor)
                                filenames.append(file)
                            except Exception as e:
                                print(f"Error loading {full_path}: {e}")
                                continue

            if not images:
                raise RuntimeError(f"No supported images found in directory: {directory_path}")

            # Stack images into batch
            image_batch = torch.stack(images)

            return (image_batch, filenames)

        except Exception as e:
            raise RuntimeError(f"Error loading images from {directory_path}: {e}")



class BatchSaveImages(ComfyNodeABC):
    """
    Batch Save Images Node

    Saves batch of images with optional auto-renaming.
    Supports multiple formats and custom output directory.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of images to save"
                }),
                "filenames": ("STRING", {
                    "tooltip": "List of filenames (used when auto_rename is False)"
                }),
                "output_directory": ("STRING", {
                    "default": "./output_images",
                    "tooltip": "Directory to save images to"
                }),
                "auto_rename": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-generate filenames in format 0001_filename.jpg (ignores input filenames)"
                }),
                "base_filename": ("STRING", {
                    "default": "image",
                    "tooltip": "Base name for auto-generated filenames (only used when auto_rename is True)"
                }),
                "format": (["jpg", "png", "webp"], {
                    "default": "jpg",
                    "tooltip": "Output image format"
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Image quality (for JPG/WEBP formats)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths")
    OUTPUT_TOOLTIPS = ("List of paths where images were saved",)
    FUNCTION = "save_images"
    CATEGORY = "pirog/image"
    DESCRIPTION = "Saves batch of images with auto-renaming option. Generates filenames like 0001_basename.jpg or uses provided filenames. Supports JPG, PNG, WEBP formats."

    def save_images(self, images, filenames, output_directory, auto_rename, base_filename, format, quality):
        """Save batch of images to disk"""
        import os
        from PIL import Image

        saved_paths = []

        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Ensure filenames is a list
            if isinstance(filenames, str):
                # If single string, create list with one element
                filenames = [filenames] * images.shape[0]
            elif not isinstance(filenames, list):
                filenames = [str(filenames)] * images.shape[0]

            # Ensure we have enough filenames
            while len(filenames) < images.shape[0]:
                filenames.append(f"image_{len(filenames)}")

            batch_size = images.shape[0]

            for i in range(batch_size):
                try:
                    # Get image tensor
                    img_tensor = images[i]

                    # Convert to PIL
                    pil_image = tensor_to_pil(img_tensor)

                    # Generate filename
                    if auto_rename:
                        # Format: 0001_basename.jpg
                        filename = "04d"
                        if base_filename:
                            filename = f"{filename}_{base_filename}"
                        filename = f"{filename}.{format}"
                    else:
                        # Use provided filename but change extension
                        original_name = filenames[i] if i < len(filenames) else f"image_{i}"
                        # Remove extension and add new one
                        name_without_ext = os.path.splitext(original_name)[0]
                        filename = f"{name_without_ext}.{format}"

                    # Full path
                    full_path = os.path.join(output_directory, filename)

                    # Save with appropriate format settings
                    if format.lower() == "jpg":
                        pil_image.save(full_path, "JPEG", quality=quality)
                    elif format.lower() == "png":
                        pil_image.save(full_path, "PNG")
                    elif format.lower() == "webp":
                        pil_image.save(full_path, "WEBP", quality=quality)

                    saved_paths.append(full_path)
                    print(f"Saved image to: {full_path}")

                except Exception as e:
                    print(f"Error saving image {i}: {e}")
                    continue

            return (saved_paths,)

        except Exception as e:
            raise RuntimeError(f"Error saving images: {e}")



class BlendImages(ComfyNodeABC):
    """
    Blend Images Node

    Blends two images using various blend modes supported by PIL.
    Provides full control over blend amount and supports all major blend modes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", {"tooltip": "First image to blend"}),
                "image2": ("IMAGE", {"tooltip": "Second image to blend"}),
                "blend_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Opacity control: 0.0 = original image, 1.0 = full blend effect. For normal mode: blends between image1/image2 directly."
                }),
                "blend_mode": ([
                    "normal", "add", "subtract", "multiply", "screen",
                    "overlay", "soft_light", "hard_light", "color_dodge",
                    "color_burn", "lighten", "darken", "difference"
                ], {"tooltip": "Blend mode to use for combining the images"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Blended image result",)
    FUNCTION = "blend_images"
    CATEGORY = "pirog/image"
    DESCRIPTION = "Blends two images using various PIL-supported blend modes. Blend amount controls opacity: 0.0 = original image, 1.0 = full blend effect. Normal mode blends directly between input images."

    def blend_images(self, image1, image2, blend_amount, blend_mode):
        """Blend two images using the specified blend mode and amount"""
        try:
            # Ensure tensors are on the same device
            device = image1.device
            image2 = image2.to(device)

            # Handle batch processing - use the maximum batch size
            batch_size = max(image1.shape[0], image2.shape[0])
            results = []

            for i in range(batch_size):
                # Cycle through batches if sizes differ
                img1_idx = i % image1.shape[0]
                img2_idx = i % image2.shape[0]

                # Convert tensors to PIL images
                pil_img1 = tensor_to_pil(image1[img1_idx])
                pil_img2 = tensor_to_pil(image2[img2_idx])

                # Ensure images are the same size
                if pil_img1.size != pil_img2.size:
                    # Resize image2 to match image1
                    pil_img2 = pil_img2.resize(pil_img1.size, Image.LANCZOS)

                # Ensure both images are in RGB mode
                if pil_img1.mode != 'RGB':
                    pil_img1 = pil_img1.convert('RGB')
                if pil_img2.mode != 'RGB':
                    pil_img2 = pil_img2.convert('RGB')

                # Apply the blend mode
                if blend_mode == "normal":
                    # Simple alpha blending between image1 and image2
                    blended = Image.blend(pil_img1, pil_img2, blend_amount)
                else:
                    # For non-normal modes: compute full blend result C, then blend between A and C
                    full_blend_result = self._apply_blend_mode(pil_img1, pil_img2, blend_mode)
                    # blend_amount controls opacity between original image1 (A) and full blend result (C)
                    blended = Image.blend(pil_img1, full_blend_result, blend_amount)

                # Convert back to tensor
                result_tensor = pil_to_tensor(blended)
                results.append(result_tensor)

            # Concatenate results along the batch dimension
            final_result = torch.cat(results, dim=0)
            return (final_result,)

        except Exception as e:
            logger.error(f"Error in blend_images: {str(e)}")
            # Return first image as fallback
            return (image1,)

    def _apply_blend_mode(self, img1, img2, mode):
        """Apply specific blend mode using ImageChops"""
        if mode == "add":
            return ImageChops.add(img1, img2, scale=1.0, offset=0)
        elif mode == "subtract":
            return ImageChops.subtract(img1, img2, scale=1.0, offset=0)
        elif mode == "multiply":
            return ImageChops.multiply(img1, img2)
        elif mode == "screen":
            # Screen blend: 1 - (1-a)(1-b)
            return ImageChops.screen(img1, img2)
        elif mode == "overlay":
            return ImageChops.overlay(img1, img2)
        elif mode == "soft_light":
            return ImageChops.soft_light(img1, img2)
        elif mode == "hard_light":
            return ImageChops.hard_light(img1, img2)
        elif mode == "color_dodge":
            return ImageChops.add(img1, img2, scale=1.0, offset=0)  # Approximation
        elif mode == "color_burn":
            return ImageChops.subtract(img1, img2, scale=1.0, offset=0)  # Approximation
        elif mode == "lighten":
            return ImageChops.lighter(img1, img2)
        elif mode == "darken":
            return ImageChops.darker(img1, img2)
        elif mode == "difference":
            return ImageChops.difference(img1, img2)
        else:
            # Fallback to normal blend
            return Image.blend(img1, img2, 0.5)


class IfNode(ComfyNodeABC):
    """
    IF Node

    Conditional node that returns one of two inputs based on boolean condition.
    Supports any data type for true/false inputs.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "condition": ("BOOLEAN", {
                    "tooltip": "Boolean condition - True returns input_true, False returns input_false"
                }),
                "input_true": ("*", {
                    "tooltip": "Value to return when condition is True"
                }),
                "input_false": ("*", {
                    "tooltip": "Value to return when condition is False"
                }),
            }
        }

    RETURN_TYPES = ("*",)
    OUTPUT_TOOLTIPS = ("The selected input based on condition",)
    FUNCTION = "execute"
    CATEGORY = "pirog/logic"
    DESCRIPTION = "Conditional node that selects between two inputs based on boolean condition. Returns input_true if condition is True, input_false if condition is False."

    def execute(self, condition, input_true, input_false):
        """Execute conditional logic"""
        if condition:
            return (input_true,)
        else:
            return (input_false,)


class LMStudioUnloadModel(ComfyNodeABC):
    """
    LM Studio Unload Model Node

    Companion node to unload models from LM Studio memory.
    Takes model reference and unloads the model, with passthrough for text and images.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_ref": ("STRING", {
                    "tooltip": "Reference to the model name to unload from LM Studio"
                }),
                "server_url": ("STRING", {
                    "default": "http://localhost:1234",
                    "tooltip": "LM Studio server URL"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {"tooltip": "Passthrough for system prompt"}),
                "user_prompt": ("STRING", {"tooltip": "Passthrough for user prompt"}),
                "images": ("IMAGE", {"tooltip": "Passthrough for images"}),
                "generated_texts": ("STRING", {"tooltip": "Passthrough for generated texts"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt", "images", "generated_texts")
    OUTPUT_TOOLTIPS = ("Passthrough system prompt", "Passthrough user prompt", "Passthrough images", "Passthrough generated texts")
    FUNCTION = "unload_model"
    CATEGORY = "pirog/ai"
    DESCRIPTION = "Unload a model from LM Studio memory with passthrough for text and images to maintain chain connectivity."

    def unload_model(self, model_ref, server_url, system_prompt="", user_prompt="", images=None, generated_texts=""):

        try:
            if not model_ref:
                logger.warning("No model reference provided for unloading")
                return (system_prompt, user_prompt, images, generated_texts)

            # Try to unload the model using SDK
            success = self._unload_model(server_url, model_ref)
            if success:
                logger.info(f"Successfully unloaded model: {model_ref}")
            else:
                logger.warning(f"Failed to unload model: {model_ref} (SDK not available or model not found)")

            # Return passthrough values
            return (system_prompt, user_prompt, images, generated_texts)

        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            return (system_prompt, user_prompt, images, generated_texts)

    def _check_server(self, server_url):
        """Check if LM Studio server is running"""
        try:
            response = requests.get(f"{server_url}/v1/models", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _unload_model(self, server_url, model_name):
        """Unload a model from LM Studio using SDK only"""
        if not LMSTUDIO_SDK_AVAILABLE:
            logger.error(f"Cannot unload model {model_name}: LM Studio SDK not available. Install with: pip install lmstudio")
            return False

        try:
            # Extract host and port from server_url (remove http:// prefix)
            parsed_url = urlparse(server_url)
            api_host = f"{parsed_url.hostname}:{parsed_url.port}"

            # Use SDK for proper unloading - direct unload by model key (recommended approach)
            client = lmstudio.Client(api_host=api_host)
            client.llm.unload(model_name)
            logger.info(f"Successfully unloaded model via SDK: {model_name}")
            return True
        except Exception as e:
            logger.error(f"SDK unload failed for {model_name}: {e}")
            return False
