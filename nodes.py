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

# Import our advanced noise generation library
# from .noise_generator import create_spectral_diverse_noise, create_hierarchical_noise

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

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Scaled image(s) with proper aspect ratio preservation",)
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
                pil_image = self.tensor_to_pil(image[i])
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


def srgb_to_linear(img):
    """Converts sRGB image data to linear RGB."""
    mask = img <= 0.04045
    return np.where(mask, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    """Converts linear RGB image data to sRGB."""
    mask = img <= 0.0031308
    srgb = np.where(mask, img * 12.92, 1.055 * (img ** (1.0 / 2.4)) - 0.055)
    return np.clip(srgb, 0.0, 1.0)

def apply_noise_blur(noise, blur_sigma):
    """Applies a Gaussian blur to the generated noise field."""
    if blur_sigma > 0:
        for c in range(noise.shape[2]):
            noise[:, :, c] = gaussian_filter(noise[:, :, c], sigma=blur_sigma)
    return noise


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
            shot_noise_e = np.random.poisson(np.maximum(scaled_signal_for_noise_calc, 0)) - scaled_signal_for_noise_calc

            # 3. Thermal Noise (Dark Current & DSNU) (Pre-gain)
            # This part correctly uses exposure_time already, so no changes are needed.
            temp_factor = 2.0 ** ((temperature - 20.0) / 7.0)
            dark_current_e = params["dark_current_base"] * temp_factor * exposure_time
            dsnu_pattern = np.random.normal(1.0, params["dsnu_factor"], current_signal_e.shape)
            dark_current_with_dsnu = dark_current_e * dsnu_pattern
            thermal_noise_e = np.random.poisson(np.maximum(dark_current_with_dsnu, 0)) - dark_current_with_dsnu

            # 4. ISO Gain and Read Noise (Post-gain)
            gain = max(iso / 100.0, 1.0)
            read_noise_std = params["read_noise_base_e"] + (gain - 1.0) * params["read_noise_iso_scaling"]

            base_luma_noise = np.random.normal(0, read_noise_std, (height, width, 1))

            chroma_h, chroma_w = height // 2, width // 2
            lowres_chroma_noise = np.random.normal(0, read_noise_std, (chroma_h, chroma_w, channels))
            chroma_noise = np.repeat(np.repeat(lowres_chroma_noise, 2, axis=0), 2, axis=1)
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
                "image": ("IMAGE", {
                    "tooltip": "Input image to apply lens-simulated bloom effect"
                }),
                "psf_kernel": ("IMAGE", {
                    "tooltip": "Point Spread Function kernel defining diffraction spike shape (grayscale recommended)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Base brightness threshold (0.0-1.0). With progressive_threshold enabled, this is the starting threshold that increases each pass up to 1.0"
                }),
                "haze_passes": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "tooltip": "Number of downsampling/blurring passes for soft haze effect"
                }),
                "haze_spread": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Gaussian blur sigma for haze passes - controls haze softness"
                }),
                "haze_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Multiplier for soft haze brightness (normalized, safe range 0.1-3.0)"
                }),
                "spike_intensity": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Multiplier for structured spike brightness (normalized, safe range 0.1-5.0)"
                }),
                "progressive_threshold": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable progressive threshold: each haze pass uses higher threshold for finer detail isolation"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Image with lens-simulated bloom effect applied",)
    FUNCTION = "apply_bloom"
    CATEGORY = "pirog/image"
    DESCRIPTION = "Applies realistic lens bloom using hybrid approach: multi-pass blur for soft haze + FFT convolution for sharp spikes. Requires NumPy, OpenCV, and SciPy."

    def apply_bloom(self, image, psf_kernel, threshold, haze_passes, haze_spread, haze_intensity, spike_intensity, progressive_threshold=True):
        """Apply the hybrid bloom effect to input images"""
        try:
            # Import required libraries
            import cv2
            
            # Convert ComfyUI tensors to numpy arrays
            img_np = self.tensor_to_numpy(image)
            kernel_np = self.tensor_to_numpy(psf_kernel)
            
            # Process each image in the batch
            batch_size = img_np.shape[0]
            output_images = []
            
            for i in range(batch_size):
                # Get single image and convert to proper format
                single_img = img_np[i]
                
                # Use first kernel image or cycle through if multiple kernels
                kernel_idx = min(i, kernel_np.shape[0] - 1)
                single_kernel = kernel_np[kernel_idx]
                
                # Convert kernel to grayscale if it's color
                if len(single_kernel.shape) == 3:
                    single_kernel = cv2.cvtColor(single_kernel, cv2.COLOR_RGB2GRAY)
                
                # Apply the hybrid bloom effect
                bloom_result = self.apply_hybrid_bloom(
                    single_img,
                    single_kernel,
                    threshold,
                    haze_passes,
                    haze_spread,
                    haze_intensity,
                    spike_intensity,
                    progressive_threshold
                )
                
                output_images.append(bloom_result)
            
            # Stack results and convert back to tensor
            result_array = np.stack(output_images, axis=0)
            result_tensor = self.numpy_to_tensor(result_array)
            
            return (result_tensor,)
            
        except ImportError as e:
            raise RuntimeError(f"Missing required library for Lens-simulated Bloom: {e}. Please install numpy, opencv-python, and scipy.")
        except Exception as e:
            raise RuntimeError(f"Error applying lens bloom: {e}")

    def apply_hybrid_bloom(self, image, psf_kernel, threshold=0.85, haze_passes=6, 
                          haze_spread=2.0, haze_intensity=1.0, spike_intensity=1.5, progressive_threshold=True):
        """
        Applies a high-quality, hybrid bloom effect to an input image.

        This method combines two techniques for a visually superior result:
        1.  A multi-pass, downsampling blur (the "Haze") for a soft, atmospheric glow.
            Progressive threshold mode creates realistic gradation from general to specific highlights
        2.  An FFT convolution with a custom kernel (the "Spikes") for sharp,
            structured diffraction artifacts like starbursts or anamorphic streaks.
            
        Args:
            progressive_threshold (bool): When True, each haze pass uses progressively higher 
                threshold values (e.g., 0.2 → 0.36 → 0.52 → ... → 1.0), creating more 
                realistic bloom with fine gradation. When False, uses original single-threshold method.
        """
        import numpy as np
        import cv2
        from scipy.signal import fftconvolve

        # --- 1. PRE-PROCESSING AND BRIGHTNESS EXTRACTION ---

        # Ensure image is in a floating-point format (0.0 to 1.0) for calculations
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.copy()

        # Create a brightness map. For color images, convert to grayscale.
        # The brightness map determines which pixels are bright enough to "bloom".
        if len(image_float.shape) == 3:
            brightness_map = cv2.cvtColor(image_float, cv2.COLOR_RGB2GRAY)
        else:
            brightness_map = image_float

        # --- 2. HAZE GENERATION (PROGRESSIVE THRESHOLD MULTI-PASS BLUR) ---

        if progressive_threshold:
            # Progressive threshold mode: each pass uses increasingly higher threshold
            # This creates more realistic bloom with fine gradation from general to specific highlights
            
            # Calculate threshold progression from base threshold to 1.0
            if haze_passes > 1:
                threshold_step = (1.0 - threshold) / (haze_passes - 1)
                thresholds = [threshold + i * threshold_step for i in range(haze_passes)]
            else:
                thresholds = [threshold]
            
            # Create haze layers with progressive thresholds
            haze_layers = []
            for pass_idx in range(haze_passes):
                current_threshold = thresholds[pass_idx]
                
                # Threshold the brightness map for this pass
                _, brights_mask = cv2.threshold(brightness_map, current_threshold, 1.0, cv2.THRESH_BINARY)
                
                # Create bright pixels for this threshold level
                if len(image_float.shape) == 3:
                    brights_mask_3ch = cv2.cvtColor(brights_mask, cv2.COLOR_GRAY2RGB)
                    bright_pixels_pass = image_float * brights_mask_3ch
                else:
                    bright_pixels_pass = brights_mask
                
                # Downsample for pyramid effect
                current_layer = bright_pixels_pass
                for _ in range(pass_idx):
                    current_layer = cv2.pyrDown(current_layer)
                
                haze_layers.append(current_layer)
            
            # Blur each threshold layer with appropriate scaling
            blurred_layers = [cv2.GaussianBlur(layer, (0, 0), haze_spread) for layer in haze_layers]
            
            # Composite progressive threshold layers
            haze_composite = None
            for i, layer in enumerate(blurred_layers):
                # Upsample to original size using safer approach
                upsampled = layer
                
                # Instead of forcing pyrUp to exact dimensions, use step-by-step upsampling
                # then resize to exact dimensions at the end
                for _ in range(i):
                    # Use pyrUp without forcing exact dimensions
                    upsampled = cv2.pyrUp(upsampled)
                
                # Now safely resize to exact original dimensions
                if upsampled.shape[:2] != image_float.shape[:2]:
                    upsampled = cv2.resize(upsampled, (image_float.shape[1], image_float.shape[0]))
                
                if haze_composite is None:
                    haze_composite = upsampled
                else:
                    haze_composite += upsampled
            
        else:
            # Original single-threshold mode for compatibility
            _, brights_mask = cv2.threshold(brightness_map, threshold, 1.0, cv2.THRESH_BINARY)
            
            if len(image_float.shape) == 3:
                brights_mask_3ch = cv2.cvtColor(brights_mask, cv2.COLOR_GRAY2RGB)
                bright_pixels = image_float * brights_mask_3ch
            else:
                bright_pixels = brights_mask
            
            haze_layers = [bright_pixels]
            current_layer = bright_pixels
            
            # Create a pyramid of downsampled images
            for _ in range(haze_passes - 1):
                current_layer = cv2.pyrDown(current_layer)
                haze_layers.append(current_layer)
            
            # Blur each layer in the pyramid
            blurred_layers = [cv2.GaussianBlur(layer, (0, 0), haze_spread) for layer in haze_layers]
            
            # Composite the blurred layers back together
            haze_composite = blurred_layers[-1]
            for i in range(haze_passes - 2, -1, -1):
                # Use pyrUp without forcing exact dimensions, then resize if needed
                haze_composite = cv2.pyrUp(haze_composite)
                
                # Ensure dimensions match the target layer
                if haze_composite.shape[:2] != blurred_layers[i].shape[:2]:
                    haze_composite = cv2.resize(haze_composite, (blurred_layers[i].shape[1], blurred_layers[i].shape[0]))
                
                haze_composite += blurred_layers[i]
        
        # Normalize by the number of layers to prevent brightness accumulation
        # This ensures haze_intensity behaves predictably regardless of haze_passes
        haze_composite = haze_composite / haze_passes

        # For spike generation, use the original threshold method to maintain compatibility
        _, brights_mask = cv2.threshold(brightness_map, threshold, 1.0, cv2.THRESH_BINARY)
        if len(image_float.shape) == 3:
            brights_mask_3ch = cv2.cvtColor(brights_mask, cv2.COLOR_GRAY2RGB)
            bright_pixels = image_float * brights_mask_3ch
        else:
            bright_pixels = brights_mask

        # --- 3. SPIKE GENERATION (FFT CONVOLUTION) ---

        # Normalize the PSF kernel to prevent value explosion during convolution
        # This ensures the convolution output stays in a reasonable range
        psf_normalized = psf_kernel / np.sum(psf_kernel) if np.sum(psf_kernel) > 0 else psf_kernel

        # Convolve the bright pixels with the custom PSF kernel using FFT for efficiency
        # and accuracy. This creates the structured light spikes.
        if len(image_float.shape) == 3:
            spike_layer = np.zeros_like(image_float)
            # Perform convolution on each color channel separately
            for i in range(3):
                spike_layer[:, :, i] = fftconvolve(bright_pixels[:, :, i], psf_normalized, mode='same')
        else:
            spike_layer = fftconvolve(bright_pixels, psf_normalized, mode='same')

        # --- 4. FINAL COMPOSITION ---

        # Combine the original image, the soft haze, and the sharp spikes.
        # The intensities are used to control the final look.
        bloom_image = (
            image_float +
            (haze_composite * haze_intensity) +
            (spike_layer * spike_intensity)
        )

        # Clip the result to the valid [0.0, 1.0] range to prevent wrapping artifacts
        bloom_image = np.clip(bloom_image, 0.0, 1.0)

        # Convert back to uint8 format for standard image display/saving
        return (bloom_image * 255).astype(np.uint8)

    def tensor_to_numpy(self, tensor):
        """Convert ComfyUI image tensor to numpy array"""
        # ComfyUI tensors are in format [batch, height, width, channels]
        # Convert to uint8 if in float format
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            # Clamp to [0,1] and convert to [0,255]
            numpy_array = (torch.clamp(tensor, 0, 1) * 255).byte().cpu().numpy()
        else:
            numpy_array = tensor.cpu().numpy()
        
        return numpy_array

    def numpy_to_tensor(self, numpy_array):
        """Convert numpy array to ComfyUI image tensor"""
        # Ensure array is in uint8 format
        if numpy_array.dtype != np.uint8:
            numpy_array = np.clip(numpy_array, 0, 255).astype(np.uint8)
        
        # Convert to float32 and normalize to [0,1]
        tensor = torch.from_numpy(numpy_array.astype(np.float32) / 255.0)
        
        return tensor


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
    