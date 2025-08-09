# Pirog's Nodes for ComfyUI

A comprehensive custom node pack for ComfyUI providing enhanced sampling, image processing, text manipulation, and utility nodes with professional architecture and modern UI design.

## 📦 Current Nodes

### 🎲 KSampler (Multi-Seed)
**Category:** `pirog/sampling`  
**Function:** Generates multiple images by incrementing the seed for each generation across all latents in the input batch.

**Features:**
- Multi-seed batch processing with progress tracking
- Automatic seed incrementation 
- Full compatibility with ComfyUI's latest sampling methods
- Uses the official `common_ksampler` function for maximum compatibility

### 🔗 Combine strings
**Category:** `pirog/text`  
**Function:** Concatenates two multiline text inputs based on a boolean toggle with fancy UI styling.

**Features:**
- Two multiline text input fields
- Sleek boolean toggle with smooth animations (following KJ/Essentials design patterns)
- Logic: `enabled = text1 + text2`, `disabled = text1 only`

### 🖼️ Watermark
**Category:** `pirog/image`  
**Function:** Advanced watermark application with multiple blend modes and positioning options.

**Features:**
- 11 blend modes (normal, overlay, soft_light, hard_light, difference, etc.)
- 9 positioning options (corners, edges, center, random)
- Precise opacity control (0.001-1.0 range)
- Scale control with fine precision
- Black pixel masking and watermark inversion
- Batch processing support

### 📸 DSLR Camera Noise
**Category:** `pirog/image`  
**Function:** Scientifically accurate DSLR camera sensor noise simulation based on real sensor physics and measurements.

**Features:**
- **🔬 Research-based noise model** implementing Poisson-Gaussian sensor physics from real DSLR studies
- **🎛️ Multiple noise components**: Shot noise, read noise, thermal noise, pattern noise, PRNU (pixel variation)
- **📷 Camera sensor presets**: Modern full-frame, APS-C, older sensors, high-end models with real parameters
- **🌡️ Environmental controls**: Temperature and exposure time effects with exponential thermal modeling
- **🌈 Advanced color handling**: Adjustable color vs luminance noise correlation for realistic behavior
- **⚙️ Individual component control**: Fine-tune each noise type independently with detailed tooltips
- **🎨 Proper color space handling**: Works in linear sensor space with accurate sRGB conversion (no color desaturation!)
- **⚡ Batch processing**: Efficient processing of multiple images with NumPy optimization
- **🎲 Reproducible results**: Seed-based random generation for consistent outputs
- **🔄 Reset functionality**: One-click reset to restore all parameters to scientifically accurate defaults
- **🌫️ HQ Noise Blur**: High-quality Gaussian blur applied to noise patterns (not final image) for realistic sensor behavior
- **📚 Comprehensive tooltips**: Each parameter includes detailed explanations, ranges, and real-world examples

**Quick Settings Guide:**
```
📊 ISO Settings:
• 100-400: Clean (daylight photography)
• 800-1600: Moderate (indoor/evening)
• 3200-6400: High noise (low light)
• 12800+: Very noisy (extreme conditions)

📷 Camera Models:
• Modern Full-Frame: Cleanest (Canon R5, Sony A7R)
• Modern APS-C: Good performance (Fuji X-T5)
• Older Sensor: Vintage look (Canon 20D era)
• High-End: Professional (Phase One, Canon R3)

🎛️ Advanced Controls:
• Shot Noise: Main noise component (Poisson statistics)
• Read Noise: Electronic noise (visible in shadows)
• Thermal Noise: Heat effects (long exposures)
• Pattern Noise: Banding (older cameras only)
• PRNU: Pixel variation (high signal areas)
• Color Ratio: 0.25 = realistic, 1.0 = full color noise
• Noise Blur: 0.0 = sharp noise, 0.3+ = realistic noise softening (blurs noise only!)
```

### ⚡ Proportional Image Scaling
**Category:** `pirog/transform`  
**Function:** Professional image scaling with aspect ratio preservation and advanced controls.

**Features:**
- MX-style slider compatibility for precise control
- Resolution step alignment (8px, 16px, etc.)
- Smart resolution limits (min/max constraints)
- Professional-grade scaling algorithms
- Batch processing with progress tracking

### 🎰 Prompt Randomizer
**Category:** `pirog/text`  
**Function:** Advanced prompt randomization using pattern replacement with dictionary support.

**Features:**
- Pattern-based randomization: `?color? ?animal? in ?background?`
- Automatic dictionary creation with 8 default categories
- Relative path handling (dictionary stored in node pack folder)
- Configurable text cleaning and newline handling
- Seed control for reproducible results
- Fallback to dummy dictionary if file missing

## 🏗️ Architecture & Structure

This node pack follows professional ComfyUI custom node conventions with clear separation of concerns:

```
Pirogs-Nodes/
├── __init__.py              # Node mappings and exports only
├── nodes.py                 # All node implementations
├── README.md                # This file
├── prompt_dictionary.json   # Dictionary for Prompt Randomizer (auto-created)
└── pyproject.toml           # Package configuration
```

### 📄 File Structure Explained

#### `__init__.py` - Clean Mappings Only
```python
from .nodes import KSamplerMultiSeed, StringCombine, Watermark, ImageScalePro, PromptRandomizer

NODE_CLASS_MAPPINGS = {
    "KSamplerMultiSeed": KSamplerMultiSeed,
    "StringCombine": StringCombine,
    "Watermark": Watermark,
    "ImageScalePro": ImageScalePro,
    "PromptRandomizer": PromptRandomizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerMultiSeed": "KSampler (Multi-Seed)",
    "StringCombine": "Combine strings", 
    "Watermark": "Watermark",
    "ImageScalePro": "Proportional Image Scaling",
    "PromptRandomizer": "Prompt Randomizer"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

#### `nodes.py` - All Implementations
Contains all node class definitions with proper import consolidation:

```python
# All imports consolidated at the top for best practices
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

# ComfyUI specific imports
import comfy.samplers
import comfy.utils
from nodes import common_ksampler

# Node class definitions follow...
```

**Key Architecture Principles:**
- ✅ **Consolidated imports** at module level (no local imports in methods)
- ✅ **Single responsibility** - each file has one clear purpose
- ✅ **Professional error handling** with logging and fallbacks
- ✅ **Consistent naming** with `pirog/*` categories

## 🔧 Adding New Nodes

### 1. **Basic Node Structure**

```python
class YourNewNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_name": ("TYPE", {
                    "default": default_value,
                    "min": min_val,           # For numbers
                    "max": max_val,           # For numbers  
                    "step": step_val,         # For numbers
                    "multiline": True,        # For strings
                    "tooltip": "Description"  # Helpful tooltip
                }),
            },
            "optional": {
                "optional_input": ("TYPE",),
            }
        }

    RETURN_TYPES = ("OUTPUT_TYPE",)
    OUTPUT_TOOLTIPS = ("Description of output",)
    FUNCTION = "your_function_name"
    CATEGORY = "pirog/category"
    DESCRIPTION = "What this node does"

    def your_function_name(self, input_name, optional_input=None):
        # Your logic here
        result = process(input_name)
        return (result,)
```

### 2. **Input Types Reference**

| Type | Usage | Options |
|------|-------|---------|
| `"STRING"` | Text input | `{"multiline": True/False, "default": "text"}` |
| `"INT"` | Integer | `{"default": 0, "min": 0, "max": 100, "step": 1}` |
| `"FLOAT"` | Decimal | `{"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}` |
| `"BOOLEAN"` | Toggle | `{"default": True}` - Auto-renders as fancy toggle |
| `"IMAGE"` | Image tensor | No options needed |
| `"LATENT"` | Latent space | No options needed |
| `("ENUM", ["opt1", "opt2"])` | Dropdown | List of string options |

### 3. **Adding Fancy UI Elements**

#### **Fancy Boolean Toggles**
```python
"toggle_name": ("BOOLEAN", {
    "default": True, 
    "tooltip": "Enable/disable feature"
})
```
ComfyUI automatically renders this as a sleek toggle button with:
- Smooth slide animation
- Color change (green=enabled, gray=disabled)
- Visual state indication

#### **Multiline Text Areas**
```python
"text_input": ("STRING", {
    "multiline": True,
    "default": "Enter text here...",
    "tooltip": "Your text description"
})
```

### 4. **Advanced UI: MX Toolkit Style Sliders**

For **ultra-fancy sliders** like in the MX Toolkit (the image you showed), you need:

#### **Frontend JavaScript Extension** (Advanced)
1. Create `web/js/` directory in your node pack
2. Add custom JavaScript for widget rendering
3. Reference in `__init__.py`: `WEB_DIRECTORY = "./js"`

**Example MX-style slider structure:**
```python
# In nodes.py
class FancySliderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_int": ("INT", {"default": 20, "min": 0, "max": 100}),
                "value_float": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0}),
                "is_float": ("INT", {"default": 0, "min": 0, "max": 1}),  # Float mode toggle
            }
        }
```

```javascript
// In web/js/fancy_slider.js
import { app } from "../../scripts/app.js";

class FancySlider {
    constructor(node) {
        // Custom rendering logic here
        // Override onDrawForeground for custom draw
        // Handle mouse events for interaction
    }
}

app.registerExtension({
    name: "pirog.FancySlider",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FancySliderNode") {
            // Register custom widget behavior
        }
    }
});
```

### 5. **Update Node Mappings**

After creating your node, update `__init__.py`:

```python
# 1. Add import
from .nodes import KSamplerMultiSeed, StringCombine, Watermark, ImageScalePro, PromptRandomizer, YourNewNode

# 2. Add to class mapping
NODE_CLASS_MAPPINGS = {
    "KSamplerMultiSeed": KSamplerMultiSeed,
    "StringCombine": StringCombine,
    "Watermark": Watermark,
    "ImageScalePro": ImageScalePro,
    "PromptRandomizer": PromptRandomizer,
    "YourNewNode": YourNewNode,  # Add here
}

# 3. Add to display mapping
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerMultiSeed": "KSampler (Multi-Seed)",
    "StringCombine": "Combine strings",
    "Watermark": "Watermark", 
    "ImageScalePro": "Proportional Image Scaling",
    "PromptRandomizer": "Prompt Randomizer",
    "YourNewNode": "Your Display Name",  # Add here
}
```

## 🎨 UI Design Principles

### **Standard Widgets** 
ComfyUI provides beautiful built-in widgets:
- **Boolean**: Auto-renders as smooth toggle switches
- **Sliders**: Clean, responsive number inputs
- **Dropdowns**: Elegant selection menus
- **Text Areas**: Resizable multiline inputs

### **Custom Styling** (Advanced)
For ultra-custom UI like MX sliders:
1. **JavaScript Extensions**: Custom widget rendering
2. **CSS Styling**: Custom appearance
3. **Mouse Interaction**: Custom behavior handling
4. **Canvas Drawing**: Direct graphics rendering

### **Design Inspiration Sources**
- **KJ Nodes**: Professional boolean toggles and dropdowns
- **Essentials Pack**: Clean multiline text areas
- **MX Toolkit**: Ultra-fancy custom sliders with gradients

## 🚀 Best Practices

### **Import Management** ⭐ CRITICAL
- ✅ **Consolidate ALL imports** at the top of `nodes.py` 
- ❌ **NEVER use local imports** inside methods/functions
- ✅ **Import once, use everywhere** - better performance and maintainability
- ✅ **Group imports logically**: Standard library → Third party → ComfyUI

**Example of proper import structure:**
```python
# Standard library
import os, sys, json, random, re, logging

# Third party 
import torch, numpy as np
from PIL import Image, ImageChops, ImageOps

# ComfyUI specific
import comfy.samplers, comfy.utils
from nodes import common_ksampler
```

**Common mistakes to avoid:**
```python
# ❌ DON'T DO THIS - Local imports in methods
def some_method(self):
    import numpy as np  # WRONG!
    import re          # WRONG!
    # ...

# ✅ DO THIS - All imports at top
import numpy as np
import re

def some_method(self):
    # Direct usage, no imports needed
    result = np.array(data)
```

### **Code Quality**
- ✅ Use descriptive variable names
- ✅ Add comprehensive tooltips
- ✅ Include docstrings for complex nodes
- ✅ Handle edge cases gracefully
- ✅ Follow ComfyUI typing conventions

### **User Experience**
- ✅ Provide sensible defaults
- ✅ Use appropriate input ranges
- ✅ Add helpful descriptions
- ✅ Group related functionality
- ✅ Consider workflow context

### **Compatibility**
- ✅ Use ComfyUI's built-in functions when possible
- ✅ Import from official modules
- ✅ Test with different input types
- ✅ Handle batch processing correctly
- ✅ Follow semantic versioning

### **Performance**
- ✅ Use progress bars for long operations
- ✅ Batch process when possible
- ✅ Minimize memory allocation
- ✅ Cache expensive computations
- ✅ Handle interruption gracefully

## 📝 Prompt Randomizer Dictionary

The **Prompt Randomizer** node automatically creates a `prompt_dictionary.json` file in the Pirogs-Nodes folder. You can customize it with your own categories:

### **Default Categories:**
- `color`: Basic colors (red, blue, green, etc.)
- `animal`: Common animals (cat, dog, bird, etc.)  
- `style`: Art styles (photorealistic, anime, etc.)
- `lighting`: Lighting types (soft, dramatic, natural, etc.)
- `mood`: Emotional tones (happy, serene, mysterious, etc.)
- `quality`: Quality descriptors (high quality, masterpiece, etc.)
- `background`: Scene locations (forest, city, beach, etc.)
- `weather`: Weather conditions (sunny, cloudy, rainy, etc.)

### **Usage Examples:**
```
?color? ?animal? in a ?background? with ?lighting?
→ "red cat in a forest with soft lighting"

A ?quality? ?style? portrait with ?mood? atmosphere
→ "A masterpiece photorealistic portrait with serene atmosphere"
```

### **Custom Categories:**
Add your own categories to the JSON file:
```json
{
  "pose": ["standing", "sitting", "running", "dancing"],
  "clothing": ["dress", "suit", "casual wear", "armor"],
  "expression": ["smiling", "serious", "surprised", "contemplative"]
}
```

## 🔗 References

- **[ComfyUI Custom Nodes](https://github.com/comfyanonymous/ComfyUI)**: Official documentation
- **[KJ Nodes](https://github.com/kijai/ComfyUI-KJNodes)**: Professional node structure
- **[Essentials Pack](https://github.com/cubiq/ComfyUI_essentials)**: Clean UI patterns
- **[MX Toolkit](https://github.com/maxmaxjian/ComfyUI-mxToolkit)**: Advanced custom widgets

---

*Built with ❤️ for the ComfyUI community*