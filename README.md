# Pirog's Nodes for ComfyUI

A collection of image processing tools that help you craft beautiful visuals. These nodes grew from a need to add subtle realism and artistic flair to generated images - the kind of details that make pictures feel alive.

## What's Inside

### 🎲 KSampler (Multi-Seed)
**Category:** `pirog/sampling`

Generates variations by stepping through seeds automatically. Instead of manually changing seeds for each generation, this lets you explore multiple possibilities in one go. Useful when you want to see how your prompt plays out across different random states.

### 🔗 Combine strings
**Category:** `pirog/text`

Merges two text inputs with a simple toggle. Sometimes you want to combine prompts, sometimes you want just one. The toggle makes it easy to switch between modes while you're experimenting.

### 🖼️ Watermark
**Category:** `pirog/image`

Adds watermarks with different blend modes and positioning. Goes beyond basic overlay - you can soften edges, mask out black areas, and position precisely. Handy for protecting your work or adding signature touches.

### 📸 DSLR Camera Noise
**Category:** `pirog/image`

Simulates real camera sensor noise based on actual physics. Not just random grain - this recreates the specific patterns you'd see from different camera models, ISO settings, and environmental conditions. 

Perfect for making clean AI-generated images feel more authentic, or for adding that subtle imperfection that makes digital art feel organic. Each type of noise behaves differently: shot noise affects bright areas, read noise shows in shadows, thermal noise increases with temperature.

### ✨ Lens-simulated Bloom
**Category:** `pirog/image`

Creates realistic lens bloom effects through a hybrid approach. Combines soft atmospheric glow with sharp diffraction spikes - the kind you see when bright lights hit camera lenses at certain angles.

The magic happens in two stages: multi-pass downsampling creates that dreamy haze around bright areas, while FFT convolution with custom kernels adds those crisp starbursts and streaks. Progressive thresholding means you get natural gradation from general highlights to the brightest spots.

Great for adding cinematic quality to your images - that subtle glow that makes light sources feel powerful and atmosphere feel thick.

**PSF Kernel Guide:** The bloom effect needs a Point Spread Function (PSF) kernel - basically a template that defines how light spreads. Think of it as a stamp that gets applied to bright areas. Your kernel should be a black image with white/bright patterns radiating from the center. Sharp white lines create defined spikes (like in your example), while softer gradients make more subtle glows. The center should be brightest, fading toward the edges. Common patterns: cross shapes for 4-pointed stars, asterisk patterns for 6 or 8 points, or circular gradients for soft halos. Size matters - larger kernels create more dramatic effects.

### ✂️ Crop-Uncrop Pipeline
**Category:** `pirog/image`

Four interconnected nodes that streamline the crop-and-process workflow. Crop images intelligently (from masks or manual sides), work on the cropped sections, then seamlessly blend them back into the originals.

**Crop Image Sides** - Manual cropping from any side. Sometimes you just need to trim edges or focus on a specific area.

**Crop From Mask** - Automatically finds the interesting parts using masks, with optional expansion and binary thresholding. Perfect for isolating subjects or regions that need special attention. Binary mode gives you precise control over which mask values count as active.

**Uncrop Image** - Blends processed crops back into their original context with smart edge feathering. No harsh boundaries, just natural integration.

**Crop Mask by BBox** - Keeps masks aligned with their corresponding image crops. The small detail that prevents headaches later.

This pipeline eliminates the tedious manual work of calculating coordinates and managing multiple image states. Crop what matters, process it however you want, then put it back like it always belonged there.

### ⚡ Proportional Image Scaling
**Category:** `pirog/transform`

Scales images while keeping proportions intact. Includes smart alignment to common resolution steps and professional-grade algorithms. Sometimes the simple tools work best.

### 🎭 Mask Processing Suite
**Category:** `pirog/mask`

Three essential mask manipulation tools that solve common workflow bottlenecks and save you from constantly switching between different node packs.

**Blur Mask** - Intelligent blur with boundary preservation. When you blur a mask that touches image edges, standard algorithms create unwanted feathering by treating the outside as empty space. This node detects pixels near image boundaries and uses reflection-based blur to maintain crisp edges where they matter while smoothing interior transitions.

**Invert Mask** - Simple but essential. Flips mask values instantly (black becomes white, white becomes black). The kind of basic operation you need constantly but don't want to think about.

**Gradient Mask Generator** - Creates perfect linear gradients with precise control. Four directions, balance adjustment to shift the midpoint anywhere you want, contrast control to make sharp black/white transitions, and directional softening that blurs only along the gradient direction (no side bleeding). Finally, gradient generation that works exactly how you expect it to.

The blur algorithm is particularly clever - it calculates distance maps from image boundaries and blends between reflection-mode blur (near edges) and standard blur (interior) using smoothstep interpolation. No more manual edge protection or complex workarounds.

### 🖼️ Image Processing with Masks
**Category:** `pirog/image`

Two powerful nodes that handle the most common mask-based image operations. Finally, you don't need to hunt through different node packs for basic compositing operations.

**Image Blend by Mask** - Combines two images using a mask to control where and how much blending occurs. Pure PyTorch implementation with automatic device handling, smart batch processing, and automatic resizing. The classic compositing operation done right - fast, robust, and foolproof.

**Blur by Mask** - Selectively blurs an image based on mask values. Uses custom separable Gaussian convolution for maximum speed (processes all color channels simultaneously). Where the mask is white, the image gets blurred; where it's black, it stays sharp. Perfect for depth-of-field effects, selective softening, or focus isolation.

Both nodes handle dimension mismatches gracefully, work with any batch sizes, and include mask inversion options. Built for speed with memory-efficient PyTorch operations - no external dependencies, no performance bottlenecks.

### 🎰 Prompt Randomizer
**Category:** `pirog/text`

Randomizes parts of your prompts using pattern replacement. Write `?color? ?animal? in ?background?` and it fills in the blanks from customizable dictionaries. Useful for generating variations or discovering unexpected combinations.

## Why These Tools Matter

This isn't just another random collection of nodes - it's about gathering all the missing pieces that make image generation workflows actually work smoothly. 

Instead of hunting through dozens of node packs for basic operations like "blur a mask without edge artifacts" or "make a proper gradient," everything you need is right here. No more installing three different extensions just to crop, process, and blend an image back together.

The realism nodes (DSLR noise, lens bloom) aren't just aesthetic - they're scientifically modeled on how actual cameras behave. Real sensors have quirks that our eyes recognize as authentic. The processing nodes solve the tedious technical problems that eat up your time.

This saves hours of workflow assembly. You can focus on the creative work instead of constantly switching between node packs or working around their limitations. Everything works together, nothing breaks your flow.

## Technical Notes

Built with clean architecture and proper error handling. Each node does one thing well. Import consolidation and modular design make everything stable and maintainable.

**Reset Button Magic:** Every numeric parameter gets a reset button that instantly restores the default value. No more manually typing in defaults or remembering what they were. Just click the little button on the right side of any slider or input field.

### File Structure
```
Pirogs-Nodes/
├── __init__.py              # Node mappings
├── nodes.py                 # All implementations
├── README.md                # This file
├── prompt_dictionary.json   # For randomizer
└── pyproject.toml           # Package config
```

### Dependencies
Most nodes work with standard ComfyUI. The Lens-simulated Bloom requires NumPy, OpenCV, and SciPy for the advanced filtering operations.

### Prompt Dictionary
The randomizer creates a JSON file with common categories: colors, animals, styles, lighting, moods. You can edit it to add your own categories and terms.

Example patterns:
- `?color? ?animal? in a ?background?` → "blue cat in a forest"
- `A ?quality? ?style? portrait` → "A masterpiece photorealistic portrait"

## Installation

Drop the folder into your ComfyUI custom_nodes directory and restart. That's it.

---

*Made for people who care about the small details that make big differences.*