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

### ⚡ Proportional Image Scaling
**Category:** `pirog/transform`

Scales images while keeping proportions intact. Includes smart alignment to common resolution steps and professional-grade algorithms. Sometimes the simple tools work best.

### 🎰 Prompt Randomizer
**Category:** `pirog/text`

Randomizes parts of your prompts using pattern replacement. Write `?color? ?animal? in ?background?` and it fills in the blanks from customizable dictionaries. Useful for generating variations or discovering unexpected combinations.

## Why These Tools Matter

Digital art often feels too perfect, too clean. Real cameras have quirks - noise patterns, lens flares, subtle imperfections that our eyes recognize as authentic. These nodes let you add back that humanity.

The DSLR noise isn't just aesthetic - it's scientifically modeled on how actual sensors behave. The bloom effect mimics real optical physics. Even the watermarking respects how light actually blends.

This isn't about making things look worse - it's about making them feel real. That slight grain that makes skin look alive. The soft glow that makes light sources feel powerful. The imperfections that make perfection possible.

## Technical Notes

Built with clean architecture and proper error handling. Each node does one thing well. Import consolidation and modular design make everything stable and maintainable.

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