from .nodes import KSamplerMultiSeed, KSamplerMultiSeedPlus, StringCombine, Watermark, ImageScalePro, PromptRandomizer, DSLRNoise, TestResetButton, LensSimulatedBloom, CropImage, BatchCropFromMaskSimple, BatchUncropSimple, CropMaskByBBox, BlurMask, InvertMask, GradientMaskGenerator, ImageBlendByMask, BlurByMask

NODE_CLASS_MAPPINGS = {
    "KSamplerMultiSeed": KSamplerMultiSeed,
    "KSamplerMultiSeedPlus": KSamplerMultiSeedPlus,
    "StringCombine": StringCombine,
    "Watermark": Watermark,
    "ImageScalePro": ImageScalePro,
    "PromptRandomizer": PromptRandomizer,
    "DSLRNoise": DSLRNoise,
    "TestResetButton": TestResetButton,
    "LensSimulatedBloom": LensSimulatedBloom,
    "CropImage": CropImage,
    "BatchCropFromMaskSimple": BatchCropFromMaskSimple,
    "BatchUncropSimple": BatchUncropSimple,
    "CropMaskByBBox": CropMaskByBBox,
    "BlurMask": BlurMask,
    "InvertMask": InvertMask,
    "GradientMaskGenerator": GradientMaskGenerator,
    "ImageBlendByMask": ImageBlendByMask,
    "BlurByMask": BlurByMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerMultiSeed": "KSampler (Multi-Seed)",
    "KSamplerMultiSeedPlus": "KSampler (Multi-Seed+)",
    "StringCombine": "Combine strings",
    "Watermark": "Watermark",
    "ImageScalePro": "Proportional Image Scaling",
    "PromptRandomizer": "Prompt Randomizer",
    "DSLRNoise": "DSLR Camera Noise",
    "TestResetButton": "Test Reset Button",
    "LensSimulatedBloom": "Lens-simulated Bloom",
    "CropImage": "Crop Image Sides",
    "BatchCropFromMaskSimple": "Crop From Mask",
    "BatchUncropSimple": "Uncrop Image",
    "CropMaskByBBox": "Crop Mask by BBox",
    "BlurMask": "Blur Mask",
    "InvertMask": "Invert Mask",
    "GradientMaskGenerator": "Gradient Mask Generator",
    "ImageBlendByMask": "Image Blend by Mask",
    "BlurByMask": "Blur by Mask"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
