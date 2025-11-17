# Import enhanced KSampler classes from the node_modules package
from .node_modules.samplermultiseed import KSamplerMultiSeed, KSamplerMultiSeedPlus

# Import other classes from the main nodes.py file
from .nodes import StringCombine, Watermark, ImageScalePro, GetImageSize, PromptRandomizer, DSLRNoise, TestResetButton, LensSimulatedBloom, BlendImages, CropImage, BatchCropFromMaskSimple, BatchUncropSimple, CropMaskByBBox, BlurMask, InvertMask, GradientMaskGenerator, ImageBlendByMask, BlurByMask, PreviewImageQueue, LMStudioQuery, LMStudioUnloadModel, CLIPTextEncodeMultiple, CLIPTextEncodeFluxMultiple

NODE_CLASS_MAPPINGS = {
    "KSamplerMultiSeed": KSamplerMultiSeed,
    "KSamplerMultiSeedPlus": KSamplerMultiSeedPlus,
    "StringCombine": StringCombine,
    "Watermark": Watermark,
    "ImageScalePro": ImageScalePro,
    "GetImageSize": GetImageSize,
    "PromptRandomizer": PromptRandomizer,
    "DSLRNoise": DSLRNoise,
    "TestResetButton": TestResetButton,
    "LensSimulatedBloom": LensSimulatedBloom,
    "BlendImages": BlendImages,
    "CropImage": CropImage,
    "BatchCropFromMaskSimple": BatchCropFromMaskSimple,
    "BatchUncropSimple": BatchUncropSimple,
    "CropMaskByBBox": CropMaskByBBox,
    "BlurMask": BlurMask,
    "InvertMask": InvertMask,
    "GradientMaskGenerator": GradientMaskGenerator,
    "ImageBlendByMask": ImageBlendByMask,
    "BlurByMask": BlurByMask,
    "PreviewImageQueue": PreviewImageQueue,
    "LMStudioQuery": LMStudioQuery,
    "LMStudioUnloadModel": LMStudioUnloadModel,
    "CLIPTextEncodeMultiple": CLIPTextEncodeMultiple,
    "CLIPTextEncodeFluxMultiple": CLIPTextEncodeFluxMultiple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerMultiSeed": "KSampler (Multi-Seed)",
    "KSamplerMultiSeedPlus": "KSampler (Multi-Seed+)",
    "StringCombine": "Combine strings",
    "Watermark": "Watermark",
    "ImageScalePro": "Proportional Image Scaling",
    "GetImageSize": "Get Image Size",
    "PromptRandomizer": "Prompt Randomizer",
    "DSLRNoise": "DSLR Camera Noise",
    "TestResetButton": "Test Reset Button",
    "LensSimulatedBloom": "Lens-simulated Bloom",
    "BlendImages": "Blend Images",
    "CropImage": "Crop Image Sides",
    "BatchCropFromMaskSimple": "Crop From Mask",
    "BatchUncropSimple": "Uncrop Image",
    "CropMaskByBBox": "Crop Mask by BBox",
    "BlurMask": "Blur Mask",
    "InvertMask": "Invert Mask",
    "GradientMaskGenerator": "Gradient Mask Generator",
    "ImageBlendByMask": "Image Blend by Mask",
    "BlurByMask": "Blur by Mask",
    "PreviewImageQueue": "Preview Image (Queue)",
    "LMStudioQuery": "LM Studio Query",
    "LMStudioUnloadModel": "LM Studio Unload Model",
    "CLIPTextEncodeMultiple": "CLIP Text Encode (Multiple)",
    "CLIPTextEncodeFluxMultiple": "CLIP Text Encode Flux (Multiple)"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
