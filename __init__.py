from .nodes import KSamplerMultiSeed, KSamplerMultiSeedPlus, StringCombine, Watermark, ImageScalePro, PromptRandomizer, DSLRNoise, TestResetButton

NODE_CLASS_MAPPINGS = {
    "KSamplerMultiSeed": KSamplerMultiSeed,
    "KSamplerMultiSeedPlus": KSamplerMultiSeedPlus,
    "StringCombine": StringCombine,
    "Watermark": Watermark,
    "ImageScalePro": ImageScalePro,
    "PromptRandomizer": PromptRandomizer,
    "DSLRNoise": DSLRNoise,
    "TestResetButton": TestResetButton
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerMultiSeed": "KSampler (Multi-Seed)",
    "KSamplerMultiSeedPlus": "KSampler (Multi-Seed+)",
    "StringCombine": "Combine strings",
    "Watermark": "Watermark",
    "ImageScalePro": "Proportional Image Scaling",
    "PromptRandomizer": "Prompt Randomizer",
    "DSLRNoise": "DSLR Camera Noise",
    "TestResetButton": "Test Reset Button"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
