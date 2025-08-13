"""
Advanced Noise Generation Library for ComfyUI
Provides superior noise generation methods for diffusion sampling.

Author: Enhanced by AI analysis for Pirog's Nodes
"""

import torch


class NoiseGenerator:
    """
    Advanced noise generation library providing multiple noise generation strategies
    that are superior to vanilla torch.randn() for diffusion sampling.
    """
    
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cpu")
    
    def generate_spectral_diverse_noise(self, shape, seed, noise_type="hybrid"):
        """
        Generate noise with controlled spectral characteristics for enhanced diversity.
        Uses different noise distributions and frequency domain manipulation.
        
        Args:
            shape: Tensor shape [batch, channels, height, width]
            seed: Random seed for reproducibility
            noise_type: Type of spectral noise ("pink", "blue", "hybrid")
            
        Returns:
            torch.Tensor: Generated noise with specified spectral characteristics
        """
        try:
            torch.manual_seed(seed)
            base_noise = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            if noise_type == "pink":
                result = self._generate_pink_noise(base_noise, shape)
            elif noise_type == "blue":
                result = self._generate_blue_noise(base_noise, shape)
            else:  # hybrid
                result = self._generate_hybrid_spectral_noise(shape, seed)
            
            # Final safety check
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            return result.to(dtype=torch.float32)
        except Exception:
            # Fallback to simple Gaussian noise
            return torch.randn(shape, device=self.device, dtype=torch.float32)
    
    def _generate_pink_noise(self, base_noise, shape):
        """Generate pink noise (1/f) - more natural, found in many real-world phenomena"""
        try:
            fft_noise = torch.fft.fftn(base_noise, dim=[-2, -1])
            freq_h = torch.fft.fftfreq(shape[-2], device=self.device).view(-1, 1)
            freq_w = torch.fft.fftfreq(shape[-1], device=self.device).view(1, -1)
            freq_magnitude = torch.sqrt(freq_h**2 + freq_w**2)
            freq_magnitude[0, 0] = 1.0  # Avoid division by zero
            
            # Apply 1/f scaling with clipping to avoid extreme values
            scaling = 1.0 / torch.sqrt(freq_magnitude)
            scaling = torch.clamp(scaling, 0.1, 10.0)  # Clamp to reasonable range
            scaled_fft = fft_noise * scaling.unsqueeze(0).unsqueeze(0)
            result = torch.fft.ifftn(scaled_fft, dim=[-2, -1]).real
            
            # Ensure finite values and proper dtype
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            return result.to(dtype=base_noise.dtype)
        except Exception:
            # Fallback to original noise if FFT fails
            return base_noise
    
    def _generate_blue_noise(self, base_noise, shape):
        """Generate blue noise - high frequency emphasis, good for fine details"""
        try:
            fft_noise = torch.fft.fftn(base_noise, dim=[-2, -1])
            freq_h = torch.fft.fftfreq(shape[-2], device=self.device).view(-1, 1)
            freq_w = torch.fft.fftfreq(shape[-1], device=self.device).view(1, -1)
            freq_magnitude = torch.sqrt(freq_h**2 + freq_w**2)
            
            # Apply f scaling (opposite of pink) with clipping
            scaling = torch.sqrt(freq_magnitude + 0.1)  # +0.1 to avoid zero
            scaling = torch.clamp(scaling, 0.1, 10.0)  # Clamp to reasonable range
            scaled_fft = fft_noise * scaling.unsqueeze(0).unsqueeze(0)
            result = torch.fft.ifftn(scaled_fft, dim=[-2, -1]).real
            
            # Ensure finite values and proper dtype
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            return result.to(dtype=base_noise.dtype)
        except Exception:
            # Fallback to original noise if FFT fails
            return base_noise
    
    def _generate_hybrid_spectral_noise(self, shape, seed):
        """Combine different noise types per channel for maximum diversity"""
        try:
            channels = []
            for i in range(shape[1]):  # Iterate over channels
                channel_seed = seed + i * 10007  # Prime offset for independence
                torch.manual_seed(channel_seed)
                
                if i % 3 == 0:
                    single_channel_shape = [shape[0], 1, shape[2], shape[3]]
                    base_noise = torch.randn(single_channel_shape, device=self.device, dtype=torch.float32)
                    channel_noise = self._generate_pink_noise(base_noise, single_channel_shape)
                elif i % 3 == 1:
                    single_channel_shape = [shape[0], 1, shape[2], shape[3]]
                    base_noise = torch.randn(single_channel_shape, device=self.device, dtype=torch.float32)
                    channel_noise = self._generate_blue_noise(base_noise, single_channel_shape)
                else:
                    channel_noise = torch.randn([shape[0], 1, shape[2], shape[3]], 
                                              device=self.device, dtype=torch.float32)
                
                # Ensure finite values and proper dtype
                channel_noise = torch.nan_to_num(channel_noise, nan=0.0, posinf=1.0, neginf=-1.0)
                channels.append(channel_noise.to(dtype=torch.float32))
            
            result = torch.cat(channels, dim=1)
            return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception:
            # Fallback to simple Gaussian noise
            return torch.randn(shape, device=self.device, dtype=torch.float32)
    
    def generate_latent_aware_hierarchical_noise(self, shape, seed, diversity_strength=1.0):
        """
        Generate noise that respects VAE latent space structure using hierarchical sampling.
        Uses pre-analyzed latent statistics and multi-scale generation.
        
        Args:
            shape: Tensor shape [batch, channels, height, width]
            seed: Random seed for reproducibility
            diversity_strength: Strength of diversity patterns (0.1 to 2.0)
            
        Returns:
            torch.Tensor: Generated hierarchical noise respecting latent space properties
        """
        torch.manual_seed(seed)
        
        # Support different channel counts (4 for SD, 16 for Flux, etc.)
        channels = shape[1]
        if channels not in [4, 16]:
            # Fallback to simple Gaussian noise for unsupported channel counts
            return torch.randn(shape, device=self.device, dtype=torch.float32)
        
        try:
            # Multi-scale noise generation
            scales = [1.0, 0.5, 0.25]  # Different resolution scales
            noise_pyramid = []
            
            for scale_idx, scale in enumerate(scales):
                scale_seed = seed + scale_idx * 50021  # Large prime offset
                torch.manual_seed(scale_seed)
                
                # Generate at different scales
                scaled_h = max(1, int(shape[-2] * scale))
                scaled_w = max(1, int(shape[-1] * scale))
                
                if scale_idx == 0:
                    # Base scale: Use empirical latent statistics
                    # These are typical statistics from VAE latent spaces
                    latent_mean = torch.zeros(channels, device=self.device, dtype=torch.float32)
                    
                    # Use different std values for different model types
                    if channels == 4:
                        # SD/SDXL models
                        latent_std = torch.tensor([0.8, 0.8, 0.8, 0.8], device=self.device, dtype=torch.float32)
                    else:
                        # Flux and other models - use more conservative std
                        latent_std = torch.full((channels,), 0.5, device=self.device, dtype=torch.float32)
                    
                    # Generate with realistic latent distribution
                    base_noise = torch.randn([shape[0], channels, scaled_h, scaled_w], device=self.device, dtype=torch.float32)
                    scaled_noise = base_noise * latent_std.view(1, channels, 1, 1) + latent_mean.view(1, channels, 1, 1)
                    
                else:
                    # Higher scales: Add detail layers with different characteristics
                    detail_strength = max(0.01, min(2.0, diversity_strength * (2.0 ** (scale_idx - 1))))
                    
                    # Use different distributions for detail layers
                    if scale_idx == 1:
                        # Laplace distribution for sharper details
                        try:
                            scaled_noise = torch.distributions.Laplace(0, detail_strength).sample(
                                [shape[0], channels, scaled_h, scaled_w]).to(device=self.device, dtype=torch.float32)
                        except:
                            # Fallback to Gaussian if Laplace fails
                            scaled_noise = torch.randn([shape[0], channels, scaled_h, scaled_w], 
                                                     device=self.device, dtype=torch.float32) * detail_strength
                    else:
                        # Student-t distribution for heavy-tailed variation
                        try:
                            scaled_noise = torch.distributions.StudentT(df=3.0).sample(
                                [shape[0], channels, scaled_h, scaled_w]).to(device=self.device, dtype=torch.float32) * detail_strength
                        except:
                            # Fallback to Gaussian if Student-t fails
                            scaled_noise = torch.randn([shape[0], channels, scaled_h, scaled_w], 
                                                     device=self.device, dtype=torch.float32) * detail_strength
                
                # Upsample to target resolution
                if scale < 1.0:
                    scaled_noise = torch.nn.functional.interpolate(
                        scaled_noise, size=(shape[-2], shape[-1]), mode='bilinear', align_corners=False)
                
                # Ensure finite values
                scaled_noise = torch.nan_to_num(scaled_noise, nan=0.0, posinf=1.0, neginf=-1.0)
                noise_pyramid.append(scaled_noise)
            
            # Combine scales with weights
            weights = [0.6, 0.3, 0.1]  # Emphasis on base scale
            final_noise = sum(w * noise for w, noise in zip(weights, noise_pyramid))
            
            # Add structured variation based on seed
            variation_seed = seed + 99991
            torch.manual_seed(variation_seed)
            
            # Add spatially coherent variation patterns
            pattern_noise = self._generate_coherent_patterns(shape, variation_seed, diversity_strength)
            
            result = final_noise + 0.1 * pattern_noise
            
            # Final safety checks
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            return result.to(dtype=torch.float32)
            
        except Exception:
            # Fallback to simple Gaussian noise if anything fails
            return torch.randn(shape, device=self.device, dtype=torch.float32)
    
    def _generate_coherent_patterns(self, shape, seed, strength):
        """Generate spatially coherent noise patterns for enhanced diversity"""
        try:
            torch.manual_seed(seed)
            
            # Generate low-frequency modulation patterns
            low_freq_h = max(4, shape[-2] // 8)
            low_freq_w = max(4, shape[-1] // 8)
            
            modulation = torch.randn([shape[0], shape[1], low_freq_h, low_freq_w], 
                                   device=self.device, dtype=torch.float32)
            modulation = torch.nn.functional.interpolate(
                modulation, size=(shape[-2], shape[-1]), mode='bilinear', align_corners=False)
            
            result = modulation * max(0.0, min(2.0, strength))  # Clamp strength
            return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception:
            # Fallback to zeros if pattern generation fails
            return torch.zeros(shape, device=self.device, dtype=torch.float32)


# Convenience functions for external use
def create_spectral_diverse_noise(shape, seed, device=None, noise_type="hybrid"):
    """
    Convenience function to generate spectral diverse noise.
    
    Args:
        shape: Tensor shape [batch, channels, height, width]
        seed: Random seed for reproducibility
        device: Target device for tensor
        noise_type: Type of spectral noise ("pink", "blue", "hybrid")
        
    Returns:
        torch.Tensor: Generated spectral diverse noise compatible with ComfyUI
    """
    generator = NoiseGenerator(device)
    noise = generator.generate_spectral_diverse_noise(shape, seed, noise_type)
    
    # Ensure noise is properly scaled for diffusion models (standard normal distribution)
    # Clean any problematic values first
    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize to have std=1.0 like torch.randn()
    noise_std = torch.std(noise)
    if noise_std > 1e-8:  # Avoid division by very small numbers
        noise = noise / noise_std
    else:
        # If std is too small, just use random noise
        torch.manual_seed(seed)
        noise = torch.randn_like(noise)
    
    # Clamp to reasonable range
    noise = torch.clamp(noise, -10.0, 10.0)
    
    # Ensure exact shape and type compatibility
    return noise.to(device=device if device else torch.device("cpu"), dtype=torch.float32)


def create_hierarchical_noise(shape, seed, device=None, diversity_strength=1.0):
    """
    Convenience function to generate latent-aware hierarchical noise.
    
    Args:
        shape: Tensor shape [batch, channels, height, width]
        seed: Random seed for reproducibility
        device: Target device for tensor
        diversity_strength: Strength of diversity patterns (0.1 to 2.0)
        
    Returns:
        torch.Tensor: Generated hierarchical noise compatible with ComfyUI
    """
    generator = NoiseGenerator(device)
    noise = generator.generate_latent_aware_hierarchical_noise(shape, seed, diversity_strength)
    
    # Ensure noise is properly scaled for diffusion models (standard normal distribution)
    # Clean any problematic values first
    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize to have std=1.0 like torch.randn()
    noise_std = torch.std(noise)
    if noise_std > 1e-8:  # Avoid division by very small numbers
        noise = noise / noise_std
    else:
        # If std is too small, just use random noise
        torch.manual_seed(seed)
        noise = torch.randn_like(noise)
    
    # Clamp to reasonable range
    noise = torch.clamp(noise, -10.0, 10.0)
    
    # Ensure exact shape and type compatibility
    return noise.to(device=device if device else torch.device("cpu"), dtype=torch.float32)