import torch
import numpy as np
import comfy.samplers
import comfy.sample
import comfy.utils
import traceback
import os
import folder_paths

class CFGExplorer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "cfg_end": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "cfg_steps": ("INT", {"default": 5, "min": 2, "max": 50}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT",)
    RETURN_NAMES = ("images", "latents")
    FUNCTION = "generate"
    CATEGORY = "VariationLab"

    def generate(self, model, positive, negative, latent, vae, seed, steps, cfg_start, cfg_end, cfg_steps,
                sampler_name, scheduler, denoise):
        try:
            # Generate CFG values
            cfg_values = np.linspace(cfg_start, cfg_end, cfg_steps)
            
            results = []
            latent_results = []
            
            # Generate images for each CFG value
            noise = comfy.sample.prepare_noise(latent["samples"], seed)
            noise_mask = None
            
            for cfg in cfg_values:
                # Sample the latent
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, noise_mask=noise_mask)
                latent_results.append({"samples": samples.clone()})
                
                # Decode the latent
                image = vae.decode(samples)
                results.append(image)
            
            # Stack all images into a batch
            batched_images = torch.cat(results, dim=0)
            batched_latents = {"samples": torch.cat([x["samples"] for x in latent_results], dim=0)}
            
            return (batched_images, batched_latents)
        except Exception as e:
            traceback.print_exc()
            # Return a small error image
            error_image = torch.zeros(1, 64, 64, 3)
            error_latent = {"samples": torch.zeros(1, 4, 8, 8)}
            return (error_image, error_latent)


class StepExplorer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps_start": ("INT", {"default": 5, "min": 1, "max": 500}),
                "steps_end": ("INT", {"default": 50, "min": 1, "max": 500}),
                "step_count": ("INT", {"default": 5, "min": 2, "max": 20}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT",)
    RETURN_NAMES = ("images", "latents")
    FUNCTION = "generate"
    CATEGORY = "VariationLab"

    def generate(self, model, positive, negative, latent, vae, seed, steps_start, steps_end, step_count,
                cfg, sampler_name, scheduler, denoise):
        try:
            # Generate step values (rounded to integers)
            step_values = np.linspace(steps_start, steps_end, step_count, dtype=int)
            
            results = []
            latent_results = []
            
            # Generate images for each step count
            noise = comfy.sample.prepare_noise(latent["samples"], seed)
            noise_mask = None
            
            for steps in step_values:
                # Sample the latent
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, noise_mask=noise_mask)
                latent_results.append({"samples": samples.clone()})
                
                # Decode the latent
                image = vae.decode(samples)
                results.append(image)
            
            # Stack all images into a batch
            batched_images = torch.cat(results, dim=0)
            batched_latents = {"samples": torch.cat([x["samples"] for x in latent_results], dim=0)}
            
            return (batched_images, batched_latents)
        except Exception as e:
            traceback.print_exc()
            # Return a small error image
            error_image = torch.zeros(1, 64, 64, 3)
            error_latent = {"samples": torch.zeros(1, 4, 8, 8)}
            return (error_image, error_latent)


class CheckpointExplorer:
    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "latent": ("LATENT",),
                "base_positive_prompt": ("STRING", {"multiline": True, "default": "a photograph of a person"}),
                "base_negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "checkpoint1": (checkpoints,),
                "checkpoint1_clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "checkpoint1_positive_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint1_negative_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint1_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "checkpoint1_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "checkpoint1_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "checkpoint1_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                
                "checkpoint2": (checkpoints,),
                "checkpoint2_clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "checkpoint2_positive_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint2_negative_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint2_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "checkpoint2_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "checkpoint2_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "checkpoint2_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                
                "checkpoint3": (checkpoints,),
                "checkpoint3_clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "checkpoint3_positive_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint3_negative_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint3_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "checkpoint3_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "checkpoint3_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "checkpoint3_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                
                "checkpoint4": (checkpoints,),
                "checkpoint4_clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "checkpoint4_positive_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint4_negative_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint4_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "checkpoint4_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "checkpoint4_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "checkpoint4_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                
                "checkpoint5": (checkpoints,),
                "checkpoint5_clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "checkpoint5_positive_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint5_negative_suffix": ("STRING", {"multiline": True, "default": ""}),
                "checkpoint5_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "checkpoint5_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "checkpoint5_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "checkpoint5_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "VariationLab"

    def generate(self, latent, base_positive_prompt, base_negative_prompt, seed, steps, cfg, denoise, **kwargs):
        try:
            results = []
            checkpoint_info = []
            
            # Collect checkpoint info
            for i in range(1, 6):
                checkpoint_key = f"checkpoint{i}"
                if checkpoint_key in kwargs and kwargs[checkpoint_key] is not None:
                    checkpoint_name = kwargs[checkpoint_key]
                    clip_skip = kwargs.get(f"{checkpoint_key}_clip_skip", -1)
                    pos_suffix = kwargs.get(f"{checkpoint_key}_positive_suffix", "")
                    neg_suffix = kwargs.get(f"{checkpoint_key}_negative_suffix", "")
                    
                    # Get checkpoint-specific sampler settings or use defaults
                    sampler = kwargs.get(f"{checkpoint_key}_sampler", "euler")
                    scheduler = kwargs.get(f"{checkpoint_key}_scheduler", "normal")
                    checkpoint_steps = kwargs.get(f"{checkpoint_key}_steps", steps)
                    checkpoint_cfg = kwargs.get(f"{checkpoint_key}_cfg", cfg)
                    
                    checkpoint_info.append({
                        "name": checkpoint_name,
                        "clip_skip": clip_skip,
                        "positive_prompt": base_positive_prompt + (f", {pos_suffix}" if pos_suffix else ""),
                        "negative_prompt": base_negative_prompt + (f", {neg_suffix}" if neg_suffix else ""),
                        "sampler": sampler,
                        "scheduler": scheduler,
                        "steps": checkpoint_steps,
                        "cfg": checkpoint_cfg
                    })
            
            # No checkpoints specified
            if not checkpoint_info:
                error_image = torch.zeros(1, 64, 64, 3)
                return (error_image,)
            
            # Generate image for each checkpoint
            for info in checkpoint_info:
                checkpoint_path = folder_paths.get_full_path("checkpoints", info["name"])
                
                # Load model
                model, clip, vae = comfy.sd.load_checkpoint_guess_config(checkpoint_path)
                
                # Apply clip skip
                if info["clip_skip"] != -1:
                    clip = comfy.sd.CLIP(clip.tokenizer, clip.transformer, clip.device, int(info["clip_skip"]))
                
                # Encode positive and negative prompts
                positive = comfy.sd.encode_prompt(clip, info["positive_prompt"])
                negative = comfy.sd.encode_prompt(clip, info["negative_prompt"])
                
                # Sample the latent
                noise = comfy.sample.prepare_noise(latent["samples"], seed)
                samples = comfy.sample.sample(model, noise, info["steps"], info["cfg"], 
                                              info["sampler"], info["scheduler"], 
                                              positive, negative, latent["samples"],
                                              denoise=denoise, noise_mask=None)
                
                # Decode the latent
                image = vae.decode(samples)
                results.append(image)
            
            # Stack all images into a batch
            batched_images = torch.cat(results, dim=0)
            
            return (batched_images,)
        except Exception as e:
            traceback.print_exc()
            # Return a small error image
            error_image = torch.zeros(1, 64, 64, 3)
            return (error_image,)


NODE_CLASS_MAPPINGS = {
    "CFGExplorer": CFGExplorer,
    "StepExplorer": StepExplorer,
    "CheckpointExplorer": CheckpointExplorer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGExplorer": "VariationLab: CFG Explorer",
    "StepExplorer": "VariationLab: Step Explorer",
    "CheckpointExplorer": "VariationLab: Checkpoint Explorer",
}
