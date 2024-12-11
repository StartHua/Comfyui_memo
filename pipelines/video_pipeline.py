import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange


@dataclass
class VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class VideoPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae,
        reference_net,
        diffusion_net,
        image_proj,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_net=reference_net,
            diffusion_net=diffusion_net,
            scheduler=scheduler,
            image_proj=image_proj,
        )

        self.vae_scale_factor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
        )

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_latents(
        self,
        batch_size: int,  # Number of videos to generate in parallel
        num_channels_latents: int,  # Number of channels in the latents
        width: int,  # Width of the video frame
        height: int,  # Height of the video frame
        video_length: int,  # Length of the video in frames
        dtype: torch.dtype,  # Data type of the latents
        device: torch.device,  # Device to store the latents on
        generator: Optional[torch.Generator] = None,  # Random number generator for reproducibility
        latents: Optional[torch.Tensor] = None,  # Pre-generated latents (optional)
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        face_emb,
        audio_tensor,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        audio_emotion=None,
        emotion_class_num=None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # prepare clip image embeddings
        clip_image_embeds = face_emb
        clip_image_embeds = clip_image_embeds.to(self.image_proj.device, self.image_proj.dtype)

        encoder_hidden_states = self.image_proj(clip_image_embeds)
        uncond_encoder_hidden_states = self.image_proj(torch.zeros_like(clip_image_embeds))

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

        num_channels_latents = self.diffusion_net.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = rearrange(ref_image, "b f c h w -> (b f) c h w")
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image_tensor, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        # To save memory on GPUs like RTX 4090, we encode each frame separately
        # ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = []
        for frame_idx in range(ref_image_tensor.shape[0]):
            ref_image_latents.append(self.vae.encode(ref_image_tensor[frame_idx : frame_idx + 1]).latent_dist.mean)
        ref_image_latents = torch.cat(ref_image_latents, dim=0)

        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        if do_classifier_free_guidance:
            uncond_audio_tensor = torch.zeros_like(audio_tensor)
            audio_tensor = torch.cat([uncond_audio_tensor, audio_tensor], dim=0)
            audio_tensor = audio_tensor.to(dtype=self.diffusion_net.dtype, device=self.diffusion_net.device)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps)):
                t = timesteps[i]
                # Forward reference image
                if i == 0:
                    ref_features = self.reference_net(
                        ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                        torch.zeros_like(t),
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                audio_emotion = torch.tensor(torch.mode(audio_emotion).values.item()).to(
                    dtype=torch.int, device=self.diffusion_net.device
                )
                if do_classifier_free_guidance:
                    uncond_audio_emotion = torch.full_like(audio_emotion, emotion_class_num)
                    audio_emotion = torch.cat(
                        [uncond_audio_emotion.unsqueeze(0), audio_emotion.unsqueeze(0)],
                        dim=0,
                    )

                    uc_mask = (
                        torch.Tensor(
                            [1] * batch_size * num_images_per_prompt * 16
                            + [0] * batch_size * num_images_per_prompt * 16
                        )
                        .to(device)
                        .bool()
                    )
                else:
                    uc_mask = None

                noise_pred = self.diffusion_net(
                    latent_model_input,
                    ref_features,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_embedding=audio_tensor,
                    audio_emotion=audio_emotion,
                    uc_mask=uc_mask,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return VideoPipelineOutput(videos=images)
