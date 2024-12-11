from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput
from einops import rearrange, repeat
from torch import nn

from .attention import JointAudioTemporalBasicTransformerBlock, TemporalBasicTransformerBlock


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)

        return module(*inputs)

    return custom_forward


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_audio_module=False,
        depth=0,
        unet_block_name=None,
        emo_drop_rate=0.3,
        is_final_block=False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_audio_module = use_audio_module
        # Define input layers
        self.in_channels = in_channels
        self.is_final_block = is_final_block

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        if use_audio_module:
            self.transformer_blocks = nn.ModuleList(
                [
                    JointAudioTemporalBasicTransformerBlock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        depth=depth,
                        unet_block_name=unet_block_name,
                        use_ada_layer_norm=True,
                        emo_drop_rate=emo_drop_rate,
                        is_final_block=(is_final_block and d == num_layers - 1),
                    )
                    for d in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalBasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                    for _ in range(num_layers)
                ]
            )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states,
        ref_img_feature=None,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        emotion=None,
        uc_mask=None,
        return_dict: bool = True,
    ):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        if self.use_audio_module:
            if encoder_hidden_states.dim() == 4:
                encoder_hidden_states = rearrange(
                    encoder_hidden_states,
                    "bs f margin dim -> (bs f) margin dim",
                )
        else:
            if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
                encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b f) n c", f=video_length)

        batch, _, height, weight = hidden_states.shape
        residual = hidden_states
        if self.use_audio_module:
            residual_audio = encoder_hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                if isinstance(block, TemporalBasicTransformerBlock):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        ref_img_feature,
                        None,  # attention_mask
                        encoder_hidden_states,
                        timestep,
                        None,  # cross_attention_kwargs
                        video_length,
                        uc_mask,
                    )
                elif isinstance(block, JointAudioTemporalBasicTransformerBlock):
                    (
                        hidden_states,
                        encoder_hidden_states,
                    ) = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        attention_mask,
                        emotion,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        timestep,
                        attention_mask,
                        video_length,
                    )
            else:
                if isinstance(block, TemporalBasicTransformerBlock):
                    hidden_states = block(
                        hidden_states=hidden_states,
                        ref_img_feature=ref_img_feature,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        video_length=video_length,
                        uc_mask=uc_mask,
                    )
                elif isinstance(block, JointAudioTemporalBasicTransformerBlock):
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,  # shape [2, 4096, 320]
                        encoder_hidden_states=encoder_hidden_states,  # shape [2, 20, 640]
                        attention_mask=attention_mask,
                        emotion=emotion,
                    )
                else:
                    hidden_states = block(
                        hidden_states,  # shape [2, 4096, 320]
                        encoder_hidden_states=encoder_hidden_states,  # shape [2, 20, 640]
                        attention_mask=attention_mask,
                        timestep=timestep,
                        video_length=video_length,
                    )

        # Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if self.use_audio_module and not self.is_final_block:
            audio_output = encoder_hidden_states + residual_audio
        else:
            audio_output = None

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            if self.use_audio_module:
                return output, audio_output
            else:
                return output

        if self.use_audio_module:
            return output, audio_output
        else:
            return output
