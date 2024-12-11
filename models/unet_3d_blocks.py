from typing import Any, Dict

import torch
from diffusers.utils import is_torch_version
from einops import rearrange
from torch import nn

from .motion_module import MemoryLinearAttnTemporalModule
from .resnet import Downsample3D, ResnetBlock3D, Upsample3D
from .transformer_3d import Transformer3DModel


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)

        return module(*inputs)

    return custom_forward


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    audio_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_kwargs=None,
    depth=0,
    emo_drop_rate=0.3,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
        )

    if down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            audio_attention_dim=audio_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
            depth=depth,
            emo_drop_rate=emo_drop_rate,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    audio_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_kwargs=None,
    depth=0,
    emo_drop_rate=0.3,
    is_final_block=False,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
        )

    if up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            audio_attention_dim=audio_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
            depth=depth,
            emo_drop_rate=emo_drop_rate,
            is_final_block=is_final_block,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        audio_attention_dim=1024,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        motion_module_kwargs=None,
        depth=0,
        emo_drop_rate=0.3,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
        ]
        attentions = []
        motion_modules = []
        audio_modules = []

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            audio_modules.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=audio_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=False,
                    upcast_attention=upcast_attention,
                    use_audio_module=True,
                    depth=depth,
                    unet_block_name="mid",
                    emo_drop_rate=emo_drop_rate,
                )
            )

            motion_modules.append(
                MemoryLinearAttnTemporalModule(
                    in_channels=in_channels,
                    **motion_module_kwargs,
                )
            )
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.audio_modules = nn.ModuleList(audio_modules)
        self.motion_modules = nn.ModuleList(motion_modules)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        ref_feature_list,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        audio_embedding=None,
        emotion=None,
        uc_mask=None,
        is_new_audio=True,
        update_past_memory=False,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        for i, (attn, resnet, audio_module, motion_module) in enumerate(
            zip(
                self.attentions,
                self.resnets[1:],
                self.audio_modules,
                self.motion_modules,
            )
        ):
            ref_feature = ref_feature_list[i]
            ref_feature = ref_feature[0]
            ref_feature = rearrange(
                ref_feature,
                "(b f) (h w) c -> b c f h w",
                b=hidden_states.shape[0],
                w=hidden_states.shape[-1],
            )
            ref_img_feature = ref_feature[:, :, :1, :, :]
            ref_img_feature = rearrange(
                ref_img_feature,
                "b c f h w -> (b f) (h w) c",
            )
            motion_frames = ref_feature[:, :, 1:, :, :]

            hidden_states = attn(
                hidden_states,
                ref_img_feature,
                encoder_hidden_states=encoder_hidden_states,
                uc_mask=uc_mask,
                return_dict=False,
            )
            if audio_module is not None:
                hidden_states, audio_embedding = audio_module(
                    hidden_states,
                    ref_img_feature=None,
                    encoder_hidden_states=audio_embedding,
                    attention_mask=attention_mask,
                    return_dict=False,
                    emotion=emotion,
                )
            if motion_module is not None:
                motion_frames = motion_frames.to(device=hidden_states.device, dtype=hidden_states.dtype)
                hidden_states = motion_module(
                    hidden_states=hidden_states,
                    motion_frames=motion_frames,
                    encoder_hidden_states=encoder_hidden_states,
                    is_new_audio=is_new_audio,
                    update_past_memory=update_past_memory,
                )

            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)

        if audio_module is not None:
            return hidden_states, audio_embedding
        else:
            return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        audio_attention_dim=1024,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_kwargs=None,
        depth=0,
        emo_drop_rate=0.3,
    ):
        super().__init__()
        resnets = []
        attentions = []
        audio_modules = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            audio_modules.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=audio_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_audio_module=True,
                    depth=depth,
                    unet_block_name="down",
                    emo_drop_rate=emo_drop_rate,
                )
            )
            motion_modules.append(
                MemoryLinearAttnTemporalModule(
                    in_channels=out_channels,
                    **motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.audio_modules = nn.ModuleList(audio_modules)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        ref_feature_list,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        audio_embedding=None,
        emotion=None,
        uc_mask=None,
        is_new_audio=True,
        update_past_memory=False,
    ):
        output_states = ()

        for i, (resnet, attn, audio_module, motion_module) in enumerate(
            zip(self.resnets, self.attentions, self.audio_modules, self.motion_modules)
        ):
            ref_feature = ref_feature_list[i]
            ref_feature = ref_feature[0]
            ref_feature = rearrange(
                ref_feature,
                "(b f) (h w) c -> b c f h w",
                b=hidden_states.shape[0],
                w=hidden_states.shape[-1],
            )
            ref_img_feature = ref_feature[:, :, :1, :, :]
            ref_img_feature = rearrange(
                ref_img_feature,
                "b c f h w -> (b f) (h w) c",
            )
            motion_frames = ref_feature[:, :, 1:, :, :]

            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            hidden_states = attn(
                hidden_states,
                ref_img_feature,
                encoder_hidden_states=encoder_hidden_states,
                uc_mask=uc_mask,
                return_dict=False,
            )

            if audio_module is not None:
                hidden_states, audio_embedding = audio_module(
                    hidden_states,
                    ref_img_feature=None,
                    encoder_hidden_states=audio_embedding,
                    attention_mask=attention_mask,
                    return_dict=False,
                    emotion=emotion,
                )

            # add motion module
            if motion_module is not None:
                motion_frames = motion_frames.to(device=hidden_states.device, dtype=hidden_states.dtype)
                hidden_states = motion_module(
                    hidden_states=hidden_states,
                    motion_frames=motion_frames,
                    encoder_hidden_states=encoder_hidden_states,
                    is_new_audio=is_new_audio,
                    update_past_memory=update_past_memory,
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        if audio_module is not None:
            return hidden_states, output_states, audio_embedding
        else:
            return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            motion_modules.append(
                MemoryLinearAttnTemporalModule(
                    in_channels=out_channels,
                    **motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        ref_feature_list,
        temb=None,
        encoder_hidden_states=None,
        is_new_audio=True,
        update_past_memory=False,
    ):
        output_states = ()

        for i, (resnet, motion_module) in enumerate(zip(self.resnets, self.motion_modules)):
            ref_feature = ref_feature_list[i]
            ref_feature = rearrange(
                ref_feature,
                "(b f) c h w -> b c f h w",
                b=hidden_states.shape[0],
                w=hidden_states.shape[-1],
            )
            motion_frames = ref_feature[:, :, 1:, :, :]

            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            if motion_module is not None:
                hidden_states = motion_module(
                    hidden_states=hidden_states,
                    motion_frames=motion_frames,
                    encoder_hidden_states=encoder_hidden_states,
                    is_new_audio=is_new_audio,
                    update_past_memory=update_past_memory,
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        audio_attention_dim=1024,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_motion_module=None,
        use_inflated_groupnorm=None,
        motion_module_kwargs=None,
        depth=0,
        emo_drop_rate=0.3,
        is_final_block=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        audio_modules = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        self.is_final_block = is_final_block

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            audio_modules.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=audio_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_audio_module=True,
                    depth=depth,
                    unet_block_name="up",
                    emo_drop_rate=emo_drop_rate,
                    is_final_block=(is_final_block and i == num_layers - 1),
                )
            )
            motion_modules.append(
                MemoryLinearAttnTemporalModule(
                    in_channels=out_channels,
                    **motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.audio_modules = nn.ModuleList(audio_modules)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        ref_feature_list,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        audio_embedding=None,
        emotion=None,
        uc_mask=None,
        is_new_audio=True,
        update_past_memory=False,
    ):
        for i, (resnet, attn, audio_module, motion_module) in enumerate(
            zip(self.resnets, self.attentions, self.audio_modules, self.motion_modules)
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            ref_feature = ref_feature_list[i]
            ref_feature = ref_feature[0]
            ref_feature = rearrange(
                ref_feature,
                "(b f) (h w) c -> b c f h w",
                b=hidden_states.shape[0],
                w=hidden_states.shape[-1],
            )
            ref_img_feature = ref_feature[:, :, :1, :, :]
            ref_img_feature = rearrange(
                ref_img_feature,
                "b c f h w -> (b f) (h w) c",
            )
            motion_frames = ref_feature[:, :, 1:, :, :]

            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            hidden_states = attn(
                hidden_states,
                ref_img_feature,
                encoder_hidden_states=encoder_hidden_states,
                uc_mask=uc_mask,
                return_dict=False,
            )

            if audio_module is not None:
                hidden_states, audio_embedding = audio_module(
                    hidden_states,
                    ref_img_feature=None,
                    encoder_hidden_states=audio_embedding,
                    attention_mask=attention_mask,
                    return_dict=False,
                    emotion=emotion,
                )

            # add motion module
            if motion_module is not None:
                motion_frames = motion_frames.to(device=hidden_states.device, dtype=hidden_states.dtype)
                hidden_states = motion_module(
                    hidden_states,
                    motion_frames,
                    encoder_hidden_states,
                    is_new_audio=is_new_audio,
                    update_past_memory=update_past_memory,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        if audio_module is not None:
            return hidden_states, audio_embedding
        else:
            return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            motion_modules.append(
                MemoryLinearAttnTemporalModule(
                    in_channels=out_channels,
                    **motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        ref_feature_list,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        encoder_hidden_states=None,
        is_new_audio=True,
        update_past_memory=False,
    ):
        for i, (resnet, motion_module) in enumerate(zip(self.resnets, self.motion_modules)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            ref_feature = ref_feature_list[i]
            ref_feature = rearrange(
                ref_feature,
                "(b f) c h w -> b c f h w",
                b=hidden_states.shape[0],
                w=hidden_states.shape[-1],
            )
            motion_frames = ref_feature[:, :, 1:, :, :]

            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            if motion_module is not None:
                hidden_states = motion_module(
                    hidden_states=hidden_states,
                    motion_frames=motion_frames,
                    encoder_hidden_states=encoder_hidden_states,
                    is_new_audio=is_new_audio,
                    update_past_memory=update_past_memory,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
