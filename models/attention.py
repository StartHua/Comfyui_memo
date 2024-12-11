from typing import Any, Dict, Optional

import torch
from diffusers.models.attention import (
    AdaLayerNorm,
    AdaLayerNormZero,
    Attention,
    FeedForward,
)
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from einops import rearrange
from torch import nn

from .attention_processor import Attention as CustomAttention
from .attention_processor import JointAttnProcessor2_0


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        is_final_block: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.is_final_block = is_final_block

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        if not is_final_block:
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )

            # 2. Cross-Attn
            if cross_attention_dim is not None or double_self_attention:
                # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
                # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
                # the second cross attention block.
                self.norm2 = (
                    AdaLayerNorm(dim, num_embeds_ada_norm)
                    if self.use_ada_layer_norm
                    else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
                )
                self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=(cross_attention_dim if not double_self_attention else None),
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
            else:
                self.norm2 = None
                self.attn2 = None

            # 3. Feed-forward
            if not self.use_ada_layer_norm_single:
                self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
            )

            # 4. Fuser
            if attention_type in {"gated", "gated-text-image"}:  # Updated line
                self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

            # 5. Scale-shift for PixArt-Alpha.
            if self.use_ada_layer_norm_single:
                self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

            # let chunk size default to None
            self._chunk_size = None
            self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        gate_msa = None
        scale_mlp = None
        shift_mlp = None
        gate_mlp = None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        ref_feature = norm_hidden_states
        if self.is_final_block:
            return None, ref_feature
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(encoder_hidden_states if self.only_cross_attention else None),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states.repeat(
                    norm_hidden_states.shape[0] // encoder_hidden_states.shape[0], 1, 1
                ),
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states, ref_feature


class TemporalBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.use_ada_layer_norm_zero = False

        # Temp-Attn
        if unet_use_temporal_attention is None:
            unet_use_temporal_attention = False
        if unet_use_temporal_attention:
            self.attn_temp = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        ref_img_feature: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        video_length=None,
        uc_mask=None,
    ):
        norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        ref_img_feature = ref_img_feature.repeat(video_length, 1, 1)
        modify_norm_hidden_states = torch.cat((norm_hidden_states, ref_img_feature), dim=1).to(
            dtype=norm_hidden_states.dtype
        )
        hidden_states_uc = (
            self.attn1(
                norm_hidden_states,
                encoder_hidden_states=modify_norm_hidden_states,
                attention_mask=attention_mask,
            )
            + hidden_states
        )
        if uc_mask is not None:
            hidden_states_c = hidden_states_uc.clone()
            _uc_mask = uc_mask.clone()
            if hidden_states.shape[0] != _uc_mask.shape[0]:
                _uc_mask = (
                    torch.Tensor([1] * (hidden_states.shape[0] // 2) + [0] * (hidden_states.shape[0] // 2))
                    .to(hidden_states_uc.device)
                    .bool()
                )
            hidden_states_c[_uc_mask] = (
                self.attn1(
                    norm_hidden_states[_uc_mask],
                    encoder_hidden_states=norm_hidden_states[_uc_mask],
                    attention_mask=attention_mask,
                )
                + hidden_states[_uc_mask]
            )
            hidden_states = hidden_states_c.clone()
        else:
            hidden_states = hidden_states_uc

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        # Drops labels to enable classifier-free guidance.
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = torch.tensor(force_drop_ids == 1)
        labels = torch.where(drop_ids, self.num_classes, labels)

        return labels

    def forward(self, labels: torch.LongTensor, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)

        return embeddings


class EmoAdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=9,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        class_dropout_prob=0.3,
    ):
        super().__init__()
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True))

    def forward(self, x, emotion=None):
        emo_embedding = self.class_embedder(emotion)
        shift, scale = self.adaLN_modulation(emo_embedding).chunk(2, dim=1)
        if emotion.shape[0] > 1:
            repeat = x.shape[0] // emo_embedding.shape[0]
            scale = scale.unsqueeze(1)
            scale = torch.repeat_interleave(scale, repeats=repeat, dim=0)
            shift = shift.unsqueeze(1)
            shift = torch.repeat_interleave(shift, repeats=repeat, dim=0)
        else:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        x = self.norm(x) * (1 + scale) + shift

        return x


class JointAudioTemporalBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        depth=0,
        unet_block_name=None,
        use_ada_layer_norm=False,
        emo_drop_rate=0.3,
        is_final_block=False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = use_ada_layer_norm
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.unet_block_name = unet_block_name
        self.depth = depth
        self.is_final_block = is_final_block

        self.norm1 = (
            EmoAdaLayerNorm(dim, num_classes=9, class_dropout_prob=emo_drop_rate)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )
        self.attn1 = CustomAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        self.audio_norm1 = (
            EmoAdaLayerNorm(cross_attention_dim, num_classes=9, class_dropout_prob=emo_drop_rate)
            if self.use_ada_layer_norm
            else nn.LayerNorm(cross_attention_dim)
        )
        self.audio_attn1 = CustomAttention(
            query_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        self.norm2 = (
            EmoAdaLayerNorm(dim, num_classes=9, class_dropout_prob=emo_drop_rate)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )
        self.audio_norm2 = (
            EmoAdaLayerNorm(cross_attention_dim, num_classes=9, class_dropout_prob=emo_drop_rate)
            if self.use_ada_layer_norm
            else nn.LayerNorm(cross_attention_dim)
        )

        # Joint Attention
        self.attn2 = CustomAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=dim,
            added_kv_proj_dim=cross_attention_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            only_cross_attention=False,
            out_dim=dim,
            context_out_dim=cross_attention_dim,
            context_pre_only=False,
            processor=JointAttnProcessor2_0(),
            is_final_block=is_final_block,
        )

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        if not is_final_block:
            self.audio_ff = FeedForward(cross_attention_dim, dropout=dropout, activation_fn=activation_fn)
            self.audio_norm3 = nn.LayerNorm(cross_attention_dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        emotion=None,
    ):
        norm_hidden_states = (
            self.norm1(hidden_states, emotion) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        norm_encoder_hidden_states = (
            self.audio_norm1(encoder_hidden_states, emotion)
            if self.use_ada_layer_norm
            else self.audio_norm1(encoder_hidden_states)
        )

        hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        encoder_hidden_states = (
            self.audio_attn1(norm_encoder_hidden_states, attention_mask=attention_mask) + encoder_hidden_states
        )

        norm_hidden_states = (
            self.norm2(hidden_states, emotion) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )
        norm_encoder_hidden_states = (
            self.audio_norm2(encoder_hidden_states, emotion)
            if self.use_ada_layer_norm
            else self.audio_norm2(encoder_hidden_states)
        )

        joint_hidden_states, joint_encoder_hidden_states = self.attn2(
            norm_hidden_states,
            norm_encoder_hidden_states,
        )

        hidden_states = joint_hidden_states + hidden_states
        if not self.is_final_block:
            encoder_hidden_states = joint_encoder_hidden_states + encoder_hidden_states

        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        if not self.is_final_block:
            encoder_hidden_states = self.audio_ff(self.audio_norm3(encoder_hidden_states)) + encoder_hidden_states
        else:
            encoder_hidden_states = None

        return hidden_states, encoder_hidden_states


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)

    return module
