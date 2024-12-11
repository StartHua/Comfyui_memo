import torch
from diffusers import ConfigMixin, ModelMixin
from einops import rearrange
from torch import nn


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)

        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens
