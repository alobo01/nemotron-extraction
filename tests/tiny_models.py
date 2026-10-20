"""Three-layer stand-in for the Nemotron decoder: Mamba, MoE and attention mixers."""
import torch
from torch import nn


class TinyMamba(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2


class TinyMoE(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 3


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))

    def forward(self, hidden_states):
        return self.o_proj(torch.cat([hidden_states, hidden_states + 5], -1))


class Block(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer

    def forward(self, hidden_states):
        return hidden_states + self.mixer(hidden_states)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(TinyMamba()), Block(TinyMoE()), Block(TinyAttention())])
        self.fail = False

    def forward(self, input_ids, use_cache=False):
        hidden = torch.stack([input_ids.float(), input_ids.float() + 1], -1)
        for layer in self.layers:
            hidden = layer(hidden)
        if self.fail:
            raise RuntimeError("forward failure")
        return hidden


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()
