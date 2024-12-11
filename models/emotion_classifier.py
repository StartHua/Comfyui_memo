from collections import OrderedDict

import torch
from diffusers import ConfigMixin, ModelMixin


class AudioEmotionClassifierModel(ModelMixin, ConfigMixin):
    num_emotion_classes = 9

    def __init__(self, num_classifier_layers=5, num_classifier_channels=2048):
        super().__init__()

        if num_classifier_layers == 1:
            self.layers = torch.nn.Linear(1024, self.num_emotion_classes)
        else:
            layer_list = [
                ("fc1", torch.nn.Linear(1024, num_classifier_channels)),
                ("relu1", torch.nn.ReLU()),
            ]
            for n in range(num_classifier_layers - 2):
                layer_list.append((f"fc{n+2}", torch.nn.Linear(num_classifier_channels, num_classifier_channels)))
                layer_list.append((f"relu{n+2}", torch.nn.ReLU()))
            layer_list.append(
                (f"fc{num_classifier_layers}", torch.nn.Linear(num_classifier_channels, self.num_emotion_classes))
            )
            self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        x = self.layers(x)
        x = torch.softmax(x, dim=-1)
        return x
