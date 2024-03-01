from src.modeling import UNet
import torch
from safetensors.torch import load_file
file_path = 'results/checkpoint-42500/model.safetensors'
loaded = load_file(file_path)
model = UNet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], initializers=None, apply_last_layer=True, padding=True)
model.load_state_dict(loaded)