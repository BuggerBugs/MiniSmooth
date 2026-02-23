import torch
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50,
    resnet101, resnet152, resnet_custom
)
from .mobilenet_v2 import mobilenet_v2
from torchvision import models
import timm

def load_model(config):
    # Handle ConvNeXt XL
    if config['type'].startswith('convnext_xlarge'):
        model = timm.create_model('convnext_xlarge', pretrained=True, **config['kwargs'])

    # Handle ResNet-152
    elif config['type'].startswith('res152'):
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1, **config['kwargs'])

    else:
        # Fallback for other models registered in MQBench
        model = globals()[config['type']](**config['kwargs'])
        checkpoint = torch.load(config.path, map_location='cpu')
        if config.type == 'mobilenet_v2':
            checkpoint = checkpoint['model']
        model.load_state_dict(checkpoint)

    return model
