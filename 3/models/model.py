import torch.nn as nn
import torchvision.models as models


def make_model(name="resnet50"):
    if name == "alexnet":
        model = models.alexnet(pretrained=True)

        for p in model.features.parameters():
            p.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)

    elif name == "vgg16":
        model = models.vgg16(pretrained=True)

        for p in model.features.parameters():
            p.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)

    elif name == "resnet50":
        model = models.resnet50(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)

        for p in model.features.parameters():
            p.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)

    else:
        raise ValueError("model name is not supported")

    return model