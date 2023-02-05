import torch
import torchvision

from torch import nn


class AffordanceAdapter(nn.Module):
    def __init__(self, model, idx=0):
        super().__init__()
        self.model = model
        self.idx = idx

    def forward(self, x):
        outputs = self.model(x)
        return outputs['out'][:, self.idx:self.idx + 1, :, :]


class SegmentationModelRepository:
    @staticmethod
    def fcn_resnet_50():
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        return AffordanceAdapter(model)

    @staticmethod
    def fcn_resnet_50_rgbd(nchannels=4, pretrained=True):
        backbone = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained, progress=True)
        backbone.backbone.conv1 = nn.Conv2d(nchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return AffordanceAdapter(backbone)

    @staticmethod
    def lraspp_mobilenet_v3_large(nchannels=3):
        assert nchannels == 3, 'Only RGB images are supported'
        backbone = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, progress=True)
        return AffordanceAdapter(backbone)

    @staticmethod
    def fcn_resnet_101():
        model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True)
        return AffordanceAdapter(model)

    @staticmethod
    def by_name(model_name):
        return torch.load('../resources/models/{}.pkl'.format(model_name))
