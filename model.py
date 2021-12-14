import torch 
import torch.nn as nn
import torchvision.models as models 
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

def trafficLightDetectionModel(pretrained = True, backbone_name = 'resnet50', num_classes = 13):

    resnet_backbone = resnet_fpn_backbone(backbone_name, pretrained)
    
    model = FasterRCNN(resnet_backbone, num_classes)

    return model

