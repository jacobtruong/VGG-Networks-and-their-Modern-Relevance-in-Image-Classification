import os
import requests
import tarfile
import time

from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torchvision.models as models
import torch.nn as nn
import torch
from torch import optim
from torchsummary import summary

import PIL.Image
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import timm
import uuid
import pickle


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# VGG16 implementation following the architecture from the paper
class VGG16(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Convolutional Layer 1 - WIDTH x HEIGHT x 3 -> WIDTH x HEIGHT x 64 -> WIDTH/2 x HEIGHT/2 x 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 2 - WIDTH/2 x HEIGHT/2 x 64 -> WIDTH/2 x HEIGHT/2 x 128 -> WIDTH/4 x HEIGHT/4 x 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 3 - WIDTH/4 x HEIGHT/4 x 128 -> WIDTH/4 x HEIGHT/4 x 256 -> WIDTH/8 x HEIGHT/8 x 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 4 - WIDTH/8 x HEIGHT/8 x 256 -> WIDTH/8 x HEIGHT/8 x 512 -> WIDTH/16 x HEIGHT/16 x 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 5 - WIDTH/16 x HEIGHT/16 x 512 -> WIDTH/16 x HEIGHT/16 x 512 -> WIDTH/32 x HEIGHT/32 x 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # WIDTH/32 x HEIGHT/32 x 512 -> 7 x 7 x 512
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        # 7 x 7 x 512 -> 4096 -> 4096 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# VGG19 implementation following the architecture from the paper
class VGG19(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            # Convolutional Layer 1 - WIDTH x HEIGHT x 3 -> WIDTH x HEIGHT x 64 -> WIDTH/2 x HEIGHT/2 x 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 2 - WIDTH/2 x HEIGHT/2 x 64 -> WIDTH/2 x HEIGHT/2 x 128 -> WIDTH/4 x HEIGHT/4 x 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 3 - WIDTH/4 x HEIGHT/4 x 128 -> WIDTH/4 x HEIGHT/4 x 256 -> WIDTH/8 x HEIGHT/8 x 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 4 - WIDTH/8 x HEIGHT/8 x 256 -> WIDTH/8 x HEIGHT/8 x 512 -> WIDTH/16 x HEIGHT/16 x 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer 5 - WIDTH/16 x HEIGHT/16 x 512 -> WIDTH/16 x HEIGHT/16 x 512 -> WIDTH/32 x HEIGHT/32 x 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # WIDTH/32 x HEIGHT/32 x 512 -> 7 x 7 x 512
        # NOTE: THIS WAS NOT STRICTLY A PART OF THE PAPER. I ADDED THIS TO MAKE SURE THE ARCHITECTURE WORKS FOR ANY INPUT SIZE
        # WITHOUT THIS, ANY INPUT SIZE THAT IS NOT 224 x 224 WILL THROW AN ERROR! YOU HAVE BEEN WARNED!
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        # 7 x 7 x 512 -> 4096 -> 4096 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Define the Vision Transformer model
class CustomEfficientNetB0Model(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetB0Model, self).__init__()
        # Load the pre-trained ViT model
        self.effnet = timm.create_model('timm/efficientnet_b0.ra4_e3600_r224_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.effnet)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        for param in self.effnet.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.effnet.classifier = nn.Linear(self.effnet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.effnet(x)

# Define the Vision Transformer model
class CustomMobileNetV4_MH(nn.Module):
    def __init__(self, num_classes):
        super(CustomMobileNetV4_MH, self).__init__()
        # Load the pre-trained ViT model
        self.mobilenet = timm.create_model('timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.mobilenet)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.mobilenet.classifier = nn.Linear(self.mobilenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


# Define the Vision Transformer model
class CustomEdgeNextModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomEdgeNextModel, self).__init__()
        # Load the pre-trained ViT model
        self.edgenext = timm.create_model('timm/edgenext_x_small.in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.edgenext)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        for param in self.edgenext.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.edgenext.head.fc = nn.Linear(self.edgenext.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.edgenext(x)

# Define the Vision Transformer model
class CustomEdgeNextUSIModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomEdgeNextUSIModel, self).__init__()
        # Load the pre-trained ViT model
        self.edgenext = timm.create_model('timm/edgenext_small.usi_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.edgenext)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        for param in self.edgenext.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.edgenext.head.fc = nn.Linear(self.edgenext.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.edgenext(x)
    
# Define the Vision Transformer model
class CustomViTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomViTModel, self).__init__()
        # Load the pre-trained ViT model
        self.base_vit = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.base_vit)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        for param in self.base_vit.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.base_vit.head = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.base_vit.head.in_features, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_vit(x)

# Define the Vision Transformer model
class CustomTinyViT11mModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomTinyViT11mModel, self).__init__()
        # Load the pre-trained ViT model
        self.tinyvit = timm.create_model('timm/tiny_vit_11m_224.dist_in22k_ft_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.tinyvit)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        for param in self.tinyvit.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.tinyvit.head.fc = nn.Linear(self.tinyvit.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.tinyvit(x)

# Define the Vision Transformer model
class CustomTinyViT5mModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomTinyViT5mModel, self).__init__()
        # Load the pre-trained ViT model
        self.tinyvit = timm.create_model('timm/tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.tinyvit)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        # for param in self.tinyvit.parameters():
        #     param.requires_grad = False

        # Replace the classifier head
        self.tinyvit.head.fc = nn.Linear(self.tinyvit.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.tinyvit(x)

class CustomTinyViT21mModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomTinyViT21mModel, self).__init__()
        # Load the pre-trained ViT model
        self.tinyvit = timm.create_model('timm/tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.tinyvit)
        self.transforms_train = timm.data.create_transform(**data_config, is_training=True)
        self.transforms_val = timm.data.create_transform(**data_config, is_training=False)
        
        # Freeze the base model
        for param in self.tinyvit.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.tinyvit.head.fc = nn.Linear(self.tinyvit.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.tinyvit(x)

def create_model(model_name, num_classes, pretrained=True):

    model_dict = {
        "efficientnet_v2_l": True,
        # "VGG16_256": True,
        "VGG16": True,
        # "VGG19_256": True
        "VGG19": True,
        "ResNet152": True,
        "ResNet18": True,
        "EfficientNetB0": True,
        "MobileNetV4": True,
        "EdgeNeXtXS": True,
        "EdgeNeXtS-USI": True,
        "EVA02": True,
        "TinyViT11M": True,
        "TinyViT21M": True,
        "TinyViT5M": True,
    }

    if model_name not in model_dict:
        raise ValueError("Invalid model name!")

    match model_name:
        case "efficientnet_v2_l":
            model = models.efficientnet_v2_l(weights=None)
            model.classifier[1] = nn.Linear(1280, num_classes)

            val_transform = transforms.Compose([transforms.Resize((380,380)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])

            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_efficientnet_v2_l - 92.pth"))
                
        case "VGG16":
            model = VGG16() # Note: Have to initialise the model with 1000 classes so the pretrained weights from torchvision can be loaded
            model.classifier[6] = nn.Linear(4096, num_classes) # Change the output layer to 8 classes
        
            val_transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])

            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_VGG16_256.pth"))
        
        case "VGG19":
            model = VGG19() # Note: Have to initialise the model with 1000 classes so the pretrained weights from torchvision can be loaded
            model.classifier[6] = nn.Linear(4096, num_classes) # Change the output layer to 8 classes
        
            val_transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])

            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_VGG19_256.pth"))
        
        case "ResNet152":
            model = models.resnet152() 
            model.fc = nn.Linear(2048, num_classes)

            val_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])

            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_ResNet152_V2.pth"))
        case "ResNet18":
            model = models.resnet18(weights = "DEFAULT") 
            model.fc = nn.Linear(512, num_classes)

            val_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])
            
            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_ResNet18_V2.pth"))
        
        case "EfficientNetB0":
            model = CustomEfficientNetB0Model(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])
            
            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_EfficientNet_B0_V2.pth"))

        case "MobileNetV4":
            model = CustomMobileNetV4_MH(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])
            
            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_MobileNetV4_MH.pth"))
        
        case "EdgeNeXtXS":
            model = CustomEdgeNextModel(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((288,288)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])
            
            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_EdgeNext_XS_V2_COLD.pth"))

        case "EdgeNeXtS-USI":
            model = CustomEdgeNextUSIModel(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])
        
            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_EdgeNextS_USI_COLD.pth"))
    
        case "EVA02":
            model = CustomViTModel(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((448,448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
                                        ])
            
            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_ViT-EVA02.pth"))
            
        case "TinyViT11M":
            model = CustomTinyViT11mModel(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])
            
            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_TinyViT11m_V1_COLD.pth"))
            
        case "TinyViT21M":
            model = CustomTinyViT21mModel(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])

            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_TinyViT21m_V1_COLD.pth"))

        case "TinyViT5M":
            model = CustomTinyViT5mModel(num_classes=num_classes)

            val_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                                        ])

            if pretrained:
                model.load_state_dict(torch.load("best_model_warmed_acc_TinyViT5m_COLD.pth"))


    return (model, val_transform)
        