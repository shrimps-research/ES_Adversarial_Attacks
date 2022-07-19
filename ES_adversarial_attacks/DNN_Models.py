import numpy as np
import torch
from PIL import Image

class VGGClassifier(torch.nn.Module):
    def __init__(self):
        super(VGGClassifier, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        # get the pretrained VGG19 network
        self.weights = VGG19_Weights.IMAGENET1K_V1
        self.model = vgg19(weights=self.weights)
        self.model.eval()
    
        # model preprocessing for images
        self.transforms = self.weights.transforms()

    def __call__(self, in_arr, device):

        with torch.no_grad():
            x = torch.tensor(in_arr, dtype=torch.float32).permute((0,3,1,2))
            logits = self.model(x.to(device))

            # print("ret:",torch.nn.functional.softmax(logits, dim=1).argmax())
            return torch.nn.functional.softmax(logits, dim=1)


class XceptionTorch(torch.nn.Module):
    def __init__(self):
        super(XceptionTorch, self).__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        # get the pretrained VGG19 network
        self.weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=self.weights)
        self.model.eval()
    
        # model preprocessing for images
        self.transforms = self.weights.transforms()

    def __call__(self, in_arr, device):
        with torch.no_grad():
            x = torch.tensor(in_arr, dtype=torch.float32).permute((0,3,1,2))
            logits = self.model(x.to(device))

            return torch.nn.functional.softmax(logits, dim=1)


class XceptionClassifier:
    def __init__(self):
        import timm
        self.model = timm.create_model('xception', pretrained=True).float()
        self.model.eval()
    
    def __call__(self, x, device):
        if len(x.shape) == 3:
            # transpose dims from HxWxC to CxHxW
            x = np.transpose(x, (2, 0, 1))
            # add batch dim
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 4:
            # transpose dims from BxHxWxC to BxCxHxW
            x = np.transpose(x, (0, 3, 1, 2))
        
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32).to(device))
            return torch.nn.functional.softmax(logits, dim=1)


class ViTClassifier:
    def __init__(self):
        import timm
        self.model = timm.create_model('vit_base_patch16_224', 
                                        pretrained=True,
                                        num_classes=1000).float()
        self.model.eval()
    
    def __call__(self, x, device):
        if len(x.shape) == 3:
            # transpose dims from HxWxC to CxHxW
            x = np.transpose(x, (2, 0, 1))
            # add batch dim
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 4:
            # transpose dims from BxHxWxC to BxCxHxW
            x = np.transpose(x, (0, 3, 1, 2))
        
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32).to(device))
            return torch.nn.functional.softmax(logits, dim=1)


class PerceiverClassifier:
    def __init__(self):
        from transformers import PerceiverFeatureExtractor, PerceiverForImageClassificationLearned
        self.feature_extractor = PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-learned")
        self.model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
        self.model.eval()
    
    def __call__(self, x, device):
        if len(x.shape) == 3:
            # transpose dims from HxWxC to CxHxW
            x = np.transpose(x, (2, 0, 1))
            # add batch dim
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 4:
            # transpose dims from BxHxWxC to BxCxHxW
            x = np.transpose(x, (0, 3, 1, 2))

        # prepare input
        encoding = self.feature_extractor(list(x), return_tensors="pt")
        inputs = encoding.pixel_values
        # forward pass
        with torch.no_grad():
            outputs = self.model(inputs.to(device))
            return torch.nn.Softmax(dim=1)(outputs.logits).detach()