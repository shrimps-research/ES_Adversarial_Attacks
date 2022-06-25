import numpy as np
import torch


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