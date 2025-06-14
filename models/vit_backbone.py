import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")

class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for feature extraction.
    Outputs a sequence of patch embeddings for each image.
    """
    def __init__(self, model_name="vit_base_patch16_224_in21k", pretrained=True, output_dim=768):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        # Remove the classification head
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        elif hasattr(self.vit, 'fc'):
            self.vit.fc = nn.Identity()
        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            features: Tensor of shape (B, N_patches, output_dim)
        """
        # timm ViT models output (B, N_patches+1, D), first token is [CLS]
        features = self.vit.forward_features(x)
        # Remove [CLS] token if present
        if features.dim() == 3 and features.shape[1] > 1:
            features = features[:, 1:, :]
        return features  # (B, N_patches, output_dim)