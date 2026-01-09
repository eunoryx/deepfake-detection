import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torchvision import models


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphReasoningBlock(nn.Module):
    """
    Modified to work with patch-based input [batch, num_patches, channels]
    """
    def __init__(self, in_channels, inter_channels, dropout=0.1):
        super(GraphReasoningBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        # Linear projections instead of Conv2d
        self.theta = nn.Linear(in_channels, inter_channels)
        self.phi = nn.Linear(in_channels, inter_channels)
        self.g = nn.Linear(in_channels, inter_channels)
        
        # Output projection
        self.out_proj = nn.Linear(inter_channels, in_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj=None):
        """
        x: Input features [batch_size, num_patches, channels]
        adj: Optional adjacency matrix [batch_size, num_patches, num_patches]
        """
        batch_size, N, C = x.size()
        
        # Project features
        theta = self.theta(x)  # [B, N, inter_channels]
        phi = self.phi(x)      # [B, N, inter_channels]
        g = self.g(x)         # [B, N, inter_channels]
        
        # Calculate attention
        attention = torch.bmm(theta, phi.permute(0, 2, 1))  # [B, N, N]
        
        # Use adjacency matrix if provided
        if adj is not None:
            attention = attention * adj
        
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention
        out = torch.bmm(attention, g)  # [B, N, inter_channels]
        
        # Project back to original dimension
        out = self.out_proj(out)
        
        # Add residual connection
        out = out + x
        
        return out

class ResNetThumbnailModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(ResNetThumbnailModel, self).__init__()
        self.device = device
        
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze all parameters except the last 8 layers
        for param in list(self.features.parameters())[:-8]:
            param.requires_grad = False

        self.feature_dim = 512
        self.reduced_dim = 64  # Make sure this matches throughout
        self.dim_reduction = nn.Conv2d(self.feature_dim, self.reduced_dim, kernel_size=1)

        self.graph_attention = GraphReasoningBlock(self.reduced_dim, 32, dropout=0.5)
        
        # Updated classifier to match reduced_dim (64) -> 16 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim, 16),  # Input dim matches reduced_dim (64)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )
        
        # self.proj_layer = nn.Linear(self.reduced_dim, 32)
        
        # Buffers for consistency loss
        self.register_buffer('proj_original', None)
        self.register_buffer('graph_features', None)
        self.stored_features = None
        self.current_consistency_loss = None

    def extract_patches(self, x):
        batch_size = x.size(0)
        
        # Use autocast but ensure dtype consistency
        with torch.amp.autocast(device_type=self.device, enabled=self.device.startswith('cuda')):
            features = self.features(x)
            features = self.dim_reduction(features)  # [batch, reduced_dim, H, W]
        
        # Convert to float32 for graph attention layers
        features = features.float()
        
        _, fc, fh, fw = features.shape
        patches = features.view(batch_size, fc, -1).permute(0, 2, 1)  # [batch, N, reduced_dim]
        adj = torch.ones(batch_size, patches.size(1), patches.size(1), device=x.device)
        
        return patches, adj

    def forward(self, x):
        patches, adj = self.extract_patches(x)
        
        # Process through graph attention
        graph_features = self.graph_attention(patches, adj)  # [batch, N, reduced_dim]
        
        # Average pooling across patches
        pooled_features = torch.mean(graph_features, dim=1)  # [batch, reduced_dim]
        
        # Classifier
        logits = self.classifier(pooled_features)  # Now dimensions match
        
        # Consistency loss components
        original_features = patches.detach()
        self.proj_original = original_features
        self.graph_features = graph_features
        self.stored_features = original_features
        
        # Compute consistency loss
        consistency_loss = self.get_semantic_consistency_loss(original_features, graph_features)
        self.current_consistency_loss = consistency_loss
        
        return logits

    

    def get_consistency_loss(self):
        """
        Returns the most recently computed semantic consistency loss
        """
        if self.current_consistency_loss is not None:
            return self.current_consistency_loss
        else:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

    def get_semantic_consistency_loss(self, feat1, feat2):
        """
        Computes semantic consistency loss between two feature sets.
        Inputs are expected to be normalized feature tensors of shape [batch, N, C].
        """
        norm1 = F.normalize(feat1, dim=-1)
        norm2 = F.normalize(feat2, dim=-1)
        return F.mse_loss(norm1, norm2)

    def compute_class_weights(self, train_dataset):
        # Compute class weights for imbalanced datasets
        labels = [label for _, label in train_dataset]
        label_counts = Counter(labels)
        
        total_samples = sum(label_counts.values())
        classes = sorted(label_counts.keys())
        
        weights = [total_samples / label_counts[cls] for cls in classes]
        return torch.tensor(weights, dtype=torch.float, device=self.device)