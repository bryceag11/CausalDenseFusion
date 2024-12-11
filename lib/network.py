# network.py
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from typing import Dict, Tuple, Optional, List
# Define PSPNet models with different ResNet backbones
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):
    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18']()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    '''
    Purpose: Generates concatenated point-wise features for initial pose estimation.
    Operational Level: Point level
    Concatenated Features: (B, 1408, N)
    '''
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    '''
    Purpose: Estimates initial pose parameters per point per object.
    Operational Level: Point level per object
    out_rx: (num_obj, N, 4)
    out_tx: (num_obj, N, 3)
    out_cx: (num_obj, N, 1)
    emb: (B, D, N)
    '''
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        
        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence
        
        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous() # (B, D, N)
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points) # (B, num_obj, 4, N)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points) # (B, num_obj, 3, N)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points) # (B, num_obj, 1, N)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b]) # (num_obj, 4, N)
        out_tx = torch.index_select(tx[b], 0, obj[b]) # (num_obj, 3, N)
        out_cx = torch.index_select(cx[b], 0, obj[b]) # (num_obj, 1, N)
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous() # (num_obj, N, 4)
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous() # (num_obj, N, 1)
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous() # (num_obj, N, 3)
        
        return out_rx, out_tx, out_cx, emb.detach()

class PoseRefineNetFeat(nn.Module):
    '''
    Purpose: Extracts global features from point cloud and embeddings
    Operational Level: Object level
    ap_x: (B, 1024)
    '''
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1) # (B, 64, N)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1) # (B, 128, N)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1) # (B, 1024, N)

        self.ap1 = torch.nn.AvgPool1d(num_points) # Pooling input (B, 1024, 1)
        self.num_points = num_points

    def forward(self, x, emb):
        # x: (B, 3, N)
        # emb: (B, 32, N)
        x = F.relu(self.conv1(x)) # (B, 64, N)
        emb = F.relu(self.e_conv1(emb)) # Embedding (B, 64, N)
        pointfeat_1 = torch.cat([x, emb], dim=1) # (B, 128, N)

        x = F.relu(self.conv2(x)) # (B, 128, N)
        emb = F.relu(self.e_conv2(emb)) # (B, 128, N)
        pointfeat_2 = torch.cat([x, emb], dim=1) # (B, 256, N)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1) # (B, 384, N)

        x = F.relu(self.conv5(pointfeat_3)) # (B, 512, N)
        x = F.relu(self.conv6(x)) # (B, 1024, N)

        ap_x = self.ap1(x) # Pooling output (B, 1024, 1)

        ap_x = ap_x.view(-1, 1024) # Reshaping (B, 1024) 
        return ap_x # Information across all points so object-level

class PoseRefineNet(nn.Module):
    '''
    Purpose: Refines pose estimates using global object features
    Operational Level: Object level
    out_rx: (1, 4)
    out_tx: (1, 3)
    '''
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        # x: (B, 3, N)
        # emb: (B, 32, N)
        # obj: (B, num_obj)
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous() # (B, N, 3)
        ap_x = self.feat(x, emb) #(B, 1024)

        rx = F.relu(self.conv1_r(ap_x)) # (B, 512)
        tx = F.relu(self.conv1_t(ap_x)) # (B, 512)

        rx = F.relu(self.conv2_r(rx)) # (B, 128)
        tx = F.relu(self.conv2_t(tx)) # (B, 128)

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4) # (B, num_obj, 4) quaternion per object
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3) # (B, num_obj, 3) 

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b]) # (1, 4) single quaternion for selected object
        out_tx = torch.index_select(tx[b], 0, obj[b]) # (1, 3)

        return out_rx, out_tx # (num_obj, 4) (num_obj, 3)


class PoseRefineNetFeat(nn.Module):
    '''
    Purpose: Extracts global features from point cloud and embeddings
    Operational Level: Object level
    ap_x: (B, 1024)
    '''
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1) # (B, 64, N)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1) # (B, 128, N)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1) # (B, 1024, N)

        self.ap1 = torch.nn.AvgPool1d(num_points) # Pooling input (B, 1024, 1)
        self.num_points = num_points

    def forward(self, x, emb):
        # x: (B, 3, N)
        # emb: (B, 32, N)
        x = F.relu(self.conv1(x)) # (B, 64, N)
        emb = F.relu(self.e_conv1(emb)) # Embedding (B, 64, N)
        pointfeat_1 = torch.cat([x, emb], dim=1) # (B, 128, N)

        x = F.relu(self.conv2(x)) # (B, 128, N)
        emb = F.relu(self.e_conv2(emb)) # (B, 128, N)
        pointfeat_2 = torch.cat([x, emb], dim=1) # (B, 256, N)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1) # (B, 384, N)

        x = F.relu(self.conv5(pointfeat_3)) # (B, 512, N)
        x = F.relu(self.conv6(x)) # (B, 1024, N)

        ap_x = self.ap1(x) # Pooling output (B, 1024, 1)

        ap_x = ap_x.view(-1, 1024) # Reshaping (B, 1024) 
        return ap_x # Information across all points so object-level

class PoseRefineNet(nn.Module):
    '''
    Purpose: Refines pose estimates using global object features
    Operational Level: Object level
    out_rx: (1, 4)
    out_tx: (1, 3)
    '''
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        # x: (B, 3, N)
        # emb: (B, 32, N)
        # obj: (B, num_obj)
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous() # (B, N, 3)
        ap_x = self.feat(x, emb) #(B, 1024)

        rx = F.relu(self.conv1_r(ap_x)) # (B, 512)
        tx = F.relu(self.conv1_t(ap_x)) # (B, 512)

        rx = F.relu(self.conv2_r(rx)) # (B, 128)
        tx = F.relu(self.conv2_t(tx)) # (B, 128)

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4) # (B, num_obj, 4) quaternion per object
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3) # (B, num_obj, 3) 

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b]) # (1, 4) single quaternion for selected object
        out_tx = torch.index_select(tx[b], 0, obj[b]) # (1, 3)

        return out_rx, out_tx # (num_obj, 4) (num_obj, 3)


# ---------------------- CausalRefineNet and Associated Classes ----------------------

class GeometricFeatureExtractor(nn.Module):
    """
    Robust geometric feature extractor with input validation
    """
    def __init__(self, num_points: int):
        super(GeometricFeatureExtractor, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False)
        )
        
        self.num_points = num_points

    @staticmethod
    def validate_input(points):
        """Validate input tensor"""
        assert points is not None, "Input points cannot be None"
        assert torch.isfinite(points).all(), "Input points contain inf or nan"
        assert points.is_cuda, "Input points must be on CUDA device"
        return True

    @staticmethod
    def safe_normalize(tensor, dim=-1, eps=1e-8):
        """Safely normalize tensor"""
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True).clamp(min=eps)
        return (tensor - mean) / std

    def forward(self, points):
        # Ensure input is on the same device as the model
        points = points.to(next(self.parameters()).device)
        
        # Input validation
        # self.validate_input(points)
        
        # Shape handling
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if points.shape[1] != 3:
            points = points.transpose(1, 2)

        # Safe normalization
        points_normalized = self.safe_normalize(points, dim=2)
        
        # Forward pass
        with torch.amp.autocast('cuda'):
            features = self.backbone(points_normalized)
            global_feat = torch.max(features, dim=2, keepdim=True)[0]
        
        return {
            'relational': features.contiguous(),
            'local': features.contiguous(),
            'global': global_feat.contiguous()
        }


    # def compute_local_features(self, points):
    #     """Extract local geometric features"""
    #     return self.local_features(points)
        
    # def compute_global_features(self, points):
    #     """Extract global geometric features with proper pooling and channel handling"""
    #     # First stage feature extraction
    #     x = self.global_feat_1(points)  # [B, 64, N]
        
    #     # Global max pooling
    #     x_global = torch.max(x, dim=2, keepdim=True)[0]  # [B, 64, 1]
        
    #     # Second stage feature extraction (without BatchNorm)
    #     x_global = self.global_feat_2(x_global)  # [B, 128, 1]
        
    #     return x_global
        
    # def compute_relational_features(self, local_feat, global_feat):
    #     """Extract relational features"""
    #     combined = torch.cat([local_feat, global_feat.expand(-1, -1, local_feat.size(2))], dim=1)
    #     return self.relational_features(combined)

    # def forward(self, points):
    #     # Ensure points are in [B, 3, N] format
    #     if points.dim() == 2:
    #         points = points.unsqueeze(0)
    #     if points.shape[1] != 3:
    #         points = points.transpose(1, 2)  # Convert from [B, N, 3] to [B, 3, N]
    #     print(f"Geometric FE Points shape: {points.shape}")
    #     local_feat = self.compute_local_features(points)
    #     print(f"Local shape: {local_feat.shape}")

    #     global_feat = self.compute_global_features(points)
    #     print(f"Global shape: {global_feat.shape}")
    #     # Expand global features to match local feature size
    #     global_feat_expanded = global_feat.expand(-1, -1, self.num_points)
    #     print(f"Global Expanded Shape: {global_feat_expanded.shape}")
    #     # Combine features
    #     combined = torch.cat([local_feat, global_feat_expanded], dim=1)
    #     print(f"Combined:{combined.shape}")
    #     relational_feat = self.relational_features(combined)
    #     return {
    #         'local': local_feat,
    #         'global': global_feat,
    #         'relational': relational_feat
    #     }

class SCMFeatureExtractor(nn.Module):
    """
    Feature extraction network that maintains causal relationships
    Separates geometric and appearance features
    """
    def __init__(self, num_points: int):
        super(SCMFeatureExtractor, self).__init__()
        # Geometric feature path (point cloud)
        self.geo_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.geo_conv2 = torch.nn.Conv1d(64, 128, 1)
        
        # Appearance feature path (embeddings)
        self.app_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.app_conv2 = torch.nn.Conv1d(64, 128, 1)
        
        # Combined feature processing
        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        
        self.ap1 = torch.nn.AvgPool1d(num_points)
        
    def forward(self, points: torch.Tensor, emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract causally separated features
        
        Args:
            points: (B, 3, N) Point cloud
            emb: (B, 32, N) Image embeddings
            
        Returns:
            Dictionary containing:
            - geometric_features: Features from point cloud
            - appearance_features: Features from embeddings
            - combined_features: Joint features
        """
        # Geometric path
        geo_f1 = F.relu(self.geo_conv1(points))
        geo_f2 = F.relu(self.geo_conv2(geo_f1))
        
        # Appearance path  
        app_f1 = F.relu(self.app_conv1(emb))
        app_f2 = F.relu(self.app_conv2(app_f1))
        
        # Early fusion features for backdoor paths
        fusion1 = torch.cat((geo_f1, app_f1), dim=1)
        fusion2 = torch.cat((geo_f2, app_f2), dim=1)
        
        # Combined feature processing
        combined = torch.cat((fusion1, fusion2), dim=1)
        x = F.relu(self.conv5(combined))
        x = F.relu(self.conv6(x))
        pooled = self.ap1(x)
        
        return {
            'geometric': geo_f2,
            'appearance': app_f2,
            'early_fusion': fusion2,
            'combined': pooled.view(-1, 1024)
        }

class CausalMechanism(nn.Module):
    """
    Implements atomic causal mechanisms for rotation and translation
    """
    def __init__(self, in_dim: int, out_dim: int):
        super(CausalMechanism, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)
        
        # Causal mechanism specific noise
        self.noise_scale = nn.Parameter(torch.ones(out_dim))
        
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply causal mechanism with optional noise for interventions
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        if noise is not None:
            x = x + self.noise_scale * noise
            
        return x

class SCMPoseRefiner(nn.Module):
    """
    SCM-based pose refinement network implementing the structural equations
    f1: P -> G (Geometric Features)
    f2,f3: G,θ0 -> R (Residuals) 
    f4,f5: R,E -> δθ (Pose Update)
    f6: θ0,δθ -> θ (Final Pose)
    """
    def __init__(self, num_points: int, num_obj: int):
        super(SCMPoseRefiner, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        
        # f1: Geometric Feature Extraction
        self.geometric_features = GeometricFeatureExtractor(num_points)
        
        # f2,f3: Residual Computation 
        self.residual_net = nn.Sequential(
            nn.Linear(128 + 7, 512),  # 128 from geo features + 7 for pose (quat + trans)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # f4,f5: Pose Update Generation
        self.update_net = nn.Sequential(
            nn.Linear(128 + 32, 512),  # 128 from residuals + 1024 from embedding
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # quaternion + translation update
        )

    def do_intervention(self, points: torch.Tensor, intervention_type: str,
                    intervention_value: torch.Tensor) -> torch.Tensor:
        """Apply structural interventions"""
        if intervention_type == 'view':
            # Ensure points are properly batched
            if points.dim() == 2:
                points = points.unsqueeze(0)
            
            if points.shape[1] != 3:
                points = points.transpose(1, 2)  # Convert from [B, N, 3] to [B, 3, N]

            # Create homogeneous coordinates by adding 1s as fourth coordinate
            batch_size, num_dim, num_points = points.size()
            ones = torch.ones(batch_size, 1, num_points).to(points.device)
            homogeneous_points = torch.cat([points, ones], dim=1)  # Shape: [B, 4, N]

            # Ensure intervention_value is properly shaped - should be [B, 4, 4]
            if intervention_value.dim() == 2:
                intervention_value = intervention_value.unsqueeze(0)
                print("Intervention dim of 2")
                print(intervention_value.shape)

            if intervention_value.size(0) != batch_size:
                intervention_value = intervention_value.expand(batch_size, -1, -1)
                print("Intervention size does not match the batch size")
                print(intervention_value.shape)

            # Apply transformation
            # intervention_value: [B, 4, 4]
            # homogeneous_points: [B, 4, N]
            # Result will be: [B, 4, N]
            transformed_points = torch.bmm(intervention_value, homogeneous_points)

            # Return only the spatial coordinates
            return transformed_points[:, :3, :]
            
        elif intervention_type == 'symmetry':
            if points.dim() == 2:
                points = points.unsqueeze(0)
            if points.shape[1] == 3:  # If points are [B, 3, N]
                points = points.transpose(1, 2)  # Convert to [B, N, 3]
            print(f"points: {points.shape}")
            if intervention_value.dim() == 2:
                intervention_value = intervention_value.unsqueeze(0)
                print(f"Intervention Value: {intervention_value.shape}")
                
            if intervention_value.size(0) != points.size(0):
                intervention_value = intervention_value.expand(points.size(0), -1, -1)
                print(f"Intervention Value: {intervention_value.shape}")

            # Perform transformation and restore original shape
            transformed = torch.bmm(points, intervention_value)  # [B, N, 3]
            return transformed.transpose(1, 2)  # Return to [B, 3, N]
            

    def compute_residuals(self, geo_features: torch.Tensor, 
                         initial_pose: torch.Tensor) -> torch.Tensor:
        """f2,f3: Compute geometric residuals"""


        # Average across points to get global features
        geo_global = geo_features.mean(dim=2)  # [B, 128]
        pose_global = initial_pose.mean(dim=2)  # [B, 7]
        
        # Now both are 2D tensors that can be concatenated
        combined = torch.cat([geo_global, pose_global], dim=1)  # [B, 135]
        return self.residual_net(combined)
        
    def generate_update(self, residuals: torch.Tensor, 
                       embedding: torch.Tensor) -> torch.Tensor:
        """f4,f5: Generate pose update"""
        embedding_global = embedding.mean(dim=2)

        combined = torch.cat([residuals, embedding_global], dim=1)
        return self.update_net(combined)
        
    def compose_pose(self, initial_pose: torch.Tensor, 
                    pose_update: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compose initial pose with update
        Args:
            initial_pose: Shape [B, 7, N] or [B, 7]
            pose_update: Shape [B, 7]
        Returns:
            final_r: Updated rotation
            final_t: Updated translation 
        """
        # Handle case where initial pose has point dimension
        if len(initial_pose.shape) == 3:
            # Get the mean over points
            initial_pose = initial_pose.mean(dim=2)  # Now [B, 7]

        init_r = initial_pose[:, :4]
        init_t = initial_pose[:, 4:]
        update_r = pose_update[:, :4]
        update_t = pose_update[:, 4:]


        # Normalize quaternions
        init_r = init_r / torch.norm(init_r, dim=1, keepdim=True)
        update_r = update_r / torch.norm(update_r, dim=1, keepdim=True)

        # Modified quaternion multiplication to handle batches
        def quaternion_multiply(q1, q2):
            """
            Multiply two quaternions with batch support
            q1, q2 shape: [B, 4]
            """
            w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
            
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            
            return torch.stack([w, x, y, z], dim=1)  # [B, 4]
        
        # Compose rotations and translations
        final_r = quaternion_multiply(init_r, update_r)  # [B, 4]
        final_t = init_t + update_t  # [B, 3]
        
        return final_r, final_t
    def forward(self, points: torch.Tensor, emb: torch.Tensor, 
                initial_pose: torch.Tensor, obj: torch.Tensor,
                interventions: Optional[Dict[str, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing the SCM causal mechanisms
        
        Args:
            points: (B, 3, N) Point cloud
            emb: (B, D) Global embedding from DenseFusion
            initial_pose: (B, 7) Initial pose estimate (quaternion + translation)
            obj: Object indices
            interventions: Optional intervention values
        
        Returns:
            pred_r: (1, 4) Final rotation (quaternion)
            pred_t: (1, 3) Final translation
        """
        bs = points.size()[0]
        
        # Apply interventions if specified
        if interventions is not None:
            for intervention_type, value in interventions.items():
                points = self.do_intervention(points, intervention_type, value)
        
        # f1: Extract geometric features
        geo_features = self.geometric_features(points)['relational']
        
        # f2,f3: Compute residuals

        residuals = self.compute_residuals(geo_features, initial_pose)
        

        # f4,f5: Generate pose update
        pose_update = self.generate_update(residuals, emb)
        

        # f6: Compose final pose
        pred_r, pred_t = self.compose_pose(initial_pose, pose_update)
        
        # Select object-specific predictions 
        b = 0
        pred_r = torch.index_select(pred_r[b].view(1, 4), 0, obj[b])
        pred_t = torch.index_select(pred_t[b].view(1, 3), 0, obj[b])
        
        return pred_r, pred_t

