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
    f1: P -> G - Extract geometric features from point cloud
    """
    def __init__(self, num_points: int):
        super().__init__()
        
        # Point-wise feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Global feature aggregation
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: [B, 3, N] or [B, N, 3]
        Returns:
            Dictionary with 'local' and 'global' features
        """
        # Ensure [B, 3, N] format
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if points.shape[1] != 3:
            points = points.transpose(1, 2)
            
        # Extract point-wise features
        x = F.relu(self.bn1(self.conv1(points)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))       # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))       # [B, 256, N]
        
        # Global pooling
        global_feat = torch.max(x, dim=2)[0]      # [B, 256]
        global_feat = F.relu(self.fc1(global_feat))  # [B, 512]
        global_feat = self.fc2(global_feat)      # [B, 256]
        
        return {
            'local': x,                           # [B, 256, N]
            'global': global_feat                 # [B, 256]
        }

class ResidualComputer(nn.Module):
    """
    f2,f3: (G, θ0) -> R - Compute residuals from features and initial pose
    """
    def __init__(self):
        super().__init__()
        
        # Combine geometric features with initial pose
        self.fc1 = nn.Linear(256 + 7, 512)  # 256 from features, 7 from pose
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, geo_features: torch.Tensor, initial_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geo_features: [B, 256] global geometric features
            initial_pose: [B, 7] initial pose (quaternion + translation)
        Returns:
            residuals: [B, 128]
        """
        combined = torch.cat([geo_features, initial_pose], dim=1)  # [B, 263]
        
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        residuals = self.fc3(x)  # [B, 128]
        
        return residuals

class PoseUpdateGenerator(nn.Module):
    """
    f4,f5: (R, E) -> δθ - Generate pose update from residuals and embeddings
    """
    def __init__(self):
        super().__init__()
        
        # Combine residuals with appearance embeddings
        self.fc1 = nn.Linear(128 + 1024, 512)  # 128 residuals + 1024 embeddings
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 7)  # Output pose update
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, residuals: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residuals: [B, 128] from residual computer
            embeddings: [B, 1024] from DenseFusion CNN
        Returns:
            pose_update: [B, 7] (quaternion + translation update)
        """
        combined = torch.cat([residuals, embeddings], dim=1)  # [B, 1152]
        
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        pose_update = self.fc4(x)  # [B, 7]
        
        return pose_update

class SCMPoseRefiner(nn.Module):
    """
    Fixed SCM-based pose refinement implementing proper causal mechanisms
    """
    def __init__(self, num_points: int, num_obj: int):
        super().__init__()
        
        self.num_points = num_points
        self.num_obj = num_obj
        
        # f1: Geometric feature extraction
        self.geometric_extractor = GeometricFeatureExtractor(num_points)
        
        # f2,f3: Residual computation
        self.residual_computer = ResidualComputer()
        
        # f4,f5: Pose update generation
        self.update_generator = PoseUpdateGenerator()
        
        # Intervention parameters
        self.view_invariance_weight = nn.Parameter(torch.ones(1))
        self.symmetry_weight = nn.Parameter(torch.ones(1))
        
    def aggregate_pose_predictions(self, pred_r: torch.Tensor, pred_t: torch.Tensor, 
                                  pred_c: torch.Tensor) -> torch.Tensor:
        """
        Aggregate per-point predictions into single pose using confidence weighting
        Args:
            pred_r: [B, N, 4] per-point rotation predictions
            pred_t: [B, N, 3] per-point translation predictions
            pred_c: [B, N, 1] per-point confidence scores
        Returns:
            initial_pose: [B, 7] aggregated pose
        """
        # Normalize confidences
        pred_c = F.softmax(pred_c.squeeze(-1), dim=1)  # [B, N]
        
        # Weighted average of rotations
        weighted_r = torch.sum(pred_r * pred_c.unsqueeze(-1), dim=1)  # [B, 4]
        weighted_r = F.normalize(weighted_r, dim=1)  # Normalize quaternion
        
        # Weighted average of translations
        weighted_t = torch.sum(pred_t * pred_c.unsqueeze(-1), dim=1)  # [B, 3]
        
        return torch.cat([weighted_r, weighted_t], dim=1)  # [B, 7]
    
    def apply_intervention(self, features: Dict[str, torch.Tensor], 
                          intervention_type: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply interventions to the causal mechanism, not the raw data
        """
        if intervention_type == 'view':
            # View intervention affects how features are processed
            features['global'] = features['global'] * self.view_invariance_weight
            
        elif intervention_type == 'symmetry':
            # Symmetry intervention modulates feature responses
            features['global'] = features['global'] * self.symmetry_weight
            
        return features
    
    @staticmethod
    def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (batch support)"""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=1)
    
    def compose_pose(self, initial_pose: torch.Tensor, pose_update: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f6: (θ0, δθ) -> θ - Compose initial pose with update
        """
        # Split poses
        init_r = initial_pose[:, :4]
        init_t = initial_pose[:, 4:]
        update_r = pose_update[:, :4]
        update_t = pose_update[:, 4:]
        
        # Normalize quaternions
        init_r = F.normalize(init_r, dim=1)
        update_r = F.normalize(update_r, dim=1)
        
        # Compose rotations via quaternion multiplication
        final_r = self.quaternion_multiply(init_r, update_r)
        final_r = F.normalize(final_r, dim=1)
        
        # Add translations
        final_t = init_t + update_t
        
        return final_r, final_t
    
    def forward(self, points: torch.Tensor, emb: torch.Tensor,
                pred_r: torch.Tensor, pred_t: torch.Tensor, pred_c: torch.Tensor,
                obj: torch.Tensor, interventions: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass implementing the SCM
        Args:
            points: [B, 3, N] point cloud
            emb: [B, 32, N] embeddings from DenseFusion
            pred_r, pred_t, pred_c: Per-point predictions from PoseNet
            obj: Object indices
            interventions: Optional interventions to apply
        Returns:
            pred_r: [1, 4] refined rotation
            pred_t: [1, 3] refined translation
            features: Dictionary of intermediate features for loss computation
        """
        bs = points.size(0)
        
        # Aggregate per-point predictions into initial pose
        initial_pose = self.aggregate_pose_predictions(pred_r, pred_t, pred_c)  # [B, 7]
        
        # f1: Extract geometric features
        geo_features = self.geometric_extractor(points)
        
        # Apply interventions to features if specified
        if interventions is not None:
            for int_type, value in interventions.items():
                geo_features = self.apply_intervention(geo_features, int_type, value)
        
        # f2,f3: Compute residuals
        residuals = self.residual_computer(geo_features['global'], initial_pose)
        
        # Process embeddings to get global appearance features
        if emb.dim() == 3:  # [B, 32, N]
            emb_global = torch.max(emb, dim=2)[0]  # [B, 32]
            # Project to expected dimension
            emb_proj = nn.Linear(32, 1024).cuda()(emb_global)  # [B, 1024]
        else:
            emb_proj = emb
        
        # f4,f5: Generate pose update
        pose_update = self.update_generator(residuals, emb_proj)
        
        # f6: Compose final pose
        final_r, final_t = self.compose_pose(initial_pose, pose_update)
        
        # Select object-specific predictions
        b = 0
        obj_idx = obj[b].clamp(0, final_r.size(0) - 1)
        pred_r = final_r[b].unsqueeze(0)
        pred_t = final_t[b].unsqueeze(0)
        
        # Prepare features for loss computation
        features = {
            'geometric': geo_features,
            'residuals': residuals,
            'pose_update': pose_update,
            'initial_pose': initial_pose
        }
        
        return pred_r, pred_t, features
