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

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
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
        emb = torch.gather(emb, 2, choose).contiguous()
        
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

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        return out_rx, out_tx, out_cx, emb.detach()
 


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
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
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx



class CausalRefineNet(nn.Module):
    def __init__(self, num_points, num_obj, max_rotation_angle=np.pi/6, translation_min=-0.5, translation_max=0.5, num_iterations=2):
        super(CausalRefineNet, self).__init__()
        self.num_points = num_points
        #self.num_obj = num_obj
        self.num_iterations = num_iterations 
        
        # SCM components
        self.visibility_estimator = VisibilityModule()
        self.geometric_analyzer = GeometricModule()
        self.pose_residual = ResidualModule()
        self.confidence_updater = ConfidenceModule()

        # Define translation bounds (example values)
        self.translation_min = translation_min  # meters
        self.translation_max = translation_max  # meters
        
        # Define maximum rotation angle per iteration
        self.max_rotation_angle = max_rotation_angle  # radians,e.g., np.pi / 6 
        
    def forward(self, point_cloud, prev_pose, prev_confidence):
        """
         Args:
            point_cloud: (B, N, 3) transformed point cloud
            prev_pose: (B, 7) pose [quat + trans]
            prev_confidence: (B, 1) previous confidence score
        Returns:
            current_pose: (B, 7) refined pose
            current_conf: (B, 1) updated confidence score
        """
        batch_size = point_cloud.size(0)
        
        # Current iteration's estimate
        current_pose = prev_pose
        current_conf = prev_confidence

        for iteration in range(self.num_iterations):
            # 1. Estimate visibility of points
            vis_mask = self.visibility_estimator(point_cloud, current_pose)
            
            # 2. Extract geometric features from visible points
            geom_features = self.geometric_analyzer(point_cloud, vis_mask)
            
            # 3. Estimate pose residual using SCM
            pose_residual = self.pose_residual(geom_features, current_pose)
            
            # 4. Apply physical constraints and update pose
            current_pose = self.apply_constraints(current_pose + pose_residual)
            
            # 5. Update confidence based on visibility and residual
            current_conf = self.confidence_updater(vis_mask, pose_residual, current_conf)

        return current_pose, current_conf

    def apply_constraints(self, pose):
        """Apply physical constraints to ensure valid pose"""
        # Implement rotation matrix orthogonality constraints
        # Implement translation bounds
        # Add any object-specific constraints
      
        quat = pose[:, :4]
        trans = pose[:, 4:]

        # Normalize quaternions
        quat = self.normalize_quaternion(quat)

        # Limit rotation magnitude
        quat = self.limit_rotation(quat, max_angle=self.max_rotation_angle)

        # Limit translation
        trans = self.limit_translation(trans, 
                                  min_val=self.translation_min, 
                                  max_val=self.translation_max)

        # Combine constrained quaternion and translation
        constrained_pose = torch.cat([quat, trans], dim=1)

        return constrained_pose

    def normalize_quaternion(self,quaternion):
        """Normalize quaternions to unit length."""
        return F.normalize(quaternion, p=2, dim=1)

    def quaternion_to_angle(self,quaternion):
        """Convert quaternion to rotation angle in radians."""
        # Quaternion format: [x, y, z, w]
        # Angle = 2 * acos(w)
        w = quaternion[:, 3]
        angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
        return angle

    def limit_rotation(self, quaternion, max_angle):
        """
        Limit the rotation angle represented by the quaternion to max_angle radians.
        Args:
            quaternion: (B, 4) normalized quaternion [x, y, z, w]
            max_angle: maximum allowable rotation angle in radians
        Returns:
            limited_quaternion: (B, 4) limited quaternion
        """
        angle = self.quaternion_to_angle(quaternion)
        # Avoid division by zero
        angle = torch.clamp(angle, min=1e-6)
        # Compute scaling factor for angle
        scale = torch.ones_like(angle)
        mask = angle > max_angle
        scale[mask] = max_angle / angle[mask]
        # Compute new quaternion
        xyz = quaternion[:, :3] * scale.unsqueeze(1)
        w = torch.clamp(quaternion[:, 3], -1.0, 1.0)
        # Re-normalize after scaling
        limited_quaternion = torch.cat([xyz, w.unsqueeze(1)], dim=1)
        limited_quaternion = normalize_quaternion(limited_quaternion)
        return limited_quaternion

    def limit_translation(self,translation, min_val, max_val):
        """
        Clamp the translation vectors to be within [min_val, max_val] along each axis.
        Args:
            translation: (B, 3) translation vectors
            min_val: minimum translation value
            max_val: maximum translation value
        Returns:
            limited_translation: (B, 3) clamped translation vectors
        """
        return torch.clamp(translation, min=min_val, max=max_val)


class VisibilityModule(nn.Module):
    def __init__(self, threshold=0.02):
        super(VisibilityModule, self).__init__()
        self.threshold = threshold  # Depth difference threshold in meters

    def forward(self, point_cloud, pose):
        """
        Estimate point visibility given current pose
        Args:
            point_cloud: (B, N, 3) transformed point cloud
            pose: (B, 7) pose [quat + trans]
        Returns:
            vis_mask: (B, N) boolean visibility mask
        """
        batch_size, num_points, _ = point_cloud.shape
        
        # Convert pose to transformation matrix
        R = quaternion_to_matrix(pose[:, :4])
        t = pose[:, 4:]
        
        # Transform points to camera frame
        transformed_points = torch.bmm(point_cloud, R.transpose(1, 2)) + t.unsqueeze(1)
        
        # Project points to 2D
        projected_points = project_to_2d(transformed_points)  # Implement camera projection
        
        # Compute depth map from projected points
        depth_map = compute_depth_map(projected_points, transformed_points)
        
        # Check depth consistency
        point_depths = transformed_points[..., 2]
        depth_diffs = torch.abs(depth_map - point_depths)
        vis_mask = depth_diffs < self.threshold
        
        return vis_mask

class GeometricModule(nn.Module):
    def __init__(self, k_neighbors=20):
        super(GeometricModule, self).__init__()
        self.k = k_neighbors
        
        # Local geometric feature extraction
        self.feat_extraction = nn.Sequential(
            nn.Conv1d(9, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )

    def forward(self, point_cloud, vis_mask):
        """
        Extract geometric features from visible points
        Args:
            point_cloud: (B, N, 3) point cloud
            vis_mask: (B, N) visibility mask
        Returns:
            geom_features: Geometric features
        """
        batch_size, num_points, _ = point_cloud.shape
        
        # Only process visible points
        visible_points = point_cloud[vis_mask].reshape(batch_size, -1, 3)
        
        # Compute local neighborhood features
        knn_idx = find_knn(visible_points, k=self.k)
        local_features = compute_local_features(visible_points, knn_idx)
        
        # Extract geometric properties
        normals = estimate_normals(visible_points, knn_idx)
        curvature = estimate_curvature(visible_points, knn_idx)
        
        # Combine features
        combined_features = torch.cat([
            visible_points,
            normals,
            curvature,
            local_features
        ], dim=-1)
        
        # Extract final geometric features
        geom_features = self.feat_extraction(combined_features.transpose(1, 2))
        
        return geom_features

class ResidualModule(nn.Module):
    def __init__(self):
        super(ResidualModule, self).__init__()
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Rotation residual
        self.rot_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # quaternion
        )
        
        # Translation residual
        self.trans_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, geom_features, current_pose):
        """
        Estimate pose residual from geometric features
        Args:
            geom_features: Extracted geometric features
            current_pose: (B, 7) current pose estimate [quat + trans]
        Returns:
            pose_residual: (B, 7) pose residual
        """
        # Process geometric features
        features = self.feature_net(geom_features)
        
        # Estimate rotation and translation residuals
        rot_residual = self.rot_net(features)
        rot_residual = normalize_quaternion(rot_residual)
        trans_residual = self.trans_net(features)
        
        # Combine residuals
        pose_residual = torch.cat([rot_residual, trans_residual], dim=1)
        
        # Apply physical constraints
        pose_residual = self.constrain_residual(pose_residual, current_pose)
        
        return pose_residual
    
    def constrain_residual(self, residual, current_pose):
        """Apply physical constraints to residual"""
        # Limit rotation magnitude
        rot_residual = residual[:, :4]
        rot_magnitude = quaternion_magnitude(rot_residual)
        rot_residual = clip_rotation(rot_residual, max_angle=np.pi/6)
        
        # Limit translation magnitude
        trans_residual = residual[:, 4:]
        trans_residual = torch.clamp(trans_residual, -0.1, 0.1)  # 10cm limit
        
        return torch.cat([rot_residual, trans_residual], dim=1)

class ConfidenceModule(nn.Module):
    def __init__(self):
        super(ConfidenceModule, self).__init__()
        
        self.confidence_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, vis_mask, pose_residual, prev_conf):
        """
        Update confidence based on visibility and residual
        Args:
            vis_mask: (B, N) visibility mask
            pose_residual: (B, 7) pose residual
            prev_conf: (B, 1) previous confidence
        Returns:
            new_confidence: (B, 1) updated confidence
        """
        # Compute visibility ratio
        vis_ratio = vis_mask.float().mean(dim=1, keepdim=True)
        
        # Compute residual magnitude
        rot_magnitude = quaternion_magnitude(pose_residual[:, :4])
        trans_magnitude = torch.norm(pose_residual[:, 4:], dim=1, keepdim=True)
        
        # Combine factors for confidence update
        confidence_factors = torch.cat([
            vis_ratio,
            rot_magnitude,
            trans_magnitude,
            prev_conf
        ], dim=1)
        
        # Estimate new confidence
        new_confidence = self.confidence_net(confidence_factors)
        
        return new_confidence

import torch

def quaternion_to_matrix(quaternion):
    """
    Convert quaternion to a 3x3 rotation matrix.
    Args:
        quaternion: Tensor of shape (B, 4), where each quaternion is [x, y, z, w].
    Returns:
        Tensor of shape (B, 3, 3), representing the corresponding rotation matrices.
    """
    x, y, z, w = quaternion.unbind(dim=-1)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=-1).view(-1, 3, 3)

    return matrix

def project_to_2d(points_3d, intrinsics=None):
    """
    Project 3D points to 2D using camera intrinsics.
    Args:
        points_3d: Tensor of shape (B, N, 3).
        intrinsics: Camera intrinsic matrix of shape (3, 3).
                    Defaults to a normalized pinhole camera model if not provided.
    Returns:
        Tensor of shape (B, N, 2), representing 2D projections.
    """
    if intrinsics is None:
        intrinsics = torch.tensor([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]], dtype=points_3d.dtype, device=points_3d.device)
    
    points_3d_homo = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)  # Homogeneous coordinates
    projected_points = torch.bmm(points_3d_homo, intrinsics.t())
    projected_points = projected_points[..., :2] / projected_points[..., 2:3]  # Normalize by z
    return projected_points

def compute_depth_map(projected_points, points_3d):
    """
    Compute depth map from projected points.
    Args:
        projected_points: Tensor of shape (B, N, 2), 2D projections of points.
        points_3d: Tensor of shape (B, N, 3), original 3D points.
    Returns:
        Depth map of shape (B, N).
    """
    return points_3d[..., 2]  # The z-coordinate represents depth.

def find_knn(points, k):
    """
    Find k nearest neighbors for each point.
    Args:
        points: Tensor of shape (B, N, 3).
        k: Number of nearest neighbors to find.
    Returns:
        Indices of k nearest neighbors, of shape (B, N, k).
    """
    dists = torch.cdist(points, points)  # Compute pairwise distances
    knn_idx = dists.topk(k, largest=False).indices  # Indices of the k nearest neighbors
    return knn_idx

def compute_local_features(points, knn_idx):
    """
    Compute local neighborhood features.
    Args:
        points: Tensor of shape (B, N, 3).
        knn_idx: Indices of k nearest neighbors, of shape (B, N, k).
    Returns:
        Local features tensor of shape (B, N, k, 3).
    """
    batch_size, num_points, _ = points.shape
    k = knn_idx.shape[-1]
    neighbors = points.unsqueeze(2).expand(batch_size, num_points, k, -1)
    local_features = neighbors.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
    return local_features

def estimate_normals(points, knn_idx):
    """
    Estimate surface normals for points.
    Args:
        points: Tensor of shape (B, N, 3).
        knn_idx: Indices of k nearest neighbors, of shape (B, N, k).
    Returns:
        Normals tensor of shape (B, N, 3).
    """
    local_points = compute_local_features(points, knn_idx)
    pca_centroids = local_points.mean(dim=2, keepdim=True)
    centered = local_points - pca_centroids
    covariance_matrix = torch.einsum('bnik,bnij->bnkj', centered, centered)
    _, _, vh = torch.linalg.svd(covariance_matrix)
    normals = vh[..., -1]  # Smallest singular value's vector
    return normals

def estimate_curvature(points, knn_idx):
    """
    Estimate local curvature of points.
    Args:
        points: Tensor of shape (B, N, 3).
        knn_idx: Indices of k nearest neighbors, of shape (B, N, k).
    Returns:
        Curvature tensor of shape (B, N, 1).
    """
    local_points = compute_local_features(points, knn_idx)
    distances = torch.norm(local_points - points.unsqueeze(2), dim=-1)
    curvature = distances.mean(dim=2, keepdim=True)  # Approximate curvature by mean distance
    return curvature

def normalize_quaternion(quaternion):
    """
    Normalize quaternion to unit length.
    Args:
        quaternion: Tensor of shape (B, 4).
    Returns:
        Normalized quaternion of shape (B, 4).
    """
    return quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

def quaternion_magnitude(quaternion):
    """
    Compute quaternion rotation magnitude.
    Args:
        quaternion: Tensor of shape (B, 4).
    Returns:
        Magnitude of the quaternion rotations.
    """
    w = quaternion[..., 3]
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))  # Compute rotation angle
    return angle

def clip_rotation(quaternion, max_angle):
    """
    Clip rotation represented by quaternion to maximum angle.
    Args:
        quaternion: Tensor of shape (B, 4).
        max_angle: Maximum allowable rotation angle in radians.
    Returns:
        Quaternion clipped to the maximum angle.
    """
    angle = quaternion_magnitude(quaternion)
    scale = torch.ones_like(angle)
    mask = angle > max_angle
    scale[mask] = max_angle / angle[mask]
    quaternion = quaternion * scale.unsqueeze(-1)
    return normalize_quaternion(quaternion)
