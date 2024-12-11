from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
# from lib.knn.__init__ import KNearestNeighbor
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

def knn(x, y, k=1):
    _, dim, x_size = x.shape
    _, _, y_size = y.shape

    x = x.detach().squeeze().transpose(0, 1)
    y = y.detach().squeeze().transpose(0, 1)

    xx = (x**2).sum(dim=1, keepdim=True).expand(x_size, y_size)
    yy = (y**2).sum(dim=1, keepdim=True).expand(y_size, x_size).transpose(0, 1)

    dist_mat = xx + yy - 2 * x.matmul(y.transpose(0, 1))
    if k == 1:
        return dist_mat.argmin(dim=0)
    mink_idxs = dist_mat.argsort(dim=0)
    return mink_idxs[: k]

def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    # knn = KNearestNeighbor(1)
    pred_r = pred_r.view(1, 1, -1)
    pred_t = pred_t.view(1, 1, -1)
    bs, num_p, _ = pred_r.size()
    num_input_points = len(points[0])

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
    
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    
    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t

    pred = torch.add(torch.bmm(model_points, base), pred_t)

    if idx[0].item() in sym_list:
        target = target[0].transpose(1, 0).contiguous().view(3, -1)
        pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1))
        target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
        pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

    t = ori_t[0]
    points = points.view(1, num_input_points, 3)

    ori_base = ori_base[0].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_input_points, 1).contiguous().view(1, bs * num_input_points, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis.item(), idx[0].item())
    # del knn
    return dis, new_points.detach(), new_target.detach()


class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list


    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)
    


class SCMLoss(nn.Module):
    """
    Loss function implementing SCM principles for pose refinement
    """
    def __init__(self, num_points_mesh: int, sym_list: List[int]):
        super(SCMLoss, self).__init__()
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list
        

    @staticmethod
    def safe_tensor(value, device):
        """Safely create a tensor on the specified device"""
        try:
            if isinstance(value, (int, float)):
                return torch.tensor(float(value), device=device, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                return value.to(device)
            else:
                raise ValueError(f"Unsupported value type: {type(value)}")
        except Exception as e:
            print(f"Error creating tensor: {str(e)}")
            return torch.zeros(1, device=device, requires_grad=True)

    def compute_pose_loss(self, pred_r: torch.Tensor, pred_t: torch.Tensor,
                         target: torch.Tensor, model_points: torch.Tensor) -> torch.Tensor:
        """
        Compute pose estimation loss using rotation and translation predictions.
        
        Args:
            pred_r: Predicted rotation as quaternion [B, 4]
            pred_t: Predicted translation [B, 3]
            target: Target point positions [B, N, 3]
            model_points: Model points [B, N, 3]
        """
        # Reshape inputs
        pred_r = pred_r.view(1, 1, -1)
        pred_t = pred_t.view(1, 1, -1)
        bs, num_p, _ = pred_r.size()
        
        # Normalize quaternion
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1) + 1e-8)
        
        # Convert quaternion to rotation matrix
        base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                         (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                         (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                         (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                         (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                         (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                         (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                         (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                         (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
        
        base = base.transpose(2, 1)
        
        # Reshape points for transformation
        model_points = model_points.view(bs, 1, self.num_pt_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, self.num_pt_mesh, 3)
        target = target.view(bs, 1, self.num_pt_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, self.num_pt_mesh, 3)
        pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
        
        # Apply transformation
        pred = torch.add(torch.bmm(model_points, base), pred_t)
        
        # Compute distance loss
        return torch.mean(torch.norm((pred - target), dim=2))

    def compute_intervention_loss(self, features: torch.Tensor, intervention: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between features under intervention.
        
        Args:
            features: Point features [B, C, N]
            intervention: Transformation matrix [B, 4, 4]
        """
        # Get feature shape
        B, C, N = features.shape
        
        # Reshape features for spatial transformation
        features_points = features.transpose(1, 2)  # [B, N, C]
        
        # Take first 3 dimensions for spatial features
        spatial_feats = features_points[..., :3]  # [B, N, 3]
        
        # Apply transformation
        ones = torch.ones(B, N, 1, device=features.device)
        homog_feats = torch.cat([spatial_feats, ones], dim=-1)  # [B, N, 4]
        transformed = torch.bmm(homog_feats, intervention.transpose(1, 2))
        transformed = transformed[..., :3]  # Remove homogeneous coordinate
        
        # Compute feature consistency loss
        return F.mse_loss(transformed, spatial_feats)

    def compute_backdoor_loss(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute backdoor adjustment loss between features.
        
        Args:
            features: Dictionary containing 'relational' and 'global' features
        """
        # Get features
        point_feats = features['relational']  # [B, C, N]
        global_feats = features['global']     # [B, C, 1]
        
        # Project point features onto global feature space
        point_feats = F.normalize(point_feats, dim=1)  # Normalize along channel dimension
        global_feats = F.normalize(global_feats, dim=1)
        
        # Compute correlation using batch matrix multiplication
        similarity = torch.bmm(point_feats.transpose(1, 2), global_feats)  # [B, N, 1]
        
        # We want low correlation for backdoor adjustment
        target = torch.zeros_like(similarity)
        return F.mse_loss(similarity, target)

    def forward(self, pred_r: torch.Tensor, pred_t: torch.Tensor,
                target: torch.Tensor, model_points: torch.Tensor,
                points: torch.Tensor, features: Dict[str, torch.Tensor],
                interventions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing all losses
        """
        # Ensure all inputs are on same device
        device = pred_r.device
        
        # Compute base pose loss
        pose_loss = self.compute_pose_loss(pred_r, pred_t, target, model_points)
        
        # Initialize losses dictionary
        losses = {'pose_loss': pose_loss}
        
        # Compute intervention losses if provided
        if interventions is not None:
            if 'view' in interventions:
                view_loss = self.compute_intervention_loss(features['relational'], interventions['view'])
                losses['view_loss'] = view_loss

            if 'symmetry' in interventions:
                sym_loss = self.compute_intervention_loss(features['relational'], interventions['symmetry'])
                losses['symmetry_loss'] = sym_loss

        # Compute backdoor loss
        backdoor_loss = self.compute_backdoor_loss(features)
        losses['backdoor_loss'] = backdoor_loss

        # Compute total loss with weighting
        total_loss = pose_loss
        for name, loss in losses.items():
            if name != 'pose_loss':
                total_loss = total_loss + 0.1 * loss

        losses['total_loss'] = total_loss
        return losses