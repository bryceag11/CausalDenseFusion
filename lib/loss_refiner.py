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
    Fixed loss function implementing proper causal objectives
    """
    def __init__(self, num_points_mesh: int, sym_list: list):
        super().__init__()
        self.num_points_mesh = num_points_mesh
        self.sym_list = sym_list
        
    def compute_pose_loss(self, pred_r: torch.Tensor, pred_t: torch.Tensor,
                         target: torch.Tensor, model_points: torch.Tensor) -> torch.Tensor:
        """Standard pose estimation loss"""
        # Convert quaternion to rotation matrix
        pred_r = F.normalize(pred_r, dim=1)
        
        # Build rotation matrix from quaternion
        w, x, y, z = pred_r[:, 0], pred_r[:, 1], pred_r[:, 2], pred_r[:, 3]
        
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=1),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=1),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=1)
        ], dim=1)  # [B, 3, 3]
        
        # Transform model points
        model_points = model_points[:, :self.num_points_mesh, :]  # [B, N, 3]
        pred_points = torch.bmm(model_points, R.transpose(1, 2)) + pred_t.unsqueeze(1)
        
        # Compute distance
        target = target[:, :self.num_points_mesh, :]
        return torch.mean(torch.norm(pred_points - target, dim=2))
    
    def compute_causal_consistency_loss(self, features: Dict[str, torch.Tensor],
                                      interventions: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Ensure causal mechanisms are consistent under interventions
        """
        loss = 0.0
        
        # Residuals should be predictive of pose updates
        if 'residuals' in features and 'pose_update' in features:
            # Simple consistency: residuals should correlate with magnitude of update
            residual_norm = torch.norm(features['residuals'], dim=1)
            update_norm = torch.norm(features['pose_update'], dim=1)
            loss += F.mse_loss(residual_norm, update_norm)
        
        return loss
    
    def compute_intervention_robustness_loss(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encourage robustness to interventions
        """
        # Geometric features should maintain structure
        if 'geometric' in features and 'global' in features['geometric']:
            global_feat = features['geometric']['global']
            # Encourage feature diversity
            feat_cov = torch.mm(global_feat.t(), global_feat) / global_feat.size(0)
            loss = -torch.logdet(feat_cov + 1e-4 * torch.eye(feat_cov.size(0)).cuda())
            return loss
        return torch.tensor(0.0).cuda()
    
    def forward(self, pred_r: torch.Tensor, pred_t: torch.Tensor,
                target: torch.Tensor, model_points: torch.Tensor,
                features: Dict[str, torch.Tensor],
                interventions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        """
        # Base pose loss
        pose_loss = self.compute_pose_loss(pred_r, pred_t, target, model_points)
        
        # Causal consistency
        consistency_loss = self.compute_causal_consistency_loss(features, interventions)
        
        # Intervention robustness
        robustness_loss = self.compute_intervention_robustness_loss(features)
        
        # Total loss
        total_loss = pose_loss + 0.1 * consistency_loss + 0.05 * robustness_loss
        
        return {
            'pose_loss': pose_loss,
            'consistency_loss': consistency_loss,
            'robustness_loss': robustness_loss,
            'total_loss': total_loss
        }