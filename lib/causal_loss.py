from torch.nn.modules.loss import _Loss
import torch
import numpy as np
import torch.nn as nn
from lib.knn.__init__ import KNearestNeighbor

def knn(x, y, k=1):
    # Reuse existing KNN implementation from loss_refiner.py
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

def visibility_consistency_loss(visibility_mask, pred_pose, point_cloud):
    """Loss to ensure visibility predictions are geometrically consistent"""
    # Project points using predicted pose
    projected_points = torch.bmm(point_cloud, pred_pose[:, :3, :3].transpose(1, 2))
    projected_points += pred_pose[:, :3, 3].unsqueeze(1)
    
    # Points with negative z should be marked invisible
    should_be_invisible = (projected_points[..., 2] < 0).float()
    visibility_error = torch.mean(torch.abs(visibility_mask - should_be_invisible))
    
    return visibility_error

def geometric_consistency_loss(geom_features, point_cloud, k=20):
    """Loss to ensure geometric features respect local point cloud structure"""
    # Compute local neighborhood structure
    batch_size, num_points, _ = point_cloud.size()
    
    # For each point, find k nearest neighbors
    dist = torch.cdist(point_cloud, point_cloud)
    _, knn_idx = dist.topk(k=k, dim=2, largest=False)
    
    # Features of neighboring points should be similar
    neighbor_features = torch.gather(geom_features, 1, 
                                   knn_idx.unsqueeze(-1).expand(-1, -1, -1, geom_features.size(-1)))
    
    feature_consistency = torch.mean((geom_features.unsqueeze(2) - neighbor_features)**2)
    
    return feature_consistency

def confidence_regularization(pred_confidence, min_conf=0.1):
    """Regularize confidence predictions to avoid degenerate solutions"""
    # Ensure confidence stays in reasonable range
    conf_penalty = torch.mean(torch.max(min_conf - pred_confidence, 
                                      torch.zeros_like(pred_confidence)))
    
    # Add entropy term to avoid overconfident predictions
    entropy = -torch.mean(pred_confidence * torch.log(pred_confidence + 1e-10))
    
    return conf_penalty - 0.1 * entropy

def scm_loss_calculation(pred_pose, pred_conf, target_pose, point_cloud,
                        visibility_mask, geom_features, model_points, 
                        idx, num_point_mesh, sym_list):
    """Complete loss calculation incorporating SCM components"""
    # Original pose loss from loss_refiner.py
    base_loss, _, _ = loss_calculation(pred_pose[:,:3], pred_pose[:,3:], 
                                     target_pose, model_points, idx, point_cloud,
                                     num_point_mesh, sym_list)
    
    # Additional SCM-specific losses
    vis_loss = visibility_consistency_loss(visibility_mask, pred_pose, point_cloud)
    geom_loss = geometric_consistency_loss(geom_features, point_cloud)
    conf_loss = confidence_regularization(pred_conf)
    
    # Combine losses with weighting
    total_loss = base_loss + 0.1 * vis_loss + 0.1 * geom_loss + 0.05 * conf_loss
    
    return total_loss, base_loss, vis_loss, geom_loss, conf_loss

class CausalRefineLoss(_Loss):
    def __init__(self, num_points_mesh, sym_list):
        super(CausalRefineLoss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_pose, pred_conf, target_pose, point_cloud,
               visibility_mask, geom_features, model_points, idx):
        """
        pred_pose: (bs, 7) - First 4 for quaternion, last 3 for translation
        pred_conf: (bs, 1) - Predicted confidence
        visibility_mask: (bs, N) - Binary mask for point visibility
        geom_features: (bs, N, F) - Geometric features per point
        (Other parameters same as original loss_refiner)
        """
        return scm_loss_calculation(pred_pose, pred_conf, target_pose, point_cloud,
                                  visibility_mask, geom_features, model_points, 
                                  idx, self.num_pt_mesh, self.sym_list)