
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
import torch.nn.functional as F


# ---------------------- CausalRefineNet and Associated Classes ----------------------

class CausalRefineNet(nn.Module):
    '''
    A causally-motivated refinement network that models:
    1. Geometric misalignment -> Pose correction
    2. Local structure -> Global transformation
    3. Confidence-weighted updates
    
    Maintains same interface as PoseRefineNet for compatibility
    '''
    def __init__(self, num_points, num_obj):
        super(CausalRefineNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        
        # Local geometric feature extraction
        self.local_geo = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        
        # Structural error estimation
        self.error_est = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)  # 3D error vectors
        )
        
        # Uncertainty estimation in geometric features
        self.uncertainty = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()  # Confidence scores
        )
        
        # Pose correction from weighted error
        self.pose_correction = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final correction branches with causal chain
        self.rotation = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4 * num_obj)  # Quaternion correction per object
        )
        
        self.translation = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_obj)  # Translation correction per object
        )
        
        self.ln = nn.LayerNorm(64)

    def forward(self, x, emb, obj):
        """
        Causal refinement process:
        1. Extract local geometric features
        2. Estimate structural misalignment
        3. Compute uncertainty in estimates
        4. Generate confidence-weighted corrections
        """
        bs = x.size()[0]
        
        # Prepare point cloud
        x = x.transpose(2, 1).contiguous()  # (bs, 3, N)
        
        # Extract local geometric features
        local_features = self.local_geo(x)  # (bs, 256, N)
        
        # Estimate structural errors - this directly causes pose updates
        struct_errors = self.error_est(local_features)  # (bs, 3, N)
        
        # Estimate uncertainty in geometric features
        confidence = self.uncertainty(local_features)  # (bs, 1, N)
        
        # Weight errors by confidence
        weighted_error = struct_errors * confidence  # (bs, 3, N)
        
        # Aggregate weighted errors
        error_feat = torch.max(weighted_error, dim=2)[0]  # (bs, 3)
        error_feat = error_feat.unsqueeze(2).expand(-1, -1, local_features.size(2))
        
        # Combine with local features for context
        combined = torch.cat([local_features, error_feat], dim=1)  # (bs, 259, N)
        
        # Global pooling with confidence weighting
        global_feat = torch.sum(combined * confidence, dim=2) / (torch.sum(confidence, dim=2) + 1e-7)
        
        # Generate pose corrections using causal chain
        correction_feat = self.pose_correction(global_feat)
        correction_feat = self.ln(correction_feat)
        
        # Split into rotation and translation corrections
        rot_correction = self.rotation(correction_feat).view(bs, self.num_obj, 4)
        trans_correction = self.translation(correction_feat).view(bs, self.num_obj, 3)
        
        # Select specific object corrections
        b = 0
        out_rx = torch.index_select(rot_correction[b], 0, obj[b])  # (1, 4)
        out_tx = torch.index_select(trans_correction[b], 0, obj[b])  # (1, 3)
        
        return out_rx, out_tx

class CausalFeatureExtractor(nn.Module):
    '''
    Support module for extracting causal features from points.
    Focuses on local geometric structures that directly influence pose.
    '''
    def __init__(self, num_points):
        super(CausalFeatureExtractor, self).__init__()
        self.num_points = num_points
        
        # Local structure analysis
        self.local_feature = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        
        # Local-to-global aggregation
        self.global_feature = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1)
        )
        
        self.ap = nn.AvgPool1d(num_points)

    def forward(self, x):
        """Extract features respecting causal structure"""
        x = x.transpose(2, 1).contiguous()
        
        # Extract local features that cause pose changes
        local_feat = self.local_feature(x)
        
        # Aggregate to global features
        x = self.global_feature(local_feat)
        global_feat = self.ap(x)
        
        return global_feat.view(-1, 1024)

class GeometricModule(nn.Module):
    '''
    Purpose: Extract geometric features matching refiner dimensions
    Outputs:
        features: (bs, 1024) global geometric features
    '''
    def __init__(self, num_points):
        super(GeometricModule, self).__init__()
        
        # Point feature extraction
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(1024)
        
        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, x):
        """
        Extract geometric features from point cloud
        Args:
            x: (bs, 3, N) input points
        Returns:
            features: (bs, 1024) global geometric features
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = self.ap1(x)
        out = x.view(-1, 1024)
        
        return out


class InterventionAwareCausalRefineNet(nn.Module):
    '''
    Refinement network that explicitly models interventions:
    1. How changes in point positions affect pose estimates
    2. How different geometric transformations intervene on pose
    3. Counterfactual reasoning about pose corrections
    '''
    def __init__(self, num_points, num_obj):
        super(InterventionAwareCausalRefineNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        
        # Feature extraction remains similar
        self.local_geo = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        
        # Intervention effect estimator 
        # Predicts how changes in point positions affect pose
        self.intervention_estimator = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),  
            nn.ReLU(),
            nn.Conv1d(64, 6, 1)  # 3 for rotation effect, 3 for translation effect
        )
        
        # Counterfactual pose generator
        # Generates multiple possible pose corrections
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7 * 3)  # Generate 3 possible corrections (7D pose each)
        )
        
        # Intervention scorer - evaluates which intervention is most effective
        self.intervention_scorer = nn.Sequential(
            nn.Linear(7 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )
        
        # Final pose correction using best intervention
        self.pose_correction = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU()
        )
        
        self.rotation = nn.Linear(32, 4 * num_obj)
        self.translation = nn.Linear(32, 3 * num_obj)
        
        self.ln = nn.LayerNorm(32)

    def compute_intervention_effects(self, points, features):
        """
        Compute how different geometric transformations would affect the pose
        Returns effects tensor showing impact of possible interventions
        """
        intervention_effects = self.intervention_estimator(features)
        return intervention_effects

    def generate_counterfactuals(self, features):
        """
        Generate multiple possible pose corrections and their expected outcomes
        Returns multiple candidate corrections
        """
        global_feat = torch.max(features, dim=2)[0]
        counterfactuals = self.counterfactual_generator(global_feat)
        return counterfactuals.view(-1, 3, 7)  # 3 possible corrections

    def score_interventions(self, counterfactuals, intervention_effects):
        """
        Score different possible interventions based on their expected effects
        Returns probabilities for each intervention
        """
        flat_counterfactuals = counterfactuals.view(-1, 21)  # Flatten counterfactuals
        scores = self.intervention_scorer(flat_counterfactuals)
        return scores

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        x = x.transpose(2, 1).contiguous()
        
        # Extract geometric features
        local_features = self.local_geo(x)
        
        # Compute intervention effects
        intervention_effects = self.compute_intervention_effects(x, local_features)
        
        # Generate counterfactual poses
        counterfactuals = self.generate_counterfactuals(local_features)
        
        # Score interventions
        intervention_scores = self.score_interventions(counterfactuals, intervention_effects)
        
        # Select best counterfactual based on scores
        best_correction = torch.sum(counterfactuals * intervention_scores.unsqueeze(-1), dim=1)
        
        # Generate final pose correction
        correction_feat = self.pose_correction(best_correction)
        correction_feat = self.ln(correction_feat)
        
        # Output pose corrections
        rot_correction = self.rotation(correction_feat).view(bs, self.num_obj, 4)
        trans_correction = self.translation(correction_feat).view(bs, self.num_obj, 3)
        
        # Select specific object
        b = 0
        out_rx = torch.index_select(rot_correction[b], 0, obj[b])
        out_tx = torch.index_select(trans_correction[b], 0, obj[b])
        
        return out_rx, out_tx

    def get_intervention_analysis(self, x, emb, obj):
        """
        Additional method for analyzing intervention effects
        Returns dictionary with intervention effects and scores
        """
        x = x.transpose(2, 1).contiguous()
        local_features = self.local_geo(x)
        
        intervention_effects = self.compute_intervention_effects(x, local_features)
        counterfactuals = self.generate_counterfactuals(local_features)
        intervention_scores = self.score_interventions(counterfactuals, intervention_effects)
        
        return {
            'effects': intervention_effects,
            'counterfactuals': counterfactuals,
            'scores': intervention_scores
        }

class CausalFeatureProcessor(nn.Module):
    '''
    Purpose: Process point features with causal relationships
    Outputs: 
        processed_features: (bs, 1024) processed global features
    '''
    def __init__(self, num_points):
        super(CausalFeatureProcessor, self).__init__()
        
        # Global feature processing
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Process features maintaining causal relationships
        Args:
            x: (bs, 1024) input features
        Returns:
            processed_features: (bs, 1024) processed features
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        
        return x

class VisibilityModule(nn.Module):
    '''
    Purpose: Estimate point visibility scores
    Outputs:
        visibility_scores: (bs, N) visibility scores per point
    '''
    def __init__(self, num_points):
        super(VisibilityModule, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.conv5 = nn.Conv1d(128, 1, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x):
        """
        Estimate visibility scores for input points
        Args:
            x: (bs, 3, N) input points
        Returns:
            visibility_scores: (bs, N) visibility scores
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        
        return torch.sigmoid(x.squeeze(1))

