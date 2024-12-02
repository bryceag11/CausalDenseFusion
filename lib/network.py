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
class CausalRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(CausalRefineNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        
        # Feature extraction matching PoseRefineNetFeat
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)  # Changed from 384 to 256
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.ap1 = torch.nn.AvgPool1d(num_points)
        
        # Geometric feature extraction - separate stream
        self.geo_conv1 = nn.Conv1d(3, 64, 1)
        self.geo_conv2 = nn.Conv1d(64, 128, 1)
        self.geo_conv3 = nn.Conv1d(128, 256, 1)
        self.geo_ap = nn.AvgPool1d(num_points)
        
        # Final pose estimation - using both global and geometric features
        self.pose_r1 = nn.Linear(1024 + 256, 512)  # Combined features
        self.pose_r2 = nn.Linear(512, num_obj * 4)  
        
        self.pose_t1 = nn.Linear(1024 + 256, 512)  # Combined features
        self.pose_t2 = nn.Linear(512, num_obj * 3)
        
        # Layer normalization instead of batch norm
        self.ln_r = nn.LayerNorm(512)
        self.ln_t = nn.LayerNorm(512)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        
        # Keep original points for geometric stream
        input_points = x.transpose(2, 1).contiguous()  # (bs, 3, N)
        
        # Main feature stream
        x1 = F.relu(self.conv1(input_points))
        emb1 = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x1, emb1], dim=1)
        
        x2 = F.relu(self.conv2(x1))
        emb2 = F.relu(self.e_conv2(emb1))
        pointfeat_2 = torch.cat([x2, emb2], dim=1)
        
        # Global feature processing
        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))
        global_feat = self.ap1(x).view(bs, -1)  # (bs, 1024)
        
        # Geometric feature stream - using original points
        geo_x = F.relu(self.geo_conv1(input_points))
        geo_x = F.relu(self.geo_conv2(geo_x))
        geo_x = F.relu(self.geo_conv3(geo_x))
        geo_feat = self.geo_ap(geo_x).view(bs, -1)  # (bs, 256)
        
        # Combine global and geometric features
        combined_feat = torch.cat([global_feat, geo_feat], dim=1)  # (bs, 1280)
        
        # Estimate rotation with dropout
        rx = F.relu(self.ln_r(self.pose_r1(combined_feat)))
        rx = self.dropout(rx)
        rx = self.pose_r2(rx).view(bs, self.num_obj, 4)
        
        # Estimate translation with dropout
        tx = F.relu(self.ln_t(self.pose_t1(combined_feat)))
        tx = self.dropout(tx)
        tx = self.pose_t2(tx).view(bs, self.num_obj, 3)
        
        # Select object specific outputs
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])  # (1, 4)
        out_tx = torch.index_select(tx[b], 0, obj[b])  # (1, 3)
        
        return out_rx, out_tx

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

