#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from lib.pointops.functions import pointops
from utils import index_points, square_distance


device = torch.device('cuda')

# class PointTransformerLayer(nn.Module):
#     def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
#         super().__init__()
#         self.mid_planes = mid_planes = out_planes // 1
#         self.out_planes = out_planes
#         self.share_planes = share_planes
#         self.nsample = nsample
#         self.linear_q = nn.Linear(in_planes, mid_planes)
#         self.linear_k = nn.Linear(in_planes, mid_planes)
#         self.linear_v = nn.Linear(in_planes, out_planes)
#         self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
#         self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
#                                     nn.Linear(mid_planes, mid_planes // share_planes),
#                                     nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
#                                     nn.Linear(out_planes // share_planes, out_planes // share_planes))
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, pxo) -> torch.Tensor:
#         p, x, o = pxo  # (n, 3), (n, c), (b)
#         x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
#         x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
#         x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
#         p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
#         for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
#         w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
#         for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
#         w = self.softmax(w)  # (n, nsample, c)
#         n, nsample, c = x_v.shape; s = self.share_planes
#         x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
#         return x


class PointTransformerLayer(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)            #B*N*k*3

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)    #B*N*k*d_model

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f  left:B*N*1*3
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn



# def knn(x, k):
#     inner = -2*torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x**2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
#     return idx

# def get_graph_feature(x, k=20, idx=None, dim9=False):
#     batch_size = x.size(0)
#     num_points = x.size(2)
#     x = x.view(batch_size, -1, num_points)
#     if idx is None:
#         if dim9 == False:
#             idx = knn(x, k=k)   # (batch_size, num_points, k)
#         else:
#             idx = knn(x[:, 6:], k=k)
#     device = torch.device('cuda')

#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

#     idx = idx + idx_base

#     idx = idx.view(-1)
 
#     _, num_dims, _ = x.size()

#     x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size*num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims) 
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
#     feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
#     return feature      # (batch_size, 2*num_dims, num_points, k)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    distance = pairwise_distance.topk(k=k, dim=-1)[0]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return distance,idx

def get_graph_feature(xyz,x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)          #x.size()  B*C*N
    concat = torch.empty(batch_size,10,num_points,k)
    coords = xyz.permute([0, 2, 1])
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            dis, idx = knn(x, k=k)   # idx (batch_size, num_points, k)
        else:
            dis, idx = knn(x[:, 6:,], k=k)
            
    B, N, K = idx.size()
    # idx(B, N, K), coords(B, N, 3)
    # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
    extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
    extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
    neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)       
    concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dis.unsqueeze(-3)
    ), dim=-3).to(device) # B 10 N K

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)              #B*N*k
 
    _, num_dims, _ = x.size()


    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]  #[B*N*k,C]

    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # feature = x.permute(0, 3, 1, 2).contiguous()
    return concat, feature      # (batch_size, 2*num_dims, num_points, k)

class PointNet(nn.Module):
    def __init__(self, args, output_channels=50):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(1088, 512, kernel_size=1, bias=False)
        self.conv7 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv9 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv10 = nn.Conv1d(128, output_channels, kernel_size=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)
        # self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout()
        # self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x,l):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x1 = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # x = F.adaptive_max_pool1d(x, 1).squeeze()
        # x = F.relu(self.bn6(self.linear1(x)))
        # x = self.dp1(x)
        # x = self.linear2(x)
        x = F.adaptive_max_pool1d(x, 1).repeat(1,1,2048)
        x = torch.cat([x1,x],-2)   #B*1088*4096
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.conv10(x)
        return x





class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DFA_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DFA_partseg, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn_p1 = nn.BatchNorm1d(32)
        self.bn_p2 = nn.BatchNorm1d(64)
        self.bn_c1 = nn.BatchNorm2d(3*2)
        self.bn_c2 = nn.BatchNorm2d(64*2)
        self.bn_c3 = nn.BatchNorm2d(64*2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn0 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.point1 = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1, bias=False),self.bn_p1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.point2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, bias=False),self.bn_p2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp1 = nn.Sequential(nn.Conv2d(10, 3*2, kernel_size=1, bias=False),self.bn_c1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv2d(10, 64*2, kernel_size=1, bias=False),self.bn_c2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp3 = nn.Sequential(nn.Conv2d(10, 64*2, kernel_size=1, bias=False),self.bn_c3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(3*4, 64, kernel_size=1, bias=False),self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*4, 128, kernel_size=1, bias=False),self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*4, 64*2, kernel_size=1, bias=False),self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv0 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 512, kernel_size=1, bias=False),self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x,l):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x[:,:3,:]     #B,C,N

        concat,x0 = get_graph_feature(xyz,x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x0 = self.point1(x)
        x0 = self.point2(x0)
        x0 = x0.max(dim=-1, keepdim=False)[0]
        x0 = x0.unsqueeze(-1)
        x0 =x0.repeat(1,1,num_points)

        concat,x = get_graph_feature(xyz , x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        concat = self.mlp1(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # attention = F.softmax(x, dim=-1) 
        # attention = F.dropout(attention, self.dropout)
        # x1 = torch.sum(torch.mul(attention, x),dim = -1)

        concat,x = get_graph_feature(xyz ,x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        concat = self.mlp2(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # attention = F.softmax(x, dim=-1) 
        # attention = F.dropout(attention, self.dropout)
        # x2 = torch.sum(torch.mul(attention, x),dim = -1)

        concat,x = get_graph_feature(xyz ,x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        concat = self.mlp3(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x =self.conv0(x)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # attention = F.softmax(x, dim=-1) 
        # attention = F.dropout(attention, self.dropout)
        # x3 = torch.sum(torch.mul(attention, x),dim = -1)

        x = torch.cat((x0,x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x


class DFA_semseg(nn.Module):
    def __init__(self, args , dropout=0,alpha=0.2):
        super(DFA_semseg, self).__init__()
        self.args = args
        self.k = args.k
        self.dropout = args.dropout
        self.transform_net = Transform_Net(args)
        
        self.bn_p1 = nn.BatchNorm1d(32)
        self.bn_p2 = nn.BatchNorm1d(64)
        self.bn_c1 = nn.BatchNorm2d(9*2)
        self.bn_c2 = nn.BatchNorm2d(64*2)
        self.bn_c3 = nn.BatchNorm2d(64*2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64*2)
        self.bn = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        # self.bn7 = nn.BatchNorm1d(1024)
        # self.bn8 = nn.BatchNorm1d(512)
        # self.bn9 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)


        self.point1 = nn.Sequential(nn.Conv1d(9, 32, kernel_size=1, bias=False),self.bn_p1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.point2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, bias=False),self.bn_p2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp1 = nn.Sequential(nn.Conv2d(10, 9*2, kernel_size=1, bias=False),self.bn_c1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv2d(10, 64*2, kernel_size=1, bias=False),self.bn_c2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp3 = nn.Sequential(nn.Conv2d(10, 64*2, kernel_size=1, bias=False),self.bn_c3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(9*4, 64, kernel_size=1, bias=False),self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*4, 64*2, kernel_size=1, bias=False),self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*4, 64*2, kernel_size=1, bias=False),self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(128, 13, kernel_size=1, bias=False)
        # self.attn1 = SA_Layer(64)   #in:(B,C,N)
        # self.attn2 = SA_Layer(64)
        # self.attn3 = SA_Layer(64)

        # self.att1 = Attention(dim = 256)   #  in : B,N,C   out:B,N,C
        # self.att2 = Attention(dim = 64)
        # self.att3 = Attention(dim = 64)
        # self.attn1 = TransformerBlock(64,64)
        # self.attn2 = TransformerBlock(64,64)
        # self.attn3 = TransformerBlock(64,64)




    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x[:,:3,:]     #B,C,N

        concat,x0 = get_graph_feature(xyz,x, k=self.k,dim9=True)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x0 = self.point1(x)
        x0 = self.point2(x0)
        # x0 = self.attn1(x0)
        x0 = x0.max(dim=-1, keepdim=False)[0]
        x0 = x0.unsqueeze(-1)
        x0 =x0.repeat(1,1,num_points)
        #concat_neigh,idx = get_neigh_xzy(xyz,x,k=self.k)

        #---------------------------------------------------------------------------------------------------------------------------
        # x = self.att1(x.permute(0,2,1))
        # x = x.permute(0,2,1)
        # x_neigh = get_neigh_feature(xyz,x,idx,k=self.k)
        concat, x = get_graph_feature(xyz,x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        #concat = torch.cat([concat_neigh,concat],dim= 1)
        # x = x_neigh + x
        #x = torch.sum([x_neigh,x],dim= 1)
        concat = self.mlp1(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)

        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # attention = F.softmax(x, dim=-1) 
        # attention = F.dropout(attention, self.dropout)
        # x1 = torch.sum(torch.mul(attention, x),dim = -1)
        # x1 = self.att1(x1.permute(0,2,1))
        # x1 = x1.permute(0,2,1)
        # x1 = self.attn1.execute(x1)
        
        # x1, attn_output_weights = self.attn1(x1.permute(2,0,1), x1.permute(2,0,1), x1.permute(2,0,1))    #in:(N,B,C) out:(N,B,C) /(B,N,N)
        # x1 = x1.permute(1,2,0)
        #---------------------------------------------------------------------------------------------------------------------------

        #---------------------------------------------------------------------------------------------------------------------------
        # x = self.att2(x1.permute(0,2,1))
        # x = x.permute(0,2,1)
        # x_neigh = get_neigh_feature(xyz,x1,idx,k=self.k)
        concat, x = get_graph_feature(xyz,x1, k=self.k, dim9=False)   # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        #concat = torch.cat([concat_neigh,concat],dim= 1)
        # x = x_neigh + x
        # x = torch.sum([x_neigh,x],dim= 1)
        concat = self.mlp2(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # attention = F.softmax(x, dim=-1) 
        # attention = F.dropout(attention, self.dropout)
        # x2 = torch.sum(torch.mul(attention, x),dim = -1)
        # x2 = self.attn2.execute(x2)
        # x2, attn_output_weights = self.attn2(x2.permute(2,0,1), x2.permute(2,0,1), x2.permute(2,0,1))    #in:(N,B,C) out:(N,B,C) /(B,N,N)
        # x2 = x2.permute(1,2,0)
        #---------------------------------------------------------------------------------------------------------------------------

        #---------------------------------------------------------------------------------------------------------------------------
        # x = self.att3(x2.permute(0,2,1))
        # x = x.permute(0,2,1)
        # x_neigh = get_neigh_feature(xyz,x2,idx,k=self.k)
        concat, x = get_graph_feature(xyz,x2, k=self.k, dim9=False)   # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # concat = torch.cat([concat_neigh,concat],dim= 1)
        # x = x_neigh + x
        # x = torch.sum([x_neigh,x],dim= 1)
        concat = self.mlp3(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv(x)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # attention = F.softmax(x, dim=-1) 
        # attention = F.dropout(attention, self.dropout)
        # x3 = torch.sum(torch.mul(attention, x),dim = -1)
        # x3 = self.attn3.execute(x3)
        # x3, attn_output_weights = self.attn3(x3.permute(2,0,1), x3.permute(2,0,1), x3.permute(2,0,1))    #in:(N,B,C) out:(N,B,C) /(B,N,N)
        # x3 = x3.permute(1,2,0)
        # x4 = torch.cat([x0,x1,x2,x3],dim=1)
        # x4 = self.att1(x4.permute(0,2,1))
        # x4 = x4.permute(0,2,1)

        x = torch.cat((x0,x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x,x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)
        
        x = self.conv7(x)                       # (batch_size, 1216, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                       # (batch_size, 128, num_points) -> (batch_size, 13, num_points)
        
        return x


class DFA_cls(nn.Module):
    # def __init__(self, args ,d_points, d_model, dropout=0,alpha=0.2):
    def __init__(self, args, dropout=0,alpha=0.2):
        super(DFA_cls, self).__init__()
        self.args = args
        self.k = args.k
        self.dropout = args.dropout
        self.transform_net = Transform_Net(args)
        # self.pointtrans = PointTransformerLayer(d_points, d_model, self.k)
        
        self.bn_p1 = nn.BatchNorm1d(32)
        self.bn_p2 = nn.BatchNorm1d(64)
        self.bn_c0 = nn.BatchNorm2d(3*2)
        self.bn_c1 = nn.BatchNorm2d(64*2)
        self.bn_c2 = nn.BatchNorm2d(64*2)
        self.bn_c3 = nn.BatchNorm2d(64*2)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64*2)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn = nn.BatchNorm2d(64*3)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)


        self.point1 = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1, bias=False),self.bn_p1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.point2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, bias=False),self.bn_p2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp0 = nn.Sequential(nn.Conv2d(10, 3*2, kernel_size=1, bias=False),self.bn_c0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp1 = nn.Sequential(nn.Conv2d(10, 64*2, kernel_size=1, bias=False),self.bn_c1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv2d(10, 64*2, kernel_size=1, bias=False),self.bn_c2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp3 = nn.Sequential(nn.Conv2d(10, 64*2, kernel_size=1, bias=False),self.bn_c3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv0 = nn.Sequential(nn.Conv2d(3*4, 64, kernel_size=1, bias=False),self.bn0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(64*4, 64*2, kernel_size=1, bias=False),self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*4, 64*2, kernel_size=1, bias=False),self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*4, 64, kernel_size=1, bias=False),self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv = nn.Sequential(nn.Conv2d(64*3, 64*3, kernel_size=1, bias=False),self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(320, args.emb_dims, kernel_size=1, bias=False),self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, 40)

        self.score_fn0 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
        self.score_fn1 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
        self.score_fn2 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
        self.score_fn3 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )



    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x[:,:3,:]     #B,C,N

        # concat,x0 = get_graph_feature(xyz,x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (batch_size, 3, 3)
        # x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x00 = self.point1(x)
        x00 = self.point2(x00)
        x00 = x00.max(dim=-1, keepdim=False)[0]
        x00 = x00.unsqueeze(-1)
        x00 =x00.repeat(1,1,num_points)
        # x00,y = self.pointtrans(xyz.permute(0,2,1),x00.permute(0,2,1))
        # x00 = x00.permute(0,2,1)
        
        concat, x = get_graph_feature(xyz,x, k=self.k)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        concat = self.mlp0(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv0(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x0 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x0 = x.sum(dim=-1, keepdim=False) 
        # scores = self.score_fn0(x.permute(0,2,3,1)).permute(0,3,1,2)
        # features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze()
        # x0=x0 + features


        #---------------------------------------------------------------------------------------------------------------------------

        concat, x = get_graph_feature(xyz,x0, k=self.k)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        concat = self.mlp1(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x1 = x.sum(dim=-1, keepdim=False)
        # scores = self.score_fn1(x.permute(0,2,3,1)).permute(0,3,1,2)
        # features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze()
        # x1=x1 + features

        #---------------------------------------------------------------------------------------------------------------------------

        #---------------------------------------------------------------------------------------------------------------------------
        concat, x = get_graph_feature(xyz,x1, k=self.k)   # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        concat = self.mlp2(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x2 = x.sum(dim=-1, keepdim=False)
        # scores = self.score_fn2(x.permute(0,2,3,1)).permute(0,3,1,2)
        # features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze()
        # x2=x2 + features
        #---------------------------------------------------------------------------------------------------------------------------

        #---------------------------------------------------------------------------------------------------------------------------

        concat, x = get_graph_feature(xyz,x2, k=self.k)   # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        concat = self.mlp3(concat)
        x = torch.cat([concat, x], dim=1)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x3 = x.sum(dim=-1, keepdim=False)
        # scores = self.score_fn3(x.permute(0,2,3,1)).permute(0,3,1,2)
        # features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze()
        # x3=x3 + features


        x = torch.cat((x00,x0,x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        #x = torch.cat((x0,x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x