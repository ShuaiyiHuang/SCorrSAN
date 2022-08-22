'''
SCorrSAN: Learning Semantic Correspondence with Sparse Annotations
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import add
from functools import reduce
import numpy as np

from models.feature_backbones import resnet

class SCorrSAN(nn.Module):
    def __init__(self, sce_ksize = 7, sce_outdim = 2048, feature_size=64, freeze=False):
        super().__init__()
        # Note that fea_size=64 is stride4 for input resolution 256x256
        hids = [30] # [30] means layer4 single feature 
        self.feature_size = feature_size
        self.feature_extraction = FeatureExtraction(hids, feature_size, freeze) 

        channels = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3
        self.SCEs = nn.ModuleList([EfficientSpatialContextNet(kernel_size=sce_ksize, input_channel = channels[i], output_channel = sce_outdim) for i in hids])

        # kernel soft argmax
        self.l2norm = FeatureL2Norm()
    
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

        print('verbose...SCorrSAN model...efficient sce ksize {}, sce_outdim {}'.format(sce_ksize, sce_outdim))
    
    def forward(self, target, source, train_mode=False):
        B, _, H, W = target.size()

        src_feats = self.feature_extraction(source)
        tgt_feats = self.feature_extraction(target)
        nlayer = len(src_feats)

        corrs = []
        for i, (src, tgt) in enumerate(zip(src_feats, tgt_feats)):
            _, _, hsi, wsi = src.shape 
            _, _, hti, wti = tgt.shape

            src_feat_sce = self.SCEs[i](src) 
            tgt_feat_sce = self.SCEs[i](tgt)

            corr = self.corr(self.l2norm(src_feat_sce), self.l2norm(tgt_feat_sce)) # hsy corr (b, Lsi, Lti)
            corr = self.mutual_nn_filter(corr.unsqueeze(1))
            corr = corr.view(B, hsi, wsi, hti, wti ) # (b, Lsi, hti, wti)
            corrs.append(corr)
        
        corrs_resize = []
        for i, corr in enumerate(corrs):
            _, hsi, wsi, hti, wti,  = corr.shape
            Lti = hti * wti
            Lsi = hsi * wsi
            Lsf = self.feature_size * self.feature_size # here htf = wtf = hsf = wsf = self.feature_size
            Ltf = self.feature_size * self.feature_size

            # src to tgt
            corr_s2t = corr.view(B, Lsi, Lti).transpose(1,2).view(B, Lti, hsi, wsi)
            corr_s2t_resize = F.interpolate(corr_s2t, self.feature_size, None, 'bilinear', True) # (B, Lti, hsf, wsf)
            
            # tgt to src
            corr_t2s = corr_s2t_resize.view(B, Lti, Lsf).transpose(1,2).view(B, Lsf, hti, wti) # (B, Lsf, hti, wti)
            corr_t2s_resize =  F.interpolate(corr_t2s, self.feature_size, None, 'bilinear', True) # (B, Lsf, htf, wtf)

            corrs_resize.append(corr_t2s_resize)

        corrs_resize = torch.stack(corrs_resize, dim=1) # (B, nlayer, Lsf, hf, wf)

        refined_corr = torch.mean(corrs_resize, dim=1) # refined_corr shape: (b, hf, wf)

        grid_x, grid_y = self.soft_argmax(refined_corr.view(B, -1, self.feature_size, self.feature_size)) # grid_x/y (b,1,hf,wf)

        flow_norm = torch.cat((grid_x, grid_y), dim=1) # (B, 2, ht, wt)
        flow = unnormalise_and_convert_mapping_to_flow(flow_norm)

        return flow

    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)
    
    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
    
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        # input corr (src, trg)
        # output grid_x, grid_y means: for each loc in trg img, what is the mapping in the src img.
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y

# SCE crisscross + diags
class EfficientSpatialContextNet(nn.Module):
    def __init__(self, kernel_size=5, input_channel=1024, output_channel=1024, use_cuda=True):
        super(EfficientSpatialContextNet, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            input_channel + 4*self.kernel_size,
            output_channel,
            1,
            bias=True,
            padding_mode="zeros",
        )
        print('verbose...EfficientSpatialContextNet ksize {}, input channel {}, output channel {}'.format(kernel_size, input_channel, output_channel))

        if use_cuda:
            self.conv = self.conv.cuda()

    def forward(self, feature):
        b, c, h, w = feature.size()
        feature_normalized = F.normalize(feature, p=2, dim=1)
        feature_pad = F.pad(
            feature_normalized, (self.pad, self.pad, self.pad, self.pad), "constant", 0
        )
        output = torch.zeros(
            [4*self.kernel_size, b, h, w],
            dtype=feature.dtype,
            requires_grad=feature.requires_grad,
        )
        if feature.is_cuda:
            output = output.cuda(feature.get_device())
        
        # left-top to right-bottom
        for i in range(self.kernel_size):
            c=i
            r=i
            output[i] = (feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized).sum(1)

        # col
        for i in range(self.kernel_size):
            c=self.kernel_size//2
            r=i
            output[1*self.kernel_size+i] = (feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized).sum(1)

        # right-top to left-bottom
        for i in range(self.kernel_size):
            c=(self.kernel_size-1)-i
            r=i
            output[2*self.kernel_size+i] = (feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized).sum(1)
        
        # row
        for i in range(self.kernel_size):
            c=i
            r=self.kernel_size//2
            output[3*self.kernel_size+i] = (feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized).sum(1)

        output = output.transpose(0, 1).contiguous()
        output = torch.cat((feature, output), 1)
        output = self.conv(output)
        output = F.relu(output)

        return output

r'''
Modified from CATs
https://github.com/SunghwanHong/Cost-Aggregation-transformers
'''
class FeatureExtraction(nn.Module):
    def __init__(self, hyperpixel_ids, feature_size, freeze=True):
        super().__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids
        
        self.bid_selected = []
        self.lid_selected = []
    
    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                self.bid_selected.append(bid)
                self.lid_selected.append(lid)

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)

def unnormalise_and_convert_mapping_to_flow(map):
    r'''
    Copy-pasted from GLU-Net
    https://github.com/PruneTruong/GLU-Net
    '''

    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1) # xx (b, 1, hf, wf) ranges [0, W-1]
    yy = yy.view(1,1,H,W).repeat(B,1,1,1) # yy (b, 1, hf, wf) ranges [0, H-1]
    grid = torch.cat((xx,yy),1).float() # grid (b, 2, hf, wf)

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid # flow (b, 2, hf, wf)
    return flow