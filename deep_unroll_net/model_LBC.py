#! supervision: t = 0.5, 0.0, 1.0

import sys
import random
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from package_core.net_basics import *
from package_core.model_base import *
from package_core.losses import *
from package_core.flow_utils import *
from package_core.image_proc import *
from flow_loss import *

from BUF_Net import BUFNet

class ModelLBC(ModelBase):
    def __init__(self, opts):
        super(ModelLBC, self).__init__()
        
        self.opts = opts
        
        # create networks
        self.model_names=['flow']
        self.net_flow = BUFNet().cuda()
        
        # load in initialized network parameters
        if not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)
        
        if self.opts.is_training:
            # initialize optimizers
            
            self.optimizer_G = torch.optim.Adam([{'params': self.net_flow.parameters()}], lr=opts.lr)

            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            self.loss_fn_tv2 = VariationLoss(nc=2)

            self.downsampleX2 = nn.AvgPool2d(2, stride=2)
            
            ###Initializing VGG16 model for perceptual loss
            self.MSE_LossFn = nn.MSELoss()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
            self.vgg16_conv_4_3.to('cuda')
            for param in self.vgg16_conv_4_3.parameters():
                param.requires_grad = False

    def set_input(self, _input):
        im_rs, im_gs, gt_flow, im_gs_f = _input
        self.im_rs = im_rs.cuda()
        self.im_gs = im_gs
        self.gt_flow = gt_flow
        self.im_gs_f = im_gs_f

        if self.im_gs is not None:
            self.im_gs = self.im_gs.cuda()
        if self.im_gs_f is not None:
            self.im_gs_f = self.im_gs_f.cuda()
        if self.gt_flow is not None:
            self.gt_flow = self.gt_flow.cuda()

    def forward(self, time):# time is in [0, 1]
        
        gs_t_final, Fs_lv_0, Fs_lv_1, Fs_lv_2, Fs_lv_3, Fs_lv_4, delta_im = self.net_flow(self.im_rs, time)
        
        if self.opts.is_training:
            return gs_t_final, Fs_lv_0, Fs_lv_1, Fs_lv_2, Fs_lv_3, Fs_lv_4
        
        return gs_t_final, Fs_lv_0, Fs_lv_1, Fs_lv_2, Fs_lv_3, Fs_lv_4, delta_im


    def optimize_parameters(self):
        
        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1 = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_L1_multiscale = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_flow_smoothness = torch.tensor([0.], requires_grad=True).cuda().float()

        times = [0.0, 0.5, 1.0]#train Carla-RS and Fastec-RS for RSSR!!!
        #times = [1.0]#train BS-RSC for RSC!!!
        
        num=5#pyramid level: L=6
        for tt in range(len(times)):
            if times[tt] == 0.5:
                im_gt = self.im_gs_f[:,3:6,:,:].clone()
            if times[tt] == 0.0:
                im_gt = self.im_gs[:,0:3,:,:].clone()
            if times[tt] == 1.0:
                im_gt = self.im_gs[:,3:6,:,:].clone()
            
            pred_im, Fs_lv_0, Fs_lv_1, Fs_lv_2, Fs_lv_3, Fs_lv_4 = self.forward(times[tt])
            #===========================================================#
            #                       Compute losses                      #
            #===========================================================#
            gs_gt = im_gt.clone()
            rs_gt = self.im_rs.clone()
            
            pyr_weights = [1.0, 0.5, 0.25, 0.25, 0.25]
            pred_F = [Fs_lv_0]
            pred_F.append(Fs_lv_1)
            pred_F.append(Fs_lv_2)
            pred_F.append(Fs_lv_3)
            pred_F.append(Fs_lv_4)
            
            for l in range(1, num):
                if l > 0:
                    gs_gt = self.downsampleX2(gs_gt)
                    rs_gt = self.downsampleX2(rs_gt)
            
                warped_img0 = self.warp(rs_gt[:,0:3], pred_F[l][0])
                warped_img1 = self.warp(rs_gt[:,3:6], pred_F[l][1])

                pred_gs = pred_F[l][2] * warped_img0 + (1.0-pred_F[l][2]) * warped_img1

                self.loss_L1_multiscale += pyr_weights[l] * self.charbonier_loss(pred_gs, gs_gt, mean=True) * self.opts.lamda_L1 *10.0
                
            self.loss_L1 += self.charbonier_loss(pred_im, im_gt, mean=True) * self.opts.lamda_L1 *10.0
            self.loss_perceptual += self.loss_fn_perceptual.get_loss(pred_im, im_gt) * self.opts.lamda_perceptual 
            
            self.loss_flow_smoothness += self.opts.lamda_flow_smoothness * (self.loss_fn_tv2(Fs_lv_0[0], mean=True) + self.loss_fn_tv2(Fs_lv_0[1], mean=True)) 

            
        
        self.loss_L1_multiscale = self.loss_L1_multiscale / ((num-1.0) * len(times))
        self.loss_perceptual = self.loss_perceptual / len(times)
        self.loss_L1 = self.loss_L1 / len(times)
        self.loss_flow_smoothness = self.loss_flow_smoothness / (2.0 * len(times))

        # sum them up
        self.loss_G = self.loss_L1 +\
                        self.loss_perceptual +\
                        self.loss_L1_multiscale +\
                        self.loss_flow_smoothness
                        
        # Optimize 
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

    def charbonier_loss(self, pred_im, im_gt, epsilon=0.001, mean=True):
        x = pred_im - im_gt
        loss = torch.mean(torch.sqrt(x ** 2 + epsilon ** 2))
        return loss
    
    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)

    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)

    def get_current_scalars(self):
        losses = {}
        losses['loss_G'] = self.loss_G.item()
        losses['loss_L1'] = self.loss_L1.item()
        losses['loss_percep'] = self.loss_perceptual.item()
        losses['loss_L1_mulscale'] = self.loss_L1_multiscale.item()
        losses['loss_flow_smooth'] = self.loss_flow_smoothness.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()
        output_visuals['im_gs'] = self.im_gs_f[:,-3:,:,:].clone()
        
        return output_visuals
