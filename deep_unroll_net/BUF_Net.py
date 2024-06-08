#! directly learning U_t_0 and U_t_1 (multiply temporal offsets)
# remove weight decay; use flow residual link; use masked warped feature map to compute bilatial correlation
# remove mask residual link
# use warped original features to compute CV
# Next: add trade-off term; use dilated correlation

import os
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from correlation_package import Correlation

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 
    return grid

def add_temporal_offset(F_pred, H, W, t):# 0 <= t <= 1,  -gamma/2 <= tau <= gamma/2
    grid_rows = generate_2D_grid(H, W)[1]
    t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
    
    gamma = 1.0
    tau = gamma*(t_flow_offset-H//2)/H + 0.0001

    F_t_0 = (t-tau) * F_pred[:,0:2]
    F_t_1 = (t-tau-1.0) * F_pred[:,2:4]

    return F_t_0, F_t_1

def add_initial_temporal_offset(H, W, t):# 0 <= t <= 1,  -gamma/2 <= tau <= gamma/2
    grid_rows = generate_2D_grid(H, W)[1]
    t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
    
    gamma = 1.0
    tau = gamma*(t_flow_offset-H//2)/H + 0.0001

    T_init_0 = (t-tau) 
    T_init_1 = (t-tau-1.0)

    return T_init_0, T_init_1

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride,
                  padding = padding, dilation = dilation, bias= True),
        nn.PReLU(out_channels))

def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_channels):
    return nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)


class BUFNet(nn.Module):
    def __init__(self, md=4):#r=4
        super(BUFNet, self).__init__()
        self.md = md
        self.upfeat_ch = [16, 16, 16, 16, 16]##

        self.conv1a  = conv(3,    16, kernel_size=3, stride=1)
        self.conv1aa = conv(16,   16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,   16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,   32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,   32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,   32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,   64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,   64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,   64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,   96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,   96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,   96, kernel_size=3, stride=1)
        self.conv5a  = conv(96,  128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b  = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a  = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b  = conv(196, 196, kernel_size=3, stride=1)

        self.corr   = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2*md + 1)**2 * 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd // 2 + 2
        self.conv6_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.predict_mask6 = predict_mask(od + dd[4])
        self.deconv6 = deconv(4,          4, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], self.upfeat_ch[0], kernel_size=4, stride=2, padding=1)

        od = nd + 128 + self.upfeat_ch[0] + 4
        self.conv5_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.predict_mask5 = predict_mask(od + dd[4])
        self.deconv5 = deconv(4,          4, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], self.upfeat_ch[1], kernel_size=4, stride=2, padding=1)

        od = nd + 96 + self.upfeat_ch[1] + 4
        self.conv4_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.predict_mask4 = predict_mask(od + dd[4])
        self.deconv4 = deconv(4,          4, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], self.upfeat_ch[2], kernel_size=4, stride=2, padding=1)

        od = nd + 64 + self.upfeat_ch[2] + 4
        self.conv3_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.predict_mask3 = predict_mask(od + dd[4])
        self.deconv3 = deconv(4,          4, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], self.upfeat_ch[3], kernel_size=4, stride=2, padding=1)

        od = nd + 32 + self.upfeat_ch[3] + 4
        self.conv2_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.predict_mask2 = predict_mask(od + dd[4])
        self.deconv2 = deconv(4,          4, kernel_size=4, stride=2, padding=1)
        self.upfeat2 = deconv(od + dd[4], self.upfeat_ch[4], kernel_size=4, stride=2, padding=1)

        self.predict_img_residual = nn.Sequential(
                                        conv(self.upfeat_ch[4]+6+3, 3, kernel_size=3, stride=1),
                                        nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
    '''
    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output
    '''
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
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
    
    def bilateral_correlation(self, syn, warp0, warp1):
        bicorr_l = self.leakyRELU(self.corr(syn, warp0))
        bicorr_r = self.leakyRELU(self.corr(syn, warp1))
        bicorr = torch.cat((bicorr_l, bicorr_r), dim=1)

        '''
        self.index = torch.tensor([0, 2, 4, 6, 8, 
                10, 12, 14, 16, 
                18, 20, 21, 22, 23, 24, 26, 
                28, 29, 30, 31, 32, 33, 34, 
                36, 38, 39, 40, 41, 42, 44, 
                46, 47, 48, 49, 50, 51, 52, 
                54, 56, 57, 58, 59, 60, 62, 
                64, 66, 68, 70, 
                72, 74, 76, 78, 80])
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long()) # ICRA2021
        '''

        return bicorr

    def forward(self, x, time=0.5):
        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        T_init_0, T_init_1 = add_initial_temporal_offset(c26.size(2), c26.size(3), time)

        T = torch.cat((T_init_0.repeat(c26.size(0),1,1,1), T_init_1.repeat(c26.size(0),1,1,1)), dim=1) ###
        
        bicorr6 = torch.cat((self.corr(c26, c16), T), dim=1) ###027,127

        x = torch.cat((self.conv6_0(bicorr6), bicorr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        mask6 = self.predict_mask6(x)
        up_mask6 = F.interpolate(mask6, scale_factor=2.0, mode='nearest')
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)
        
        mask_5 = F.sigmoid(up_mask6)
        up_flow6_l, up_flow6_r = add_temporal_offset(up_flow6, up_flow6.size(2), up_flow6.size(3), time)
        warp1_5 = self.warp(c15, up_flow6_l)
        warp2_5 = self.warp(c25, up_flow6_r)
        syn_5 = mask_5 * warp1_5 + (1.0-mask_5) * warp2_5
        bicorr5 =  self.bilateral_correlation(syn_5, warp1_5, warp2_5) ###
        x = torch.cat((bicorr5, syn_5, up_flow6_l, up_flow6_r, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)  + up_flow6##
        mask5 = self.predict_mask5(x)  #+ up_mask6##
        up_mask5 = F.interpolate(mask5, scale_factor=2.0, mode='nearest')
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        mask_4 = F.sigmoid(up_mask5)
        up_flow5_l, up_flow5_r = add_temporal_offset(up_flow5, up_flow5.size(2), up_flow5.size(3), time)
        warp1_4 = self.warp(c14, up_flow5_l)
        warp2_4 = self.warp(c24, up_flow5_r)
        syn_4 = mask_4 * warp1_4 + (1.0-mask_4) * warp2_4
        bicorr4 =  self.bilateral_correlation(syn_4, warp1_4, warp2_4) ###
        x = torch.cat((bicorr4, syn_4, up_flow5_l, up_flow5_r, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)  + up_flow5##
        mask4 = self.predict_mask4(x)  #+ up_mask5##
        up_mask4 = F.interpolate(mask4, scale_factor=2.0, mode='nearest')
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        mask_3 = F.sigmoid(up_mask4)
        up_flow4_l, up_flow4_r = add_temporal_offset(up_flow4, up_flow4.size(2), up_flow4.size(3), time)
        warp1_3 = self.warp(c13, up_flow4_l)
        warp2_3 = self.warp(c23, up_flow4_r)
        syn_3 = mask_3 * warp1_3 + (1.0-mask_3) * warp2_3
        bicorr3 =  self.bilateral_correlation(syn_3, warp1_3, warp2_3) ###
        x = torch.cat((bicorr3, syn_3, up_flow4_l, up_flow4_r, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)  + up_flow4##
        mask3 = self.predict_mask3(x)  #+ up_mask4##
        up_mask3 = F.interpolate(mask3, scale_factor=2.0, mode='nearest')
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        mask_2 = F.sigmoid(up_mask3)
        up_flow3_l, up_flow3_r = add_temporal_offset(up_flow3, up_flow3.size(2), up_flow3.size(3), time)
        warp1_2 = self.warp(c12, up_flow3_l)
        warp2_2 = self.warp(c22, up_flow3_r)
        syn_2 = mask_2 * warp1_2 + (1.0-mask_2) * warp2_2
        bicorr2 =  self.bilateral_correlation(syn_2, warp1_2, warp2_2) ###
        x = torch.cat((bicorr2, syn_2, up_flow3_l, up_flow3_r, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)  + up_flow3##
        mask2 = self.predict_mask2(x)  #+ up_mask3##
        up_mask2 = F.interpolate(mask2, scale_factor=2.0, mode='nearest')
        up_flow2 = self.deconv2(flow2)
        up_feat2 = self.upfeat2(x)

        up_flow2_l, up_flow2_r = add_temporal_offset(up_flow2, up_flow2.size(2), up_flow2.size(3), time)
        warp1_1 = self.warp(im1, up_flow2_l)
        warp2_1 = self.warp(im2, up_flow2_r)
        
        mask_1 = F.sigmoid(up_mask2)
        
        im_t_ = mask_1 * warp1_1 + (1.0-mask_1) * warp2_1
        up_feat2 = torch.cat((up_feat2, warp1_1, warp2_1, im_t_), 1)

        delta_im = self.predict_img_residual(up_feat2)

        im_t = im_t_ + delta_im
        
        return torch.clamp(im_t, 0, 1),\
             [up_flow2_l, up_flow2_r, mask_1],\
             [up_flow3_l, up_flow3_r, mask_2],\
             [up_flow4_l, up_flow4_r, mask_3],\
             [up_flow5_l, up_flow5_r, mask_4],\
             [up_flow6_l, up_flow6_r, mask_5],\
             delta_im

