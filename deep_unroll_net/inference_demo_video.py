import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil

import imageio
import cv2
import flow_viz

from package_core.generic_train_test import *
from dataloader import *
from frame_utils import *
from model_LBC import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=True, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=False)
parser.add_argument('--crop_sz_H', type=int, default=448, help='cropped image size height')
parser.add_argument('--crop_sz_W', type=int, default=640, help='cropped image size width')

parser.add_argument('--model_label', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--iters', type=int, default=12)

parser.add_argument('--is_Fastec', type=int, default=0)

opts=parser.parse_args()

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelLBC(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Demo(Generic_train_test):
    def test(self):
        with torch.no_grad():
            seq_lists = os.listdir(self.opts.data_dir)
            for seq in seq_lists:
                im_rs0_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_0.png')
                im_rs1_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_1.png')

                im_rs0 = torch.from_numpy(io.imread(im_rs0_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()
                im_rs1 = torch.from_numpy(io.imread(im_rs1_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()

                im_rs = torch.cat([im_rs0,im_rs1], dim=1).float()/255.
                
                _input = [im_rs, None, None, None]
                
                self.model.set_input(_input)
                
                # save original RS images
                im_rs_0 = io.imread(im_rs0_path)
                im_rs_1 = io.imread(im_rs1_path)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_0.png'), im_rs_0)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1.png'), im_rs_1)
                
                # generate GS images for any row
                preds_0=[]
                preds_0_tensor=[]
                copies = 11
                for t in range(0, copies):
                    #convert to GS of t-th moment
                    pred_gs_t,_,_,_,_,_,_ = self.model.forward(t/(copies-1))
                    
                    # save results
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_'+str(t)+'.png'), (pred_gs_t.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    
                    preds_0.append((pred_gs_t.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    preds_0_tensor.append(pred_gs_t)

                    print('saved', self.opts.results_dir, seq+'_pred_'+str(t)+'.png')
                    
                    
                pred_imgs_rec=preds_0_tensor
                img_rec = pred_imgs_rec[0].clone()
                for t in range(1,copies):
                    img_rec *= (pred_imgs_rec[t] * 255)
                    img_rec = img_rec.clamp(0,1)
                    
                #make gif
                make_gif_flag = True
                if make_gif_flag:
                    pred_imgs_gif=preds_0
                    
                    #imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_duration_'+str(0.5)+'.gif'), pred_imgs_gif, duration = 0.5) # modify the frame duration as needed
                    imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_duration_'+str(0.1)+'.gif'), pred_imgs_gif, duration = 0.1) # modify the frame duration as needed
                
                print('\n')
                
Demo(model, opts, None, None).test()


