import argparse
import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from multiview_detector.models.dpersp_trans_detector import DPerspTransDetector
from multiview_detector.utils import projection
from multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap, add_heatmap_to_image
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.datasets import *


def main(args):
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    # dataset
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('/workspace/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('/workspace/Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              num_workers=4, pin_memory=True)

    # model and checkpoint loading
    model = DPerspTransDetector(test_set, args.arch, args.depth_scales)
    ckpt = torch.load(os.path.join(args.ckpt_path, 'MultiviewDetector.pth'))
    model.load_state_dict(ckpt)

    # create directory
    os.makedirs(os.path.join(args.ckpt_path, 'heatmap'), exist_ok=True)

    heatmap = torch.zeros((base.num_cam, args.depth_scales, args.bottleneck_dim))
    
    for batch_idx, (imgs, _, _, _) in enumerate(test_loader):
        print(f'Running batch {batch_idx + 1}')
        B, N, C, H, W = imgs.shape
        img_feature_all = model.base_pt1(imgs.view([-1,C,H,W]).to('cuda:0'))
        img_feature_all = model.base_pt2(img_feature_all.to('cuda:0'))
        F.interpolate(img_feature_all, model.upsample_shape, mode='bilinear').to('cuda:0')
        img_feature_all = model.feat_down(img_feature_all.to('cuda:0'))
        depth_select = model.depth_classifier(img_feature_all).softmax(dim=1)

        if batch_idx == 0:
            for i in range(args.depth_scales):
                heatmap = depth_select[:, i][:, None]
                for cam in range(N):
                    img = denormalize(imgs.view([-1,C,H,W])[cam]).detach().cpu().numpy().squeeze().transpose([1, 2, 0])
                    img = Image.fromarray((img * 255).astype('uint8'))
                    heatmap_img = torch.norm(heatmap[cam].detach(), dim=0).cpu().numpy()
                    visualize_img = add_heatmap_to_image(heatmap_img, img)
                    visualize_img.save(os.path.join(args.ckpt_path, f'heatmap/heatmap_{i}_{cam + 1}.png'))

    pass        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--ckpt_path', help='Absolute path of parent directory')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('--depth_scales', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    
    args = parser.parse_args()
    main(args)