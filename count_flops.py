import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.transforms as T

from ptflops import get_model_complexity_info

from multiview_detector.models.boostershot import BoosterSHOT
from multiview_detector.models.dpersp_trans_detector import DPerspTransDetector
from multiview_detector.utils import projection
from multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap, add_heatmap_to_image
from multiview_detector.datasets import *


def _target_transform(target, kernel):
    with torch.no_grad():
        target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
    return target

def main(args):
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    
    base = Wildtrack(os.path.expanduser('/workspace/Data/Wildtrack'))

    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                                num_workers=4, pin_memory=True)

    with torch.cuda.device(0):
        if args.model == 'BoosterSHOT':
            net = BoosterSHOT(test_set, args.arch, args.depth_scales, args.topk)
        elif args.model == 'SHOT':
            net = DPerspTransDetector(test_set, args.arch, args.depth_scales)
        else:
            raise Exception('This model is not supported.')
        macs, params = get_model_complexity_info(net, (7, 3, 720, 1280), as_strings=False,
                                                print_per_layer_stat=False, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('GFLOPs: ', 2 * macs / 1000 / 1000 / 1000))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count the computational cost of a model')
    parser.add_argument('--model', choices=['SHOT', 'BoosterSHOT'])
    parser.add_argument('--depth_scales', type=int, default=4)
    parser.add_argument('--topk', default=None)
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'vgg11'])

    args = parser.parse_args()

    main(args)

# FLOPs = 2 * MACs
# FLOPs = Floating point operations
# MACs = Multiply–accumulate operations