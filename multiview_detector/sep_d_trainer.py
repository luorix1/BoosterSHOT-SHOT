import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class SepDPerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, args, cls_thres=0.4, alpha=1.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        self.args = args
        self.model.criterion = criterion

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None, part='left'):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, map_gt, imgs_gt, _) in enumerate(data_loader):
            if part == 'left':
                s, e = 0, self.model.reducedgrid_shape[1]//2
            else:
                s, e = self.model.reducedgrid_shape[1]//2, self.model.reducedgrid_shape[1]
            part_idx = self.model.part_cam_index[part]

            optimizer.zero_grad()
            imgs_gt = torch.cat(imgs_gt,dim=0).cuda()[part_idx]
            map_gt = map_gt[:,:,:,s:e].cuda()
            alpha = len(imgs_gt) * self.alpha
            loss, map_res = self.model(data, imgs_gt, map_gt, alpha, part=part)
            t_f = time.time()
            t_forward += t_f - t_b
            loss.backward()
            optimizer.step()
            losses += loss.item()

            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
                      'prec: {:.1f}%, recall: {:.1f}%, \tTime: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / batch_idx, t_backward / batch_idx, map_res.max()))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, part='right'):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
            part_idx = self.model.part_cam_index[part]
            imgs_gt = [imgs_gt[i] for i in part_idx]
            if part == 'left':
                s, e = 0, self.model.reducedgrid_shape[1]//2
            else:
                s, e = self.model.reducedgrid_shape[1]//2, self.model.reducedgrid_shape[1]
            map_gt = map_gt[:,:,:,s:e]

            with torch.no_grad():
                map_res, imgs_res = self.model(data, part=part)
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = 0
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                   loss / len(imgs_gt) * self.alpha
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        t1 = time.time()
        t_epoch = t1 - t0

        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = 0, 0, 0, 0
            if not self.args.no_matlab:
                all_gt = np.loadtxt(os.path.abspath(gt_fpath))
                thresh = (250/2)*data_loader.dataset.grid_reduce
                if part == 'right':
                    mask = all_gt[:,1]>thresh
                    move = thresh
                else:
                    mask = all_gt[:,1]<=thresh
                    move = 0
                test_gt = all_gt[mask,:]
                test_gt[:,1] -= move
                test_gt_path = os.path.abspath(os.path.dirname(res_fpath))+'/gt_test.txt'
                np.savetxt(test_gt_path, test_gt, '%d')
                from multiview_detector.evaluation.evaluate import matlab_eval
                recall, precision, moda, modp = matlab_eval(
                    os.path.abspath(res_fpath), os.path.abspath(test_gt_path), data_loader.dataset.base.__name__)
                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                    format(moda, modp, precision, recall))

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100, moda


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
