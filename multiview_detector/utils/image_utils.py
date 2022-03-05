import numpy as np
import cv2
from PIL import Image
import torch


class img_color_denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1])
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1])

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


# def add_heatmap_to_image(heatmap, image):
#     heatmap = heatmap - heatmap.min()
#     heatmap = heatmap / (heatmap.max() + 1e-8)
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.resize(heatmap, (image.size))
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#     cam_result = np.uint8(heatmap * 0.3 + image * 0.5)
#     cam_result = Image.fromarray(cv2.cvtColor(cam_result, cv2.COLOR_BGR2RGB))
#     return cam_result

class img_color_denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1])
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1])

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


def add_heatmap_to_image(heatmap, image):
    heatmap = cv2.resize(np.array(array2heatmap(heatmap)), (image.size))
    cam_result = np.uint8(heatmap * 0.3 + np.array(image) * 0.5)
    cam_result = Image.fromarray(cam_result)
    return cam_result


def array2heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return heatmap
