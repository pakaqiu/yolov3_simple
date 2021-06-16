from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


import cv2



def pad_to_square(img1, pad_value):

    img = img1.transpose(2,0,1)
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = torch.from_numpy(img)
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def pad_to_hw(img1, pad_value):

    img = img1.transpose(2,0,1)
    c, h, w = img.shape
    dim_diff = 0
    if h < w:
        r = np.round(h / 32 + 0.5)
        dim_diff = int(r * 32 - h)
    else:
        r = np.round(w / 32 + 0.5)
        dim_diff = int(r * 32 - w)

    #dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = torch.from_numpy(img)
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad, dim_diff


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img[:]

    img, _ = pad_to_square(img, 0)


    dim = orig_im.shape[1], orig_im.shape[0]
    img = resize(img, inp_dim)
    img = img.numpy()
    img_resize = img.transpose(1,2,0)
    #img = cv2.resize(orig_im, (inp_dim, inp_dim))
    #img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = img.copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim, img_resize


def prep_image1(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    ratio1 = inp_dim/dim[0]
    ratio2 = inp_dim/dim[1]
    if ratio1 > ratio2:
        ratio1 = ratio2

    out_dim0 = int(dim[0] * ratio1)
    out_dim1 = int(dim[1] * ratio1)

    img = cv2.resize(orig_im, (out_dim0, out_dim1))

    img,_,diff = pad_to_hw(img,0)
    img = img.numpy()

    img_resize = img[:]
    #img_resize = img_resize.transpose(2, 0, 1)
    #img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim,img_resize,diff





'''def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim'''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/INA-T", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny-shoulders.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default = "checkpoint/yolov3_ckpt_165.pth",help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default = "checkpoint/yolov3_ckpt_165.pth", help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        print('load check point')
        model.load_state_dict(torch.load(opt.checkpoint_model,map_location=torch.device('cpu')))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    #cmap = plt.get_cmap("tab20b")
    #colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    while cap.isOpened():

        ret,frame = cap.read()

        if ret:
            img,orig_im,dim,img_resize = prep_image(frame,opt.img_size)

            input_imgs = Variable(img.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            #length = len(detections[0])
            #print('detection.len = ',len(detections))

            if detections[0] is not None:

                # Rescale boxes to original image
                detections = rescale_boxes_video(detections, opt.img_size, frame.shape[:2])
                #detections = rescale_boxes_video(detections, img.shape[2:], frame.shape[:2],diff)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

            cv2.imshow('img',frame)
            cv2.waitKey(10)
