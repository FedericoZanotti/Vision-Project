from __future__ import division
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
from time import sleep


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


centers = {}
counter = 0
counter_car = 0
counter_truck = 0
counter_other=0
counter_up=0
counter_down=0
pos_line = 350 # 550 for car_traffic_toy # 350 car_traffic_2
offset = 6
second_counter=0
passed = []

i = 0


def draw_and_write(label, c1, c2, color, img, temp_l=[], count=-1, first_frame=False):
    font = cv2.FONT_HERSHEY_PLAIN
    fontscale = 1
    thickness = 1


    if first_frame:  # first frame or object is not longer in the video
        t_size = cv2.getTextSize(str(count), font, fontscale, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, -1)
        cv2.putText(img, str(count), (int(c1[0]), int(c1[1]) + t_size[1] + 4), font,
                    fontscale, [225, 255, 255], thickness)
    else:

        t_size = cv2.getTextSize(str(temp_l[0]), font, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, -1)
        cv2.putText(img,str(temp_l[0]), (int(c1[0]), int(c1[1]) + t_size[1] + 4), font,
                    fontscale, [225, 255, 255], thickness)

def update_counters(label):
    global counter_car
    global counter_truck
    global counter_other
    if label == 'car':
        counter_car += 1
    elif label == 'truck':
        counter_truck += 1
    else:
        counter_other += 1



def write(x, img):
    global counter
    global centers
    global counter_car
    global counter_truck
    global counter_other
    global second_counter
    global counter_up
    global counter_down
    global passed
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())

    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label == 'car':
        color = (0, 0, 255)
    elif label == 'truck':
        color = (255, 0, 0)
    elif label == 'motorbike':
        color = (0, 255, 255)
    else:
        return img
    center = find_center(int(c1[0]), int(c1[1]), int(c2[0]) - int(c1[0]), int(c2[1]) - int(c1[1]))
    if center[1]<200: # <300 for car_traffic_toy <200 for car_traffic_night_2
        return img
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 1)

    updated = False
    temp_l = [12, (1, 1)]

    if frames == 0:  # if is the first frame inizialize structure centers = {id:(x,y)} (x,y) coordinates of centers
        if center not in centers.values():
            counter += 1
            centers[counter] = center
            draw_and_write(label, c1, c2, color, img, count=counter, first_frame=True)
            update_counters(label)


    else:
        best_d = 100
        for k, v in centers.items():  # for every car compute euclidean distance between centers of new object and the objects in memory
            x = v[0]
            y = v[1]
            x_new = center[0]  # new object to update
            y_new = center[1]
            euclidian_distance = np.sqrt(np.power(y - y_new, 2) + np.power(x - x_new, 2))
            if euclidian_distance < best_d:  # if statement to find minimum euclidean distance
                best_d = euclidian_distance
                temp_l[0] = k  # temp list = [id_obj, (x,y)] id_obj is the one with best euclidean distance.
                temp_l[1] = (x_new, y_new)
        if best_d < 20:  # if the best value as an important value (<40 for car_traffic_toy ) (<20 for car night)
            centers[temp_l[0]] = temp_l[1]  # update object
            updated = True
            draw_and_write(label, c1, c2, color, img, temp_l=temp_l)

    if not updated and frames != 0:  # if not updated we have another object

        counter += 1
        update_counters(label)
        centers[counter] = center
        draw_and_write(label, c1, c2, color, img, count=counter, first_frame=True)

    draw_x = int(img.shape[0]*0.05)
    draw_y = int(img.shape[1]*0.05)
    draw_x_legend = int(img.shape[0]+100)
    y_test = int(temp_l[1][1])

    if y_test<(pos_line+offset) and y_test > (pos_line-offset):
        id_vehicle = list(centers.keys())[list(centers.values()).index(temp_l[1])]
        print(f"I have seen vehicle {id_vehicle} passing trough the line")
        if len(passed)==0:
            passed.append(id_vehicle)
            second_counter += 1
            if center[0]>img.shape[0]:
                counter_up += 1
            else:
                counter_down += 1

        elif id_vehicle not in passed:
            passed.append(id_vehicle)
            second_counter += 1
            if center[0] > int(img.shape[1]/2):
                counter_up += 1
            else:
                counter_down += 1

        else:
            pass





    cv2.putText(img, 'vehicle counted: ' + str(second_counter), (draw_x, draw_y), cv2.FONT_HERSHEY_SIMPLEX,
                1, [225, 225, 225], 2)
    cv2.putText(img, 'UP: ' + str(counter_up), (draw_x, draw_y+50), cv2.FONT_HERSHEY_SIMPLEX,
                1, [225, 225, 225], 2)
    cv2.putText(img, 'DOWN: ' + str(counter_down), (draw_x, draw_y+100), cv2.FONT_HERSHEY_SIMPLEX,
                1, [225, 225, 225], 2)
    cv2.putText(img, 'Cars counted: ' + str(counter_car), (draw_x_legend, draw_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, [0, 0, 255], 2)
    cv2.putText(img, 'Truck counted: ' + str(counter_truck), (draw_x_legend, draw_y+50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, [225, 0, 0], 2)
    cv2.putText(img, 'Other counted: ' + str(counter_other), (draw_x_legend,  draw_y+100), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, [0, 255, 255], 2)

    return img


def arg_parse():
    """
    Parse arguements to the detect module
    
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help=
    "Video to run detection upon",
                        default="car_traffic_toy.avi", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="512", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    print('CUDA:\t\t\t\t', CUDA)
    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()

    videofile = 'car_traffic_night_2_Trim.mp4'
    # videofile = 'car_traffic_toy.mp4'
    print(videofile)
    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'
    global frames
    frames = 0
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./data/toy_night_final.avi', codec, fps, (width, height))
    start = time.time()
    counter = 0
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            list(map(lambda x: write(x, orig_im), output))
            cv2.line(orig_im, (0, pos_line), (orig_im.shape[1], pos_line), (255, 127, 0), 2)
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            frames += 1
            print(frames)
            out.write(orig_im)
           
        else:
            break
