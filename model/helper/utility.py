'''
Author = Ringo SW Chu
This file contains code that is modified from the below repository:
https://github.com/nalepae/bounding-box/blob/master/bounding_box/bounding_box.py

Doing so could prevent a deperciation from the future.
'''

from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import os.path as _path
import numpy as np
import cv2 as _cv2
from PIL import ImageFont
import numpy as _np
from hashlib import md5 as _md5

#_LOC = _path.realpath(_path.join(_os.getcwd(),_path.dirname(__file__)))

import torch
def cpu_or_gpu(device):
    if (device.lower() in ['cuda', 'gpu']) and torch.cuda.is_available():
        device = torch.device('cuda')
    #elif torch.cuda.is_available():
    #    device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

def git_root(*args):
    import subprocess
    import os

    git_root = subprocess.Popen(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    return os.path.abspath(os.path.join(git_root, *args))

def generate_random_colour(integer=False):
    if integer:
        return [int(np.random.choice(range(256), size=1)) for _ in range(3)]
    else:
        return [np.random.uniform() for _ in range(3)]

#https://clrs.cc/
_COLOR_NAME_TO_RGB = dict(
    navy=((0, 38, 63), (119, 193, 250)),
    blue=((0, 120, 210), (173, 220, 252)),
    aqua=((115, 221, 252), (0, 76, 100)),
    teal=((15, 205, 202), (0, 0, 0)),
    olive=((52, 153, 114), (25, 58, 45)),
    green=((0, 204, 84), (15, 64, 31)),
    lime=((1, 255, 127), (0, 102, 53)),
    yellow=((255, 216, 70), (103, 87, 28)),
    orange=((255, 125, 57), (104, 48, 19)),
    red=((255, 47, 65), (131, 0, 17)),
    maroon=((135, 13, 75), (239, 117, 173)),
    fuchsia=((246, 0, 184), (103, 0, 78)),
    purple=((179, 17, 193), (241, 167, 244)),
    black=((24, 24, 24), (220, 220, 220)),
    gray=((168, 168, 168), (0, 0, 0)),
    silver=((220, 220, 220), (0, 0, 0)),
    white=((0,0,0), (0,0,0)))

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB)
_DEFAULT_COLOR_NAME = "navy"

_FONT_PATH = _os.path.join(git_root(), 'doc', "ubuntu-b.ttf")
_FONT_HEIGHT = 15
_FONT = ImageFont.truetype(_FONT_PATH, _FONT_HEIGHT)

def plot_traj(clip, traj, motion=None):
    
    clip_np = [_cv2.imread(c['img_path']) for c in clip]

    for id, cood in traj.items():
        
        length_cood = len(cood['traj'])
        for j in range(length_cood):
        
            bbox = cood['traj'][j]
            x_min, y_min, x_max, y_max = [ int(b.item()) for b in bbox]
            
            this_img = clip_np[j]
            this_img = _cv2.rectangle(this_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

            font = _cv2.FONT_HERSHEY_SIMPLEX
            centre_x = int(x_min/2 + x_max/2)
            centre_y = int(y_min/2 + y_max/2)
            this_img = _cv2.putText(this_img, f'ID: {id}', (centre_x, centre_y), font, 0.7, (255, 255, 0), 2, _cv2.LINE_AA)
            
            if j >13: # Last frame
                motion =  np.argmax(cood['motion'].data)
                this_img = _cv2.putText(this_img, f'Motion: {motion}', (centre_x, centre_y+10), font, 0.7, (255, 255, 0), 2, _cv2.LINE_AA)
            
            clip_np[j] = this_img

    return clip_np


def _rgb_to_bgr(color):
    return list(reversed(color))


def _color_image(image, font_color, background_color):
    return background_color + (font_color - background_color) * image / 255


def _get_label_image(text, font_color_tuple_bgr=(255,255,255), background_color_tuple_bgr=(0,0,0)):
    text_image = _FONT.getmask(text)
    shape = list(reversed(text_image.size))
    bw_image = np.array(text_image).reshape(shape)
    image = [
        _color_image(bw_image, font_color, background_color)[None, ...]
        for font_color, background_color
        in zip(font_color_tuple_bgr, background_color_tuple_bgr)
    ]
    return np.concatenate(image).transpose(1, 2, 0)


def add_straight_line(frame, p1, p2, p1_name, p2_name, predicate, colour=None):

    if colour is None:
        colour = generate_random_colour()

    frame = _cv2.line(frame, p1, p2, colour, 2)

    predicate_image = _get_label_image(predicate)
    # Straight Line mid-point
    mid_y, mid_x = int(round((p1[0] + p2[0])/2)), int(round((p1[1] + p2[1])/2))
    label_height, label_width, _ = predicate_image.shape
    try:
        frame[mid_x:mid_x+label_height, mid_y:mid_y+label_width, :] = predicate_image
    except ValueError as ErrMsg:
        print(f"\tPredicate Box lies outside Frame: {ErrMsg}")
        print(f'Shape of the frame {frame.shape}, Label_image_size: {predicate_image.shape},')
        print(f' Predicate Box Location at X: {p1[1]} - {p1[1]+label_height}')
        print(f' Predicate Box Location at Y: {p1[0]} - {p1[0]+label_width}')
        print(f" You should clip the boxes using torch.ops built-in functions")

    sub_image = _get_label_image(p1_name)
    sub_label_height, sub_label_width, _ = sub_image.shape
    try:
        frame[p1[1]:p1[1]+sub_label_height, p1[0]:p1[0]+sub_label_width, :] = sub_image
    except ValueError as ErrMsg:
        print(f"\tSubject Bounding Box lies outside Frame: {ErrMsg}")
        print(f'Shape of the frame {frame.shape}, Label_image_size: {sub_image.shape},')
        print(f' Subject Box Location at X: {p1[1]} - {p1[1]+sub_label_height}')
        print(f' Object Box Location at Y: {p1[0]} - {p1[0]+sub_label_width}')
        print(f" You should clip the boxes using torch.ops built-in functions")

    obj_image = _get_label_image(p2_name)
    obj_label_height, obj_label_width, _ = obj_image.shape
    try:
        frame[p2[1]:p2[1]+obj_label_height, p2[0]:p2[0]+obj_label_width, :] = obj_image
    except ValueError as ErrMsg:
        print('\tObject Bounding Box lies outside the frame')
        print(f"\tObject Bounding Box lies outside Frame: {ErrMsg}")
        print(f'Shape of the frame {frame.shape}, Label_image_size: {obj_image.shape},')
        print(f' Object Box Location at X: {p1[1]} - {p1[1]+obj_label_height}')
        print(f' Object Box Location at Y: {p1[0]} - {p1[0]+obj_label_width}')
        print(f" You should clip the boxes using torch.ops built-in functions")
    return frame

def add_bbox(image, left, top, right, bottom, label=None, color=None):
    if type(image) is not _np.ndarray:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number")

    if label and type(label) is not str:
        raise TypeError("'label' must be a str")

    if label and not color:
        hex_digest = _md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 16) % len(_COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]

    if not color:
        color = _DEFAULT_COLOR_NAME

    if type(color) is not str:
        raise TypeError("'color' must be a str")

    if color not in _COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ", ".join(_COLOR_NAME_TO_RGB)
        raise ValueError(msg)

    colors = [_rgb_to_bgr(item) for item in _COLOR_NAME_TO_RGB[color]]
    color, color_text = colors

    color = [c/255 for c in color]
    _cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    if label:
        _, image_width, _ = image.shape
        label_image = _get_label_image(label, (255,255,255), (0,0,0))

        label_height, label_width, _ = label_image.shape

        rectangle_height, rectangle_width = 1 + label_height, 1 + label_width

        rectangle_bottom = top
        rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width

        label_top = rectangle_top + 1

        if rectangle_top < 0:
            rectangle_top = top
            rectangle_bottom = rectangle_top + label_height + 1

            label_top = rectangle_top

        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        label_right = label_left + label_width

        rec_left_top = (rectangle_left, rectangle_top)
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        _cv2.rectangle(image, rec_left_top, rec_right_bottom, color, -1)

        image[label_top:label_bottom, label_left:label_right, :] = label_image

    return image