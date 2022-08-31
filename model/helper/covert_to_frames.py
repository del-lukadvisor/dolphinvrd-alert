'''
__author__ = 'Ringo S.W. Chu'
__Description__ = 'Covert ImgNet Video to Frames
'''

from __future__ import absolute_import, division, print_function
from model.helper.parser import GeneralParser
import os, glob, json
import cv2
from tqdm import tqdm

import numpy as np

def read_video(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise Exception('Cannot open {}'.format(path))
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    return video

def main():

    parser = GeneralParser()
    parse_aug = parser.parse()

    data_path = parse_aug.data_path
    print(f'Your path to data is: {data_path}')

    #Set this variable to either 'train'/'test'
    train_test_set = 'test'

    #Json location
    all_json_annotation = sorted(glob.glob(os.path.join(data_path, train_test_set ,'*.json')))

    #Save destination
    save_destination = os.path.join(data_path, train_test_set)

    #Video location
    video_location = os.path.join(data_path, '')

    for js in tqdm(all_json_annotation):

        # Reading from file
        anno = json.loads(open(js, "r").read())

        height = anno['height']
        width = anno['width']
        fps = anno['fps']
        frame_count = anno['frame_count']
        trajectories = anno['trajectories']
        subject_objects = anno['subject/objects']
        relation_instances = anno['relation_instances']

        video_id = anno['video_id']

        # Video Path
        this_video_path = os.path.join(video_location, 'videos', video_id+'.mp4')
        video_frames = read_video(this_video_path)

        assert anno['frame_count'] == len(video_frames), \
            '{} : anno {} video {}'.format(anno['video_id'], anno['frame_count'], len(video_frames))
        assert anno['width'] == video_frames[0].shape[1] and anno['height'] == video_frames[0].shape[0], \
            '{} : anno ({}, {}) video {}'.format(anno['video_id'], anno['height'], anno['width'], video_frames[0].shape)

        video_destination = os.path.join(save_destination, video_id)
        if not os.path.exists(video_destination):
            print(f'Create Directory {video_destination}')
            os.mkdir(video_destination)

        for idx, f in zip(range(1, len(video_frames)+1), video_frames):
            cv2.imwrite(os.path.join(video_destination, str(idx).zfill(6)+'.jpeg'), f)

        # Visualise
        ratio = 720.0 / height
        boundary = 20
        size = int(round(width * ratio)) + 2 * boundary, int(round(height * ratio)) + 2 * boundary
        for i in range(anno['frame_count']):
            background = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            background[boundary:size[1] - boundary, boundary:size[0] - boundary] = cv2.resize(video_frames[i], (
            size[0] - 2 * boundary, size[1] - 2 * boundary))
            video_frames[i] = background

if __name__ == '__main__':
    main()
