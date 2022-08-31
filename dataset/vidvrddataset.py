'''
Modified from: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
'''
from __future__ import print_function, division

import os, glob, json
from PIL import Image
import cv2
import numpy as np

from .generaldataset import GeneralDataset
from dataset.image_transform import ImglistToTensor
from model.helper.utility import add_bbox, add_straight_line
from model.helper.utility import generate_random_colour, _COLOR_NAME_TO_RGB
from collections import defaultdict
import model.helper.vision.transforms as T

import torch

from dataset.vg.dataset import Dataset

class VidVRD(Dataset):
    """
    VidVRD dataset introduced in ACM MM'17
    """

    def __init__(self, anno_rpath, video_rpath, splits):
        """
        anno_rpath: the root path of annotations
        video_rpath: the root path of videos
        splits: a list of splits in the dataset to load
        """
        super(VidVRD, self).__init__(anno_rpath, video_rpath, splits)
        print('VidVRD dataset loaded.')

    def _get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.anno_rpath, '{}/*.json'.format(split)))
        assert len(anno_files) > 0, 'No annotation file found. Please check if the directory is correct.'
        return anno_files

    def get_video_path(self, vid, imagenet_struture=False):
        """
        True if the directory videos uses imagenet struture
        """
        if imagenet_struture:
            if 'train' in vid:
                path = glob.glob(os.path.join(self.video_rpath,
                                              'Data/VID/snippets/train/*/{}.mp4'.format(vid)))[0]
            elif 'val' in vid:
                path = os.path.join(self.video_rpath,
                                    'Data/VID/snippets/val/{}.mp4'.format(vid))
            else:
                raise Exception('Unknown video ID {}'.format(vid))
        else:
            path = os.path.join(self.video_rpath, '{}.mp4'.format(vid))
        return path


if __name__ == '__main__':
    """
    To generate a single JSON groundtruth file for specific split and task,
    run this script from the parent directory, for example, 
    python -m dataset.vidvrd test relation ~/vidvrd_gt_test_relation.json
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Generate a single JSON groundtruth file for VidVRD')
    parser.add_argument('split', choices=['train', 'test'],
                        help='which dataset split the groundtruth generated for')
    parser.add_argument('task', choices=['object', 'relation'],
                        help='which task the groundtruth generated for')
    parser.add_argument('output', type=str, help='Output path')
    args = parser.parse_args()

    # to load the trainning set without low memory mode for faster processing, you need sufficient large RAM
    dataset = VidVRD('../vidvrd-dataset', '../vidvrd-dataset/videos', ['train', 'test'])
    index = dataset.get_index(args.split)

    gts = dict()
    for ind in index:
        if args.task == 'object':
            gt = dataset.get_object_insts(ind)
        elif args.task == 'relation':
            gt = dataset.get_relation_insts(ind)
        gts[ind] = gt

    with open(args.output, 'w') as fout:
        json.dump(gts, fout, separators=(',', ':'))


class GeneralDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, set, transforms=None, _vis_threshold=0.2):

        # load all image files, sorting them to
        # ensure that they are aligned
        self._img_paths = []
        self._bbs_info = []

        self.root = data_path
        assert os.path.exists(self.root), "Your -->  --data_path <-- got problem"

    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):
        return None

    def __str__(self):
        return 'This is General Detection LOADER'

class VideoVRDDataset(GeneralDataset):

    def __init__(self, data_path,
                 set='train',
                 frames_per_segment: int = 5,
                 transforms=None,
                 _vis_threshold=0.2):

        super(GeneralDataset, self).__init__()

        self._frame_per_segment = frames_per_segment

        self.video_names = []       # Name [name1, name2 ... name_n]
        self._videos_frames = []    # Video frames path [ [video1 frames1]... [vidoo frames n] ]
        self._bbs_info = []         # Bounding Boxes [ [xmin ymin xmax ymax] ... [] ]
        self._classes = []          # Classes [ [ cls1_f1.. cls_n_f1], [cls2...]....]
        self._classes_info = []     # A dictionary that maps number to class
        self._relation_instace = [] # Relationship instances at each frame [ [], [], [], ... [] ]

        self._heights, self._widths = [], []

        self.ImglistToTensor = ImglistToTensor()
        self.transforms = transforms

        # # # # # # # # # # #
        # Start processing# #
        # # # # # # # # # # #
        self.root = data_path
        assert os.path.exists(self.root), "Your -->  --data_path <-- got problem"
        path_to_json = os.path.join(self.root, set, '*.json')

        self.all_json_anno = list(sorted(glob.glob(path_to_json)))
        assert len(self.all_json_anno) != 0

        for idx, js in enumerate(self.all_json_anno):
            with open(js, 'r') as l:
                info = l.read()
            info = json.loads(info)

            video_id = info['video_id']
            video_frame_path = glob.glob(os.path.join(self.root, set, video_id, '*.jpeg'))
            assert len(video_frame_path) == info['frame_count'], 'Frame Count and actual count mismatch '

            self._heights.append(info['height'])
            self._widths.append(info['width'])

            # Only keep the frames with Boxes
            tem_vfp = [video_frame_path[i] for i in range(len(info['trajectories']))]
            video_frame_path = tem_vfp

            assert len(info["trajectories"]) == len(video_frame_path), "Number of trajectories should equal to video frame path"

            this_objid = {so["tid"]:so["category"] for so in info["subject/objects"]}
            _bb, _cls = [], []
            _rel = [[] for _ in range(len(info['trajectories']))]

            for idx, traj in enumerate(info["trajectories"]):
                if np.equal(len(traj), 0):
                    _bb.append({-1: [-1, -1, -1, -1]})
                    _cls.append([-1])
                else:
                    _bbsub, _clssub = {}, []
                    for t in traj:
                        _bbsub[t["tid"]] = [t["bbox"]["xmin"], t["bbox"]["ymin"], t["bbox"]["xmax"], t["bbox"]["ymax"]]
                        _clssub.append(t["tid"])

                    _bb.append(_bbsub)
                    _cls.append(_clssub)

                for r in info["relation_instances"]:
                    if r['begin_fid'] <= idx <= r['end_fid']:
                        _rel[idx].append([r['subject_tid'], r['predicate'], r['object_tid']])

            assert len(_bb)==len(video_frame_path), "Each frame should has at least a bounding box"

            self.video_names.append(video_id)
            self._videos_frames.append(video_frame_path)
            self._bbs_info.append(_bb)
            self._classes.append(_cls)
            self._classes_info.append(this_objid)
            self._relation_instace.append(_rel)

        assert len(self.video_names) == len(self._videos_frames) == len(self._bbs_info) == len(self._classes) == len(self._classes_info)
        assert len(self.video_names) == len(self._heights) == len(self._widths)

        self._vis_threshold = _vis_threshold
        self.transforms = transforms

    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):
        blob = self._get(idx)

        '''
        # Should return a list of images
        video = blob['record']
        frames = self.gives_stack_of_frames(video)
        blob["video_frames"] = None
        if self.transforms:
            pass
        '''
        return blob

    def _get(self, idx):

        record = self._videos_frames[idx]
        bbox = self._bbs_info[idx]
        cls = self._classes[idx]
        clsinfo = self._classes_info[idx]
        relins = self._relation_instace[idx]
        height = self._heights[idx]
        width = self._widths[idx]

        blob = {"record": record,
                "bbox": bbox,
                "cls": cls,
                "clsinfo": clsinfo,
                "relins": relins,
                "height": height,
                "width": width}

        return blob

    def _load_frames(self, frames):
        if len(frames) == 1:  # Special Case for a single frame
            return [Image.open(frames[0])]
        return [Image.open(f) for f in frames]

    def gives_stack_of_frames(self, frames):
        frames = self._load_frames(frames)
        frames = self.ImglistToTensor(frames)
        return frames

    def visualise(self, index, draw_box=True, draw_relation=True):
        print(f'Visualising video {index}')
        blob = self._get(index)

        video = blob['record']
        bbox = blob['bbox']
        cls, cls_info = blob['cls'], blob['clsinfo']
        # Assign Colour for class
        cls_colour = {k: np.random.choice(list(_COLOR_NAME_TO_RGB.keys())) for k in cls_info.keys()}

        relation = blob['relins']
        # Relation Colour
        relation_colours = defaultdict(str)

        # Resizing the frame first
        height, width = blob['height'], blob['width']

        boundary = 0
        for i, (frame, bbox, cl, rel) in enumerate(zip(video, bbox, cls, relation)):

            # PUT Torch tensor back to numpy array
            frame = self.gives_stack_of_frames([frame])[0]
            frame = frame.numpy().transpose(1, 2, 0)
            frame = frame[:, :, ::-1]

            # Resize the video
            ratio = 720.0 / height
            size = int(round(width * ratio)) + 2 * boundary, int(round(height * ratio)) + 2 * boundary
            frame = cv2.resize(frame, (size[0]-2*boundary, size[1]-2*boundary))

            if draw_box:
                for c, b in bbox.items():
                    if b != [-1, -1, -1, -1] and c != -1:
                        frame = add_bbox(frame,
                                         int(round(b[0]* ratio)),
                                         int(round(b[1]* ratio)),
                                         int(round(b[2]* ratio)),
                                         int(round(b[3]* ratio)),
                                         color=str(cls_colour[c]))

            if draw_relation:
                if -1 not in bbox:
                    for r in rel:

                        this_colour = None
                        if r[1] in relation_colours:
                            this_colour = relation_colours[r[1]]
                        else:
                            relation_colours[r[1]]= generate_random_colour()
                            this_colour = relation_colours[r[1]]

                        try:
                            sub_cord = bbox[r[0]]  # subject coordinate
                            sub_center_x, sub_center_y = int(round((sub_cord[0] + sub_cord[2])/2)*ratio), int(round((sub_cord[1] + sub_cord[3])/2)*ratio)
                            sub_name = cls_info[r[0]]
                        except KeyError:
                            print(f'Subject ID has no detection at Video {index} frame {i} ')
                            continue
                        try:
                            obj_cord = bbox[r[2]]  # object coordinates
                            obj_center_x, obj_center_y = int(round((obj_cord[0] + obj_cord[2])/2)*ratio), int(round((obj_cord[1] + obj_cord[3])/2)*ratio)
                            obj_name = cls_info[r[2]]
                        except:
                            print(f'Object ID has no detection at Video {index} frame {i}')
                            continue

                        predicate = r[1]
                        subject_centre = (sub_center_x, sub_center_y)
                        object_centre = (obj_center_x, obj_center_y)
                        frame = add_straight_line(frame, subject_centre, object_centre, sub_name, obj_name, predicate, this_colour)

            cv2.imshow('Color image', frame)
            cv2.waitKey(2)

        cv2.destroyAllWindows()
        return None

    def get_segment_size(self):
        return self._frame_per_segment


#############################
# This is virtuall the same class as VidVrd, except that it returns one image for training

class ObjectDetectVidVRDDataset(VideoVRDDataset):

    def __init__(self, data_path,
                 set='train',
                 frames_per_segment: int = 5,
                 transforms=None,
                 _vis_threshold=0.2):

        super(GeneralDataset, self).__init__()

        self._frame_per_segment = frames_per_segment

        self.isTrain = True if set.lower() == 'train' else False
        self.video_names = []  # Name [name1, name2 ... name_n]
        self._videos_frames = []  # Video frames path [ [video1 frames1]... [vidoo frames n] ]

        self._bbs_info = []  # Bounding Boxes [ [xmin ymin xmax ymax] ... [] ]
        self._classes = []  # Classes [ [ cls1_f1.. cls_n_f1], [cls2...]....]
        self._classes_info = []  # A dictionary that maps number to class

        self._heights, self._widths = [], []

        self.ImglistToTensor = ImglistToTensor()
        self.transforms = transforms

        # # # # # # # # # # #
        # Start processing# #
        # # # # # # # # # # #
        self.root = data_path
        assert os.path.exists(self.root), "Your -->  --data_path <-- got problem"
        path_to_json = os.path.join(self.root, set, '*.json')

        self.all_json_anno = list(sorted(glob.glob(path_to_json)))
        assert len(self.all_json_anno) != 0

        for idx, js in enumerate(self.all_json_anno):
            with open(js, 'r') as l:
                info = l.read()
            info = json.loads(info)

            video_id = info['video_id']
            video_frame_path = glob.glob(os.path.join(self.root, set, video_id, '*.jpeg'))
            assert len(video_frame_path) == info['frame_count'], 'Frame Count and actual count mismatch '

            self._heights.append(info['height'])
            self._widths.append(info['width'])

            this_objid = {so["tid"]: so["category"] for so in info["subject/objects"]}
            temp_vfp = []

            _bb, _cls = [], []
            for idx, traj in enumerate(info["trajectories"]):
                if not np.equal(len(traj), 0):
                    temp_vfp.append(video_frame_path[idx])
                    _sub_bb, _sub_cls = [], []
                    for t in traj:
                        _sub_bb.append([t["bbox"]["xmin"],
                                    t["bbox"]["ymin"],
                                    t["bbox"]["xmax"],
                                    t["bbox"]["ymax"]])
                        _sub_cls.append(this_objid[t["tid"]])
                    _bb.append(_sub_bb)
                    _cls.append(_sub_cls)

            self.video_names.append(video_id)
            video_frame_path = temp_vfp

            assert len(video_frame_path)== len(_bb)==len(_cls)
            self._videos_frames += video_frame_path
            self._bbs_info += _bb
            self._classes += _cls
            self._classes_info.append(this_objid)

        # Count the number of the class
        self.all_classes_in_single = self.reassign_class() #{class_name: number}
        self.class_id_to_name = {v: k for k, v in self.all_classes_in_single.items()}

        self._vis_threshold = _vis_threshold
        self.transforms = transforms
        assert len(self._videos_frames)==len(self._bbs_info)==len(self._classes)

        print(f'Total number of classes: {self.get_num_classes()}')

    def reassign_class(self):
        print('Counting number of classes...')
        total_class = []
        for c in self._classes:
            c = list(set(c))
            total_class += c
        total_class = sorted(list(set(total_class)))
        total_class_with_num = {class_name: index+1 for index, class_name in enumerate(total_class)}
        print(f'Total number of class : {len(total_class)}, {total_class_with_num}')
        return total_class_with_num

    def get_num_classes(self):
        return len(self.all_classes_in_single)

    def __len__(self):
        return len(self._videos_frames)

    def __getitem__(self, idx):

        img_path = self._videos_frames[idx]
        img = self._load_frames([img_path])[0]
        bb = torch.as_tensor(self._bbs_info[idx], dtype=torch.float32)
        labels = torch.as_tensor([self.all_classes_in_single[c] for c in self._classes[idx]], dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (bb[:, 3] - bb[:, 1]) * (bb[:, 2] - bb[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = bb
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        temptrans = []
        temptrans.append(T.ToTensor())
        if self.transforms is not None:
            if self.isTrain:
                for t in self.transforms:
                    temptrans.append(t)

        transfunc = T.Compose(temptrans)
        img, target = transfunc(img, target)
        return img, target

    def visualise(self, index, draw_box=True):
        print(f'Visualising video {index}')
        img, target = self.__getitem__(index)
        #img = img.cpu().detach().numpy()
        bb, labels = target['boxes'], target['labels']
        bb, labels = list(bb.cpu().detach().numpy()), list(labels.cpu().detach().numpy())

        # PUT Torch tensor back to numpy array
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = img[:, :, ::-1]
        img = cv2.UMat(img).get()

        for c, b in zip(labels, bb):
            label_name = self.class_id_to_name[c]
            img = add_bbox(img,
                           int(round(b[0])),
                           int(round(b[1])),
                           int(round(b[2])),
                           int(round(b[3])),
                         color='orange',
                         label=label_name)

        cv2.imshow('Video', img)
        cv2.waitKey(10)