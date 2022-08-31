import numpy as np
import torch, torchvision

from model.helper.parser import DolphinParser
from model.helper.dolphin_detector_train import get_transform
from model.helper.utility import git_root, cpu_or_gpu, plot_traj
from dataset.dolphin import DOLPHIN, DOLPHINVIDEOVRD

import os, yaml
from tqdm import tqdm

from model.tracker.tracktor.network import FRCNN_FPN
from model.tracker.tracktor.tracktor import Tracker

from model.motiondetect.s3d_resnet import s3d_resnet

import cv2 
def inference():
    pass

def eval():
    pass

def train(dataset, tracker, motiondetect):
    # Run Detection
    train_vrd, test_vrd = dataset

    train_vrd_dataloader = torch.utils.data.DataLoader(train_vrd, batch_size=1, shuffle=False)
    test_vrd_dataloader = torch.utils.data.DataLoader(test_vrd, batch_size=1, shuffle=False)

    print(len(train_vrd))
    print(f'Length of VRD Training Sequence: {len(train_vrd_dataloader)}')
    print(f'Length of VRD Testomg Sequence: {len(train_vrd_dataloader)}')

    window_size = train_vrd.__get_window_size__()
    for i, (clip, motion) in enumerate(train_vrd_dataloader):
        print(f'Start Index {i} - {i+window_size}')

        if i > 1:
            print('Finished')
            break
        
        # Perform Tracking
        tracker.reset()        
        for ws, blob in enumerate(tqdm(clip)):    
            with torch.no_grad():
                tracker.step(blob, idx=ws+1)
        traj = tracker.get_results()

        #motiondetect
        '''
        # Module 2 - Pair Prosoal from Graphs
        visualise = plot_traj(clip, traj, motion)
        for i, pic in enumerate(visualise):
            cv2.imwrite(str(i).zfill(6)+'.jpg', pic)
        '''

def main():
    dp = DolphinParser()
    dp_args = dp.parse()

    DEVICE = cpu_or_gpu(dp_args.device)

    '''
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cudnn.deterministic = True
    '''

    print("+Initializing object detector+")

    try:
        obj_detect = FRCNN_FPN(num_classes=3)
        model_weight = os.path.join(git_root(), 'model', 'param', 'general_detector_30.pth') #'model', 'param', 'general_detector_0.pth')
        # .pth file needed

        checkpoint = torch.load(model_weight, map_location=DEVICE)
        obj_detect.load_state_dict(checkpoint['model_state_dict'])

    except (FileNotFoundError, Exception):
        print('Failed Loading Default Object Detector, Use torchvision instead')
        obj_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    obj_detect.to(DEVICE)
    obj_detect.eval()

    print("+Initializing Tracker")
    tracker = None
    with open(os.path.join(git_root(), 'model', 'tracker', 'tracktor', 'configuration.yaml'), 'r') as stream:
        try:
            configyaml = yaml.safe_load(stream)['tracktor']['tracker']
            tracker = Tracker(obj_detect, None, configyaml, DEVICE)
        except yaml.YAMLError as exc:
            print(exc)

    print("+Initializing Detection Motion+")
    motion_model = s3d_resnet().to(DEVICE)

    print('+Create Data Loader')
    try:
        dataset = DOLPHIN(data_path=dp_args.data_path,
                          set='Train',
                          mode='specific',
                          transforms=get_transform(train=True))

        dataset_test = DOLPHIN(data_path=dp_args.data_path,
                               set='Test',
                               mode='specific',
                               transforms=get_transform(train=False))

        train_vrd = DOLPHINVIDEOVRD(data_path=dp_args.data_path,
                                    set='Train',
                                    mode='specific',
                                    transforms=get_transform(train=False))

        test_vrd = DOLPHINVIDEOVRD(data_path=dp_args.data_path,
                                    set='Train',
                                    mode='specific',
                                    transforms=get_transform(train=False))


    except AssertionError as e:
        print('You do not have dataset in the folder')
        save_path = os.path.join(git_root(), 'dataset', 'dolphin')


        dataset = DOLPHIN(data_path=dp_args.data_path,
                          set='Train',
                          mode='specific',
                          transforms=get_transform(train=True))

        dataset_test = DOLPHIN(data_path=dp_args.data_path,
                               set='Test',
                               mode='specific',
                               transforms=get_transform(train=False))

        train_vrd = DOLPHINVIDEOVRD(data_path=dp_args.data_path,
                                    set='Train',
                                    mode='specific',
                                    transforms=get_transform(train=False))

        test_vrd = DOLPHINVIDEOVRD(data_path=dp_args.data_path,
                                    set='Train',
                                    mode='specific',
                                    transforms=get_transform(train=False))


    train((train_vrd, test_vrd), tracker, motion_model)



if __name__ == '__main__':
    main()