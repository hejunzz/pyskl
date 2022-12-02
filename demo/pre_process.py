# %%
import json
import pickle
import numpy as np
import os
import argparse
import shutil
import warnings
import time
import cv2
import mmcv
import torch
import videoreader as decord


from os.path import expanduser
from pyskl.utils.visualize import Vis2DPose
from scipy.optimize import linear_sum_assignment
from pyskl.apis import inference_recognizer, init_recognizer


# try:
#     from mmpose.apis import vis_pose_result
# except (ImportError, ModuleNotFoundError):
#     def vis_pose_result(*args, **kwargs):
#         pass

#     warnings.warn(
#         'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
#         '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
#     )
# %%
def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/pyskl/ckpt/'
                 'posec3d/slowonly_r50_ntu120_xsub/joint.pth'),
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args

# %%
'''
This pre-processing script is used to generate <video, keypoints> pair from the json file of annotation
'''
home = expanduser("~")
ann_json = os.path.join(home, 'Documents/data/demo/simple_ann.json')
kp_path = os.path.join(home, 'Documents/data/demo/pkl')
vid_path = os.path.join(home, 'Documents/data/demo/video')
temp_path = os.path.join(home, 'Documents/data/demo/temp')
config_file = os.path.join(home, 'Documents/data/demo/configs/posec3d/slowonly_r50_ntu120_xsub/joint.py')
label_file = os.path.join(home, 'Documents/data/demo/configs/label_map/nturgbd_120.txt')
ds_pickle_file = os.path.join(home, 'Documents/data/demo/simple_data.pkl')

args = parse_args()
print(args)

args.config = config_file
args.label_map = label_file


kp_files = os.listdir(kp_path)
vid_files = os.listdir(vid_path)

# joints_17_map = {0:16, 1:14, 2:12, 3:11, 4:13, 5:15, 10:10,
#                 11:8, 12:6, 13:5, 14:7, 15:9, 16:0, 17:2, 18:4, 19:1, 20:3}

#%%
with open(ann_json, 'r') as json_file:
    ann_list = json.load(json_file)

# %%
# label_dict

temporal_pose_list = []
# if True:
    # ann = ann_list[0]
for ann in ann_list:    
    base_name = os.path.basename(ann['video_url']).split('.')[0]
    vid_name = [vid for vid in vid_files if base_name in vid][0]
    kp_name = [kp for kp in kp_files if base_name in kp][0]
    print(vid_name, kp_name)

    with open(os.path.join(kp_path, kp_name), 'rb') as kp:
        kp_pickle = pickle.load(kp)
    
    vid = decord.VideoReader(os.path.join(vid_path, vid_name))

    # collect kp_pickle into numpy 
    fps = vid.frame_rate # int(kp_json['file_info']['frame_rate'])
    img_size = vid.frame_shape[:2] # kp_json['file_info']['dims']

    for clip in ann['tricks']:
        print(clip['labels'][0])  # str
        start = int(fps * clip['start'])   # second, float
        end = int(fps * clip['end'])     # second, float

        clip_kp = kp_pickle[start:end]
        clip_numpy = np.zeros((1, end-start, 17, 2), dtype=np.float16)
        clip_score_numpy = np.zeros((1, end-start, 17), dtype=np.float16)
        for i in range(end-start):
            if clip_kp[i]:
                joints = clip_kp[i][0]['keypoints'][:,:2]# np.array(clip_json[i][0]['pose2d']['joints']).reshape(-1,2)
                score = clip_kp[i][0]['keypoints'][:,-1]# np.array(clip_json[i][0]['pose2d']['scores'])
                clip_numpy[0, i, :, :] = joints[:17, :]
                clip_score_numpy[0, i, :] = score[:17]
        temporal_pose = {'keypoint': clip_numpy, 'keypoint_score': clip_score_numpy, 
                         'labels': clip['labels'][0], 'img_shape':img_size}
        temporal_pose_list.append(temporal_pose)

    print(len(temporal_pose_list))

# %%
# test the recognition pipeline
config = mmcv.Config.fromfile(args.config)

config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose'] # list of dict

model = init_recognizer(config, args.checkpoint, args.device)

# Load label_map
label_map = [x.strip() for x in open(args.label_map).readlines()]

# %%
# Visualize a sequence of poses and run the test pipeline of PoseC3D
temporal_pose = temporal_pose_list[22]
print(temporal_pose['labels'])
vid = Vis2DPose(temporal_pose, thre=0.2, out_shape=(img_size[0]//8, img_size[1]//8), layout='coco', fps=25, video=None) # (540, 960)
vid.ipython_display()
# vid.write_videofile(os.path.join(temp_path, 'clip_demo.mp4'), remove_temp=True)


#%%

fake_anno = dict(
    frame_dir='',
    label=-1,
    img_shape=temporal_pose['img_shape'], #(h, w),
    original_shape=temporal_pose['img_shape'], #(h, w),
    start_index=0,
    modality='Pose',
    total_frames=temporal_pose['keypoint'].shape[1])

fake_anno['keypoint'] = temporal_pose['keypoint']
fake_anno['keypoint_score'] = temporal_pose['keypoint_score']

t0 = time.time()
results = inference_recognizer(model, fake_anno)
print('cost: ', time.time()-t0, ' secs')


for i in range(len(results)):
    action_label = label_map[results[i][0]]
    print(i, action_label, results[i][1])

# %%
annotations = []
label2id = dict()   # fill out the dict {"label_0": 0, "label_1": 1, ...}
max_id = 0
train_split = []
val_split = []

for i, poses in enumerate(temporal_pose_list):
    if poses['labels'] not in label2id:
        label2id[poses['labels']] = max_id
        max_id += 1
    frame_dir = 'clip-%d'%(i)
    anno = dict(
        frame_dir=frame_dir, # used for matching the raw frames of these poses
        label=label2id[poses['labels']],   # 
        img_shape=poses['img_shape'], #(h, w),
        original_shape=poses['img_shape'], #(h, w),
        start_index=0,
        modality='Pose',
        total_frames=poses['keypoint'].shape[1])

    anno['keypoint'] = poses['keypoint']
    anno['keypoint_score'] = poses['keypoint_score']

    annotations.append(anno)
    train_split.append(frame_dir)

pkl = dict(split=dict(train=train_split, val=train_split), annotations=annotations)

# dump into a pickle file by mmcv.dump
mmcv.dump(pkl, ds_pickle_file)
