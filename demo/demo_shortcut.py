# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import warnings

import time
import cv2
import mmcv
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/pyskl/ckpt/'
                 'posec3d/slowonly_r50_ntu120_xsub/joint.pth'),
        help='skeleton action recognition checkpoint file/url')
    # parser.add_argument(
    #     '--det-config',
    #     default='demo/faster_rcnn_r50_fpn_2x_coco.py',
    #     help='human detection config file path (from mmdet)')
    # parser.add_argument(
    #     '--det-checkpoint',
    #     default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
    #              'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
    #              'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
    #     help='human detection checkpoint file/url')
    # parser.add_argument(
    #     '--pose-config',
    #     default='demo/hrnet_w32_coco_256x192.py',
    #     help='human pose estimation config file path (from mmpose)')
    # parser.add_argument(
    #     '--pose-checkpoint',
    #     default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
    #              'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
    #     help='human pose estimation checkpoint file/url')
    # parser.add_argument(
    #     '--det-score-thr',
    #     type=float,
    #     default=0.9,
    #     help='the threshold of human detection score')
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
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # frame_paths, original_frames = frame_extraction(args.video,
    #                                                 args.short_side)
    # num_frame = len(frame_paths)
    # h, w, _ = original_frames[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose'] # list of dict

    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    # det_results = detection_inference(args, frame_paths)
    # torch.cuda.empty_cache()

    # pose_results = pose_inference(args, frame_paths, det_results)
    # torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    # if GCN_flag:
    #     # We will keep at most `GCN_nperson` persons per frame.
    #     tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
    #     keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
    #     fake_anno['keypoint'] = keypoint
    #     fake_anno['keypoint_score'] = keypoint_score
    # else:
    # num_person = max([len(x) for x in pose_results])
    num_person = 1
    # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                dtype=np.float16)
    # for i, poses in enumerate(pose_results):
    #     for j, pose in enumerate(poses):
    #         pose = pose['keypoints']
    #         keypoint[j, i] = pose[:, :2]
    #         keypoint_score[j, i] = pose[:, 2]

    # fill out the numpy array keypoint and keypoint_score

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    print('\n', fake_anno['keypoint'].shape)
    print(fake_anno['keypoint_score'][0][0], '\n')
    print(fake_anno['keypoint'][0][0], '\n')

    t0 = time.time()
    results = inference_recognizer(model, fake_anno)
    print('cost: ', time.time()-t0, ' secs')


    action_label = label_map[results[0][0]]
    # print(label_map)

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    for frame in vis_frames:
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
