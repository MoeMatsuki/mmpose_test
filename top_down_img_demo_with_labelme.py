# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from xtcocotools.coco import COCO
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

def diff_max_min(co_list):
    return max(co_list), min(co_list)

def convert_json_bbox(dir_path):
    try:
        with open(dir_path, "r") as j:
            labelme = json.load(j)
        bboxes = labelme["shapes"]
    except:
        return []

    result = []
    for bbox in bboxes:
        co = bbox["points"]
        max_x, min_x = diff_max_min([c[0] for c in co])
        max_y, min_y = diff_max_min([c[1] for c in co])
        result.append({"bbox": np.array([min_x, min_y, max_x, max_y], dtype = 'float32')})
    return result



def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-dir',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    file_names = [f for f in os.listdir(args.img_root) if ".JPG" in f]

    # process each image
    for i in range(len(file_names)):  
        image_name = os.path.join(args.img_root, file_names[i])
        json_file = file_names[i].split(".")[0] + ".json"
        person_results = convert_json_bbox(os.path.join(args.json_dir, json_file))

        # # make person bounding boxes
        # person_results = []
        # for ann_id in ann_ids:
        #     person = {}
        #     ann = coco.anns[ann_id]
        #     # bbox format is 'xywh'
        #     person['bbox'] = ann['bbox']
        #     person_results.append(person)
        # print(person_results)

        if person_results:
            # test a single image, with a list of bboxes
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                image_name,
                person_results,
                format='xyxy',
                dataset=dataset)

            if args.out_img_root == '':
                out_file = None
            else:
                os.makedirs(args.out_img_root, exist_ok=True)
                out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

            vis_result = vis_pose_result(
                pose_model,
                image_name,
                pose_results,
                dataset=dataset,
                show=False)

            cv2.imwrite(out_file, vis_result)


if __name__ == '__main__':
    main()
