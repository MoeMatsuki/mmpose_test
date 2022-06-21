import json
from pprint import pprint
import cv2
from IPython.display import Image, display
import tempfile
import os.path as osp
import sys
import os
from conf import(
    pose_config,
    pose_checkpoint,
    det_config,
    det_checkpoint,
    dirs,
    log_dir_name,
    img_log,
    json_log,
    device
)
sys.path.append(os.path.join(os.getcwd(), "mmpose"))
sys.path.append(os.path.join(os.getcwd(), "mmpose/mmpose"))

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)


def inference(img, pose_model, det_model):
    # inference detection
    mmdet_results = inference_detector(det_model, img, device=device)

    # extract person (COCO_ID=1) bounding boxes from the detection results
    person_results = process_mmdet_results(mmdet_results, cat_id=1)
    return person_results

    # # inference pose
    # pose_results, returned_outputs = inference_top_down_pose_model(
    #     pose_model,
    #     img,
    #     person_results,
    #     bbox_thr=0.3,
    #     format='xyxy',
    #     dataset=pose_model.cfg.data.test.type)

    # # show pose estimation results
    # vis_result = vis_pose_result(
    #     pose_model,
    #     img,
    #     pose_results,
    #     dataset=pose_model.cfg.data.test.type,
    #     show=False)

    # return convert_type(pose_results), vis_result


def convert_type(pose_results):
    result_list = []
    for bbox in pose_results:
        for k, v in bbox.items():
            result_list.append({k: v.tolist()})
    return result_list


def main():
    log_dir = osp.join(os.getcwd(), log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_log, exist_ok=True)
    os.makedirs(json_log, exist_ok=True)

    files = [f for f in os.listdir(osp.join(os.getcwd(), dirs)) if ".jpg" in f]

    # initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    # initialize detector
    det_model = init_detector(det_config, det_checkpoint)
    for file_name in files:
        print(file_name + " is processing")
        pose_results = inference(
            osp.join(dirs, file_name), pose_model, det_model)
        # save image
        # cv2.imwrite(osp.join(img_log, file_name), vis_result)
        # save json
        with open(osp.join(json_log, file_name.split(".")[0] + ".json"), 'w') as f:
            json.dump(pose_results, f, indent=4)


if __name__ == "__main__":
    main()
