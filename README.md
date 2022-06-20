# setup
~~~
git clone https://github.com/MoeMatsuki/mmpose_test
cd mmpose_test
bash setup.sh
~~~

# テスト
~~~
python inference_2d_pose.py
~~~

# データを変更
conf.pyの中身を変更させてください。
~~~
import os.path as osp

pose_config = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = 'mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

dirs = "mmpose/tests/data/coco/"　# ←ここを変更

log_dir_name = "log"　# ←logの名前も変更したほうがいい
img_log = osp.join(log_dir_name, "img")
json_log = osp.join(log_dir_name, "json")
~~~

# 環境で参考にする
https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md
