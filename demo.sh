cp top_down_img_demo_with_labelme.py mmpose/demo/

mkdir models
cd model
# wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth

cd ../mmpose

python demo/top_down_img_demo_with_labelme.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    ../model/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../data/shinagawa_office/10F --json-dir ../data/shinagawa_office/10F/anno \
    --out-img-root ../logs --device=cpu

