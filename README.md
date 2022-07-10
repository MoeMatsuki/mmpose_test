# clone repository
~~~
git clone -b on_cpu https://github.com/MoeMatsuki/mmpose_test
cd mmpose_test
git clone https://github.com/open-mmlab/mmpose.git
~~~

# dockerを使う
~~~
docker build -t mmpose env/
docker run -it -v $PWD:/opt/workspace --name MMPOSE mmpose /bin/bash
~~~


# setup
~~~
bash setup.sh
~~~

# アノテーション情報の準備
[labelme](https://github.com/wkentaro/labelme)というツールを使って、人のbboxをアノテーションする。
画像ディレクトリとアノテーションファイルのpathは、"demo.sh"内の以下のコードの”--img-root”と"--json-dir"で指定。
~~~
python demo/top_down_img_demo_with_labelme.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    ../model/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../test_data/ --json-dir ../test_data/labelme \
    --out-img-root ../logs --device=cpu
~~~

~~~
test_data/
├── 000000107339.jpg
├── 000000170099.jpg
└── labelme
    ├── 000000107339.json
    └── 000000170099.json
~~~

# テスト
~~~
bash demo.sh
~~~
