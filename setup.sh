# install dependencies: (use cu111 because colab has CUDA 11.1)
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

# install mmdet for inference demo
pip install mmdet

# clone mmpose repo
rm -rf mmpose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose

# install mmpose dependencies
pip install -r requirements.txt

# install mmpose in develop mode
pip install -e .

cd ..