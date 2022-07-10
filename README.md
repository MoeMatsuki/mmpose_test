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

# テスト
~~~
bash demo.sh
~~~
