
## 环境

环境安装参考https://github.com/RVC-Boss/GPT-SoVITS

首先需要一个py39，可以用conda

conda create -n GPTSoVits python=3.9

conda activate GPTSoVits

pip包：

pip install -r requirements.txt

ffmpeg系统命令安装：

conda install ffmpeg或者apt install ffmpeg

## 推理

bash mygo.sh启动server，首次启动会自动下载大模型

python client.py使用api进行推理，返回本地url，具体看脚本
