# 1.获取预训练模型
#tar -zcvf pretrained_models.tar.gz --exclude=.gitignore *
cd GPT-SoVITS
if [ ! -d ".GPT_SoVITS/pretrained_models/chinese-hubert-base" ]; then
    wget http://jssz-inner-boss.bilibili.co/fe-api/aip-fe-upload/dynamic-comic/comic-tts-service/pretrained_models.tar.gz
    mkdir -p GPT_SoVITS/pretrained_models
    tar -zxvf pretrained_models.tar.gz -C GPT_SoVITS/pretrained_models
    rm -f pretrained_models.tar.gz
fi

# 获取当前激活环境的 Python 路径
PYTHON_PATH=$(which python)

# 使用当前激活环境的 Python 路径运行脚本
$PYTHON_PATH GPT_SoVITS/inference_webui2.py

cd ..

$PYTHON_PATH hf_based_demo.py