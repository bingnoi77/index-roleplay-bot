# 1.获取预训练模型
#tar -zcvf pretrained_models.tar.gz --exclude=.gitignore *
cd GPT-SoVITS

# 获取当前激活环境的 Python 路径
PYTHON_PATH=$(which python)

# 使用当前激活环境的 Python 路径运行脚本
$PYTHON_PATH GPT_SoVITS/inference_webui2.py