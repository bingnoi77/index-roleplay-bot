# 1.获取预训练模型
#tar -zcvf pretrained_models.tar.gz --exclude=.gitignore *
if [ ! -d "./GPT_SoVITS/pretrained_models/chinese-hubert-base" ]; then
    wget http://jssz-inner-boss.bilibili.co/fe-api/aip-fe-upload/dynamic-comic/comic-tts-service/pretrained_models.tar.gz
    tar -zxvf pretrained_models.tar.gz -C GPT_SoVITS/pretrained_models
    rm -f pretrained_models.tar.gz
fi

# 2.获取盲夫ip相关的模型和wav文件
if [ ! -d "./weights" ]; then
    wget http://jssz-inner-boss.bilibili.co/fe-api/aip-fe-upload/dynamic-comic/comic-tts-service/blind_husband/weights.tar.gz
    tar -zxvf weights.tar.gz
    rm -f weights.tar.gz
fi

if [ ! -d "./dongtaiman-wavs/ref" ]; then
    wget http://jssz-inner-boss.bilibili.co/fe-api/aip-fe-upload/dynamic-comic/comic-tts-service/blind_husband/wavs.tar.gz
    mkdir -p dongtaiman-wavs
    tar -zxvf wavs.tar.gz -C dongtaiman-wavs
    rm -f wavs.tar.gz
fi

/root/miniconda3/envs/py39webui/bin/python GPT_SoVITS/inference_webui2.py