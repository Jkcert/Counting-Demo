# Counting-Demo
1. 安装依赖\
`conda env create -n alphapose -f environment.yaml`\
**请根据自己的显卡配置自行安装相应版本pytorch**
1. 编译AlphaPose模型环境\
`python setup.py build develop`
2. 下载数据\
https://cloud.tsinghua.edu.cn/d/e4d1f5705df442c9bda2/ \
完成后数据放置如下：\
\\data\
|--- \\pull-up\
|--- \\push-up\
|--- \\sit-up
3. 下载预训练模型\
https://cloud.tsinghua.edu.cn/f/11b428c35695431da655/ 下载fast_res50_256x192.pth放置在pretrained_models目录下\
https://cloud.tsinghua.edu.cn/f/666142613e8245b58c3c/ yolov3-spp.weights放置在detector/yolo/data目录下
4. 视频拆帧\
` python -c "import utils; utils.parse_all_video()"`
5. 运行程序\
`python main.py`\