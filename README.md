# Counting-Demo
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
计数结果及中间过程将被保存在{视频名}.log文件中