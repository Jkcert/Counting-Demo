import shutil
from PIL import Image, ImageSequence
import os
import argparse
from logger import Logger
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import alpha_model
# 加载模型
demo = alpha_model.SingleImageAlphaPose()


def skeleton_extraction(data_type="--image_dir", path="./openpose/media/", task=0):
    # test_path
    dt = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    # log = Logger(dt + '_extract.log', level='error')
    # Process Image
    imagePath = path
    image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    # log.logger.info(imagePath + ' is being extracted...')
    tick = time.time()
    pose = demo.process(imagePath, image)
    tick = (time.time() - tick) * 1000
    # log.logger.info(imagePath + ' has been extracted,took ' + str(tick) + 'ms.')
    save_prefix = "./count_cache/" if task == 1 else "./generate_cache/"
    # Save Image
    f_num = path.split('/')[-1]
    cv2.imwrite(save_prefix + f"output_images/{f_num}", demo.vis(demo.getImg(), pose))
    # change output format
    all_result = [pose]
    skt_list = []
    boxes = []
    for im_res in all_result:
        # TODO: 修改为支持多人
        # for human in [im_res['result'][0]]
        if not im_res:
            return None, None
        for human in im_res['result']:
            boxes.append(human['bbox'])
            # if float(human['proposal_score']) < 2.6:
            #     continue
            keypoints = []
            human_list = []
            result = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)
            # if 'box' in human.keys():
            #     result['box'] = human['box']
            # # pose track results by PoseFlow
            # if 'idx' in human.keys():
            #     result['idx'] = human['idx']
            result['keypoints'].append((result['keypoints'][15] + result['keypoints'][18]) / 2)
            result['keypoints'].append((result['keypoints'][16] + result['keypoints'][19]) / 2)
            result['keypoints'].append((result['keypoints'][17] + result['keypoints'][20]) / 2)
            indexarr = [0, 51, 18, 24, 30, 15, 21, 27, 36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
            for i in indexarr:
                human_list.append([result['keypoints'][i], result['keypoints'][i + 1], result['keypoints'][i + 2]])
            skt_list.append(human_list)
    return boxes, skt_list


def load_skeletons(task):
    res = []
    frame = []
    skt = []
    save_prefix = "./count_cache/" if task == 1 else "./generate_cache/"
    for root, dirs, files in os.walk(save_prefix + "output_jsons"):
        f_sum = len(files)
        for f in range(f_sum):
            frame.append(f)
            temp = []
            f = open(root + '/' + str(f) + '.txt')
            for line in f.readlines():
                cos = line.split(" ")
                if len(cos) == 4:
                    skt.append([float(cos[0]), float(cos[1]), float(cos[2])])
                else:
                    temp.append(skt)
                    skt = []
            frame.append(temp)
            f.close()
            res.append(frame)
            frame = []
    return res


# mat_plot poly_fitting
def simple_fitting(points, degree):
    if len(points) > 0:
        x = []
        y = []
        for index, item in enumerate(points):
            x.append(item[0])
            y.append(item[1])
        x = np.array(x)
        y = np.array(y)
        f1 = np.polyfit(x, y, degree)
        p1 = np.poly1d(f1)
        print('p1 is :\n', p1)
        y_val = p1(x)  # 拟合y值
        print('y_val is :\n', y_val)
        # 绘图
        plot1 = plt.plot(x, y, 's', label='original values')
        plot2 = plt.plot(x, y_val, 'r', label='poly_fit values')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc=4)  # 指定legend的位置右下角
        plt.title('poly_fitting')
        plt.show()


def trapezoidal_fitting(points, degree):
    if len(points) > 0:
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        flag = 0
        for index, item in enumerate(points):
            if item[1] < 1 and flag == 0:
                x1.append(item[0])
                y1.append(item[1])
            elif item[1] == 1 and flag == 0:
                print("Inflection 1: " + str(item[0]) + ".")
                x1.append(item[0])
                y1.append(item[1])
                flag = 1
            elif item[1] == 1 and flag == 1:
                if points[index + 1][1] != 1:
                    print("Inflection 1: " + str(item[0]) + ".")
                    x2.append(item[0])
                    y2.append(item[1])
                    flag = 2
            else:
                x2.append(item[0])
                y2.append(item[1])
        if len(x2) == 0:
            simple_fitting(points, degree)
            return
        x1 = np.round(np.array(x1), 3)
        y1 = np.round(np.array(y1), 3)
        x2 = np.round(np.array(x2), 3)
        y2 = np.round(np.array(y2), 3)
        f1 = np.polyfit(x1, y1, degree)
        f2 = np.polyfit(x2, y2, degree)
        p1 = np.poly1d(f1)
        p2 = np.poly1d(f2)
        print('p1 is :\n', p1)
        print('p2 is :\n', p2)
        y_val1 = p1(x1)
        print(x1, y_val1)
        y_val2 = p2(x2)
        print(x2, y_val2)
        plot1 = plt.plot(x1, y1, 's', label='original values 1')
        plot2 = plt.plot(x1, y_val1, 'r', label='poly_fit values 1')
        plot3 = plt.plot(x2, y2, 's', label='original values 2')
        plot4 = plt.plot(x2, y_val2, 'r', label='poly_fit values 2')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc=4)  # 指定legend的位置右下角
        plt.title('trapezoidal_fitting')
        plt.show()


def frame_regularization(points):
    start = points[0][0]
    ratio = points[-1][0] - start
    for point in points:
        point[0] = (point[0] - start) / ratio
    return points


def parse_gif(gif_path):
    im = Image.open(gif_path)
    frames = ImageSequence.Iterator(im)
    file_name = gif_path.split(".")[0]
    index = 1
    pic_dir = "{0}".format(file_name)
    if os.path.isdir(pic_dir):
        shutil.rmtree(pic_dir)
    os.makedirs(pic_dir)
    for frame in frames:
        print("image %d: mode %s, size %s" % (index, frame.mode, frame.size))
        frame.save("%s/frame%d.png" % (file_name, index))
        index += 1


def parse_video(video_path, compress=True):
    video_capture = cv2.VideoCapture(video_path)
    f = 0
    file_name = video_path.split(".")[0]
    index = 1
    pic_dir = "{0}".format(file_name + '-compress')
    if os.path.isdir(pic_dir):
        shutil.rmtree(pic_dir)
    os.makedirs(pic_dir)
    while True:
        res, frame = video_capture.read()
        if res:
            # sampling every 5 frames
            # if f % 5 == 0:
            # resize to a lower 480p frame
            if compress:
                s = frame.shape
                h, w = s[0], s[1]
                while max(h, w) > 1080:
                    h = int(h / 2)
                    w = int(w / 2)
                frame = cv2.resize(frame, (w, h))
            path = pic_dir + '/' + str('%06d' % f) + '.jpg'
            cv2.imwrite(path, frame)
            f = f + 1
        else:
            video_capture.release()
            break
# parse_video('data/sit-up/11-49.mp4', True)


def parse_video_dir(video_dir, compress=False):
    print('Video ' + video_dir.split('/')[1] + ' is parsing.')
    video_paths = os.listdir(video_dir)
    for index, video_path in enumerate(video_paths):
        if "mp4" not in video_path and "mov" not in video_path and "MOV" not in video_path:
            continue
        f = 0
        file_name = video_path.split('.')[0]
        pic_dir = video_dir + "-compress/" + "{0}".format(file_name)
        print(pic_dir)
        if os.path.isdir(pic_dir):
            shutil.rmtree(pic_dir)
        os.makedirs(pic_dir)
        video_capture = cv2.VideoCapture(video_dir + "/" + video_path)
        while True:
            res, frame = video_capture.read()
            if res:
                if compress:
                    s = frame.shape
                    h, w = s[0], s[1]
                    while max(h, w) > 1080:
                        h = int(h / 2)
                        w = int(w / 2)
                    frame = cv2.resize(frame, (w, h))
                path = pic_dir + '/' + str('%06d' % f) + '.jpg'
                cv2.imwrite(path, frame)
                f = f + 1
            else:
                video_capture.release()
                break
# parse_video_dir('data/sit-up', True)
def parse_all_video():
    for dir in ['data/pull-up', 'data/sit-up', 'data/push-up']:
        parse_video_dir(dir, True)
    print('All video has been parsed.')

def insert_points(origin, new):
    res = []
    fo = 0
    fn = 0
    origin = origin[1:-1]
    while fo < len(origin) and fn < len(new):
        if origin[fo][0] < new[fn][0]:
            res.append(origin[fo])
            fo += 1
        else:
            res.append(new[fn])
            fn += 1
    for i in range(fn, len(new)):
        res.append(new[i])
    return res


def reformat_skeleton(skeleton):
    r = []
    t = []
    if len(skeleton) == 75:
        for index, item in enumerate(skeleton):
            t.append(item)
            if index % 3 == 2:
                r.append(t)
                t = []
        return r
    return -155


def mse(x, y):
    err = np.square(np.subtract(x, y)).mean()
    return err


from mapping import compute_angle
def action_recognize(action_dic, skeleton):
    action_type = None
    for key, value in action_dic.items():
        feature = value['feature']
        ratio = abs(skeleton[5][0] - skeleton[2][0]) / abs(skeleton[10][0] - skeleton[1][0] + 0.01)
        # 是否是引体向上
        if ratio > 1 and skeleton[4][1] < skeleton[3][1] < skeleton[2][1]:
            action_type = key
            break
        else:
            if key == 'pull_up':
                continue
            skt_angle, feature_angle = [], []
            for l in [[3, 2, 3, 4], [6, 5, 6, 7], [9, 8, 9, 10], [12, 11, 12, 13]]:
                skt_angle.append(compute_angle(skeleton[l[0]], skeleton[l[1]], skeleton[l[2]], skeleton[l[3]]))
                feature_angle.append(compute_angle(feature[l[0]], feature[l[1]], feature[l[2]], feature[l[3]]))
            # print(feature_angle, skt_angle)
            err = mse(np.array(feature_angle), np.array(skt_angle))
            # print(err)
            if err < 2500:
                action_type = key
                break

    return action_type


def pic_avg_diff(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    # CV difference function, optimized but slower
    err = cv2.absdiff(img1, img2)
    # err = np.maximum(img1 - img2, img2 - img1)
    diff = np.sum(err)
    # cv2.imshow('err', err)
    # print(dif f)
    # cv2.waitKey(0)
    # plt.imshow(err)
    # plt.show()
    avg_diff = diff / (err.shape[0] * err.shape[1])
    # print(avg_diff)
    return avg_diff


# pic_avg_diff("temp/pull-up/000000.jpg", "temp/pull-up/000001.jpg")

def write_video():
    import cv2

    img_root = 'data/demo6/'  # 是图片序列的位置
    fps = 30 #可以随意调整视频的帧速率

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter('data/TestVideo6.mp4', fourcc, fps, (960, 540), True)  # 最后一个是保存图片的尺寸

    for i in range(len(os.listdir(img_root))):
        frame = cv2.imread(img_root + str('%06d' % i) + '.jpg')
        frame = cv2.resize(frame, (960, 540))
        videoWriter.write(frame)
    videoWriter.release()
    #cv2.destroyAllWindows()

# write_video()