import json
import os
import cv2 as cv
from judging import ave_error_judging1d
import utils
import shutil
from mapper import pull_up_trapezoidal, push_up_trapezoidal, sit_up_trapezoidal
from mapping import pull_up_mapping, push_up_mapping, sit_up_mapping, push_up_pose, sit_up_pose, pull_up_pose
from logger import Logger
import time
import csv
import tkinter as tk
from tkinter import *
import threading

with open('action_dic.json', 'r') as json_file:
    action_dic = json.load(json_file)
# Threshold for detecting whether the action has begun
t = 0.1
# threshold for judging whether the action is valid
T = 0.3
roi_time = 0


def counting(path):
    global roi_time
    roi_time = 0
    # dt = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    log = Logger(path + '_count.log', level='debug')
    # 生成roi文件夹
    # roi_path = path + "-roi"
    # if os.path.exists(roi_path):
    #     shutil.rmtree(roi_path)
    #     os.makedirs(roi_path)
    # else:
    #     os.makedirs(roi_path)
    # 生成中间结果文件夹
    for save_dir in ["./count_cache/output_images", "./count_cache/output_jsons"]:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)
    # 动作字典的参数
    mapper = None
    model = None
    judging = None
    method = None
    waveform = None
    # 动作类型
    action_type = None
    # 动作计数
    action_count = 0
    # roi区域
    roi = False
    # 隔freq帧提取一次
    freq_dense, freq_5, freq_3 = 4, 0, 0
    freq = freq_dense
    # 视频帧序列索引
    index = 0
    # 上一帧的图片，用来算帧间残差
    last_image_path = None
    # 目前取得帧的列表
    y_list = []
    # 目前采样的帧在模板中的位置
    waveform_index = 0
    # 本次动作是否开始
    action_start = False
    skt_idx = 0
    # 上一个动作是否匹配成功
    last_action_match = False
    # 动作是否已经上升
    up = False
    # 动作是否到达顶部
    top = False
    # roi初始参数
    x1, x2, y1, y2 = 10000, 0, 10000, 0
    # 一共识别多少帧
    frame_count = 0
    # 采样3,5,dense各占多少
    c_3, c_5, c_d = 0, 0, 0
    start = time.time()

    # 对文件夹遍历，主流程开始
    image_paths = os.listdir(path)

    while index < len(image_paths):
        image_path = path + '/' + image_paths[index]
        # 裁剪图片的roi区域
        roi_start = time.time()
        # if roi:
        #     raw_img = cv.imread(image_path)
        #     roi_img = raw_img[y1: y2, x1: x2, :]
        #     cv.imwrite(roi_path + '/' + image_paths[index], roi_img)
        #     image_path = roi_path + '/' + image_paths[index]
        #     if len(os.listdir(roi_path)) == 1:
        #         last_image_path = image_path
        roi_end = time.time()
        roi_time += (roi_end - roi_start)
        # 帧间差距超过阈值，动作开始
        if not action_start:
            if last_image_path and utils.pic_avg_diff(last_image_path, image_path) > 1:
                log.logger.info('Action start.')
                action_start = True
            else:
                last_image_path = image_path
                index += 1
                continue

        tick = time.time()
        # log.logger.info('Start skeleton extraction.')
        # 提取骨骼点，如果提取失败则跳过

        boxes, skeleton_data = utils.skeleton_extraction("--image", image_path, 1)
        frame_count += 1
        if skeleton_data is None or len(skeleton_data) == 0:
            # print("No skeleton.")
            index += 1
            continue
    # 创建模板用
    #     model = eval(action_dic['pull_up']['model'])
    #     y_list.append([index, model(skeleton_data)])
    #     index += 1
    #     y_list = test.frame_regularization(y_list)
    #     test.trapezoidal_fitting(y_list, 6)
        tick = (time.time() - tick) * 1000
        log.logger.info('Skeleton fetched, spent ' + str(tick) + 'ms.')
        # 首帧识别出动作类型
        if action_type is None:
            for idx, skt in enumerate(skeleton_data):
                action_type = utils.action_recognize(action_dic, skt)
                skt_idx = idx
                if action_type:
                    break
            # 如果没有匹配的动作，则跳到下一帧
            if action_type is None:
                log.logger.info('Not standard action, continue')
                index += 1
                continue

            log.logger.info('This action is recognized as ' + action_type)
            mapper = eval(action_dic[action_type]['mapper'])
            model = eval(action_dic[action_type]['model'])
            judging = eval(action_dic[action_type]['judging'])
            method = action_dic[action_type]['method']
            waveform = []
            last_image_path = image_path
        else:
            # 动作映射，如果失败则跳过
            y = model(skeleton_data, skt_idx)
            if y == -1:
                index += 1
                log.logger.debug("Frame " + str(index) + " pose failed.")
                # print("pose fail")
                continue
            # 生成roi
            if not roi:
                box = boxes[skt_idx]
                x1, x2, y1, y2 = min(x1, int(box[0])), max(x2, int(box[0]) + int(box[2])), min(y1, int(box[1])), max(y2, int(box[1]) + int(box[3]))

            # 用动作上升，到达顶点，结尾点判断动作是否结束
            if len(y_list) > 1 and not top and y_list[-1][1] > y_list[-2][1] + 0.1:
                up = True
            if len(y_list) > 1 and up and y_list[-1][1] + 0.1 < y_list[-2][1]:
                top = True
            if top and y_list[-1][0] - y_list[0][0] > 15 and (y_list[-1][1] < 0.1 or (y_list[-1][1] > y_list[-2][1] + 0.03 and y_list[-2][1] < 0.4)\
                    or (freq != freq_dense and len(y_list) == len(waveform))):
                # 动作结束，重置参数
                action_start = False
                top = False
                up = False
                last_image_path = image_path
                waveform_index = 0
                # 剔除可能在列表中的下个动作的帧
                if y_list[-1][1] > y_list[-2][1]:
                    index -= 2 * freq
                    y_list.pop()
                else:
                    index -= freq

                frame_range = [y_list[0][0], y_list[-1][0]]
                # 帧序号映射到[0, 1]
                y_list = utils.frame_regularization(y_list)
                log.logger.debug('Action ' + str(action_count + 1) + ' ended, judging...')
                # 判断动作是否标准
                e = judging(y_list, mapper)
                if len(y_list) < 5:
                    time.sleep(1)
                elif len(y_list) < 10:
                    time.sleep(1)
                if e <= T:
                    # 统计各种次数
                    if freq == freq_3:
                        c_3 += 1
                    elif freq == freq_5:
                        c_5 += 1
                    else:
                        c_d += 1
                    action_count += 1
                    log.logger.debug('Action ' + str(action_count) + ' is valid with error:' + str(e) +
                                     ', count + 1.\nProcess set: ' + str(y_list) + '.\nFrame from ' +
                                     str(frame_range[0]) + ' to ' + str(frame_range[1]))
                    # 第一个动作记录roi框
                    if action_count == 1:
                        roi = True
                        log.logger.info('ROI has been generated.')
                    # 上次没成功，重新采样动作模板，确定频率
                    if not last_action_match:
                        waveform_dense, waveform_5, waveform_3 = [], [], []
                        mid = int(len(y_list) / 2)
                        for i in range(0, len(y_list)):
                            waveform_dense.append([i / round(len(y_list), 2), y_list[i][1]])
                            if i == 0:
                                waveform_5.append([0, y_list[i][1]])
                                waveform_3.append([0, y_list[i][1]])
                            elif i == int(mid / 2):
                                waveform_5.append([i / round(len(y_list), 2), y_list[i][1]])
                            elif i == int((mid + len(y_list) - 1) / 2):
                                waveform_5.append([i / round(len(y_list), 2), y_list[i][1]])
                            elif i == mid:
                                waveform_5.append([0.5, y_list[i][1]])
                                waveform_3.append([0.5, y_list[i][1]])
                            elif i == len(y_list) - 1:
                                waveform_5.append([1, y_list[i][1]])
                                waveform_3.append([1, y_list[i][1]])
                        if len(waveform_5) < 5:
                            waveform_5.append([1, y_list[-1][1]])
                        if len(waveform_5) < 3:
                            waveform_3.append([1, y_list[-1][1]])
                        freq_5 = int(freq * len(y_list) / 4)
                        freq_3 = int(freq * len(y_list) / 2)
                        method['3'], method['5'], method['dense'] = waveform_3, waveform_5, waveform_dense
                        waveform = method['5']
                        freq = freq_5
                        last_action_match = True
                    # 上次成功，5 to 3
                    else:
                        waveform = method['3']
                        freq = freq_3
                    # 误差过大，转为dense
                    if e > 0.2:
                        waveform = method['dense']
                        freq = freq_dense
                        last_action_match = False
                # 动作失败，转为dense
                else:
                    last_action_match = False
                    waveform = method['dense']
                    freq = freq_dense
                    log.logger.debug('Action ' + str(action_count + 1) + ' is invalid with error = ' + str(e) +
                                     '.\nProcess set: ' + str(y_list) + '.\nFrame from ' + str(frame_range[0]) +
                                     ' to ' + str(frame_range[1]))
                # 清空动作列表，进入下一个动作
                y_list = []
            # 动作没结束
            else:
                # 去除开头多余的动作
                if len(y_list) == 1 and (y < 0.03 or y_list[-1][1] + 0.06 > y):
                    y_list.pop(0)
                    y_list.append([index, y])
                    index += freq
                    continue
                # 如果是第一个动作，直接加入动作列表
                if action_count > 0:
                    # print(waveform)
                    # 采样数小于模板长度，和模板进行对比；否则直接加入动作列表
                    if waveform_index < len(waveform):
                        if abs(y - waveform[waveform_index][1]) <= 0.3:
                            y_list.append([index, y])
                            index += freq
                            waveform_index += 1
                        else:
                            # print(y, waveform[waveform_index][1])
                            last_action_match = False
                            if freq == freq_dense:
                                y_list.append([index, y])
                                index += freq
                                waveform_index += 1
                            else:
                                top = False
                                up = False
                                if len(y_list) > 0:
                                    index = y_list[0][0]
                                waveform_index = 0
                                waveform = method['dense']
                                freq = freq_dense
                                y_list = []
                    else:
                        last_action_match = False
                        y_list.append([index, y])
                        index += freq
                        waveform_index += 1
                else:
                    y_list.append([index, y])
                    index += freq
                    waveform_index += 1

    # 判断剩余的帧
    if len(y_list) > 2 and y_list[-1][0] - y_list[0][0] > 15:
        y_list = utils.frame_regularization(y_list)
        log.logger.debug('Action ' + str(action_count + 1) + ' ended, judging...')
        e = judging(y_list, mapper)
        if e <= T:
            action_count += 1
            log.logger.debug('Action ' + str(action_count) + ' is valid with error:' + str(e) +
                             ', count + 1.\nProcess set: ' + str(y_list))
        else:
            log.logger.debug(
                'Action ' + str(action_count + 1) + ' is invalid with error = ' + str(e) +
                '.\nProcess set: ' + str(y_list))
    end = time.time()
    log.logger.info('Counting result is ' + str(action_count) + '.')
    return action_count, int(end - start - roi_time), frame_count, c_3, c_5, c_d


if __name__ == '__main__':
    # for action in ["data/sit-up-compress"]:
    for action in ["data/pull-up-compress", "data/sit-up-compress", "data/push-up-compress"]:
        action_path = action
        for dir in os.listdir(action_path):
            if os.path.isdir(action_path + "/" + dir):
                if "roi" in dir:
                    continue
                print('Video ' + dir + ' is been counted.')
                img_path = action_path + "/" + dir
                action_count, action_time, frame_count, c_3, c_5, c_d = counting(img_path)
                print('Count is ' + str(action_count))
                # with open('compress.csv', 'a+', newline='') as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow([dir, round(min(action_count, int(dir.split("-")[1])) / max(action_count, int(dir.split("-")[1])), 3),
                #                      int(1000 * action_time / int(dir.split("-")[1]))])
                    # writer.writerow([dir, int(dir.split("-")[1]), c_3, c_5, c_d])