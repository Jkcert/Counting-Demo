import random
import numpy as np
import matplotlib.pyplot as plt


# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/openpose/Release')
            # sys.path.append(dir_path + '/../../python/openpose/Release');
            # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose/x64/Release;' + dir_path + '/openpose/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="./openpose/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./openpose/models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)



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
    return -1


# TODO: 返回[[x,y,c],[x,y,c],...]25个数组
def skeleton_extraction(video):
    r = []
    for i in range(25):
        r.append([i, i, random.random()])
    return r


# mat_plot poly_fitting
def simple_fitting(points):
    if len(points) > 0:
        x = []
        y = []
        for (index, item) in enumerate(points):
            x.append(item[0])
            y.append(item[1])
        x = np.array(x)
        y = np.array(y)
        f1 = np.polyfit(x, y, 3)
        p1 = np.poly1d(f1)
        print('p1 is :\n', p1)
        y_val = p1(x)  # 拟合y值
        print('yvals is :\n', y_val)
        # 绘图
        plot1 = plt.plot(x, y, 's', label='original values')
        plot2 = plt.plot(x, y_val, 'r', label='poly_fit values')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc=4)  # 指定legend的位置右下角
        plt.title('poly_fitting')
        plt.show()
