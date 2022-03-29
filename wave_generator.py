import os

import mapping
import utils
import shutil

# deciding whether the action has begun
t = 0.1
# threshold for judging whether the action is valid
sample = "train"
mapper = mapping.pull_up_mapping


def generate(videos):
    p = []
    for dir_path, dir_names, file_names in os.walk(videos):
        for path in dir_names:
            for save_dir in ["./generate_cache/output_images", "./generate_cache/output_jsons"]:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                    os.makedirs(save_dir)
                else:
                    os.makedirs(save_dir)
            f = []
            flag = [False, -1]
            skeleton_data = utils.skeleton_extraction('--image_dir', videos + "/" + path)
            for item in skeleton_data:
                y = mapper(item)
                if y[1] is not -1:
                    print("Mapped frame " + str(y[0]) + " result is " + str(y[1]))
                    deviation = abs(y[1])
                    if flag[0] is False and deviation < t:
                        flag[1] = y[0]
                    elif flag[0] is False and deviation >= t:
                        flag[0] = True
                        f.append([flag[1], 0])
                        f.append(y)
                    elif flag[0] is True and deviation >= t:
                        f.append(y)
                    elif flag[0] is True and deviation < t:
                        f.append([y[0], 0])
                        print("Waveform points for this term are:\n" + str(f))
                        p = utils.insert_points(p, utils.frame_regularization(f))
                        print("Total waveform points are:\n" + str(p))
                        f = []
                        flag = [False, y[0]]
                elif flag[0] is True:
                    print("Lost frame " + str(y[0]) + " follows last frame")
                    f.append([y[0], f[-1][1]])
            if flag[0] is True:
                f.append([f[-1][0] + 1, 0])
                print("Waveform points for this term are:\n" + str(f))
                p = utils.insert_points(p, utils.frame_regularization(f))
                print("Total waveform points are:\n" + str(p))
    utils.trapezoidal_fitting(p, 6)


if __name__ == '__main__':
    generate(sample)
