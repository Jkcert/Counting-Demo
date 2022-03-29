# Mapping a skeleton data to a process value(2-D)

# Confidence threshold of skeleton point
# C = 0.6

import math

test_skeleton = [7,
                 [[0, 0, 0], [197.164, 161.491, 0.896455], [237.798, 158.156, 0.840285], [259.772, 124.262, 0.871383],
                  [253.03, 68.2592, 0.912764], [159.813, 161.534, 0.834573], [141.215, 125.88, 0.902222],
                  [147.987, 64.9346, 0.871515], [197.162, 288.63, 0.71238], [222.506, 288.636, 0.686935],
                  [212.342, 386.976, 0.880746], [205.54, 491.988, 0.811595], [173.407, 288.634, 0.675585],
                  [186.92, 388.62, 0.857565], [183.509, 486.939, 0.829572], [0, 0, 0], [0, 0, 0],
                  [220.871, 122.566, 0.908147], [178.456, 119.106, 0.884239], [183.535, 507.266, 0.575368],
                  [178.456, 505.561, 0.614999], [181.896, 497.058, 0.770747], [212.357, 514.067, 0.613764],
                  [212.334, 512.348, 0.639208], [203.833, 500.513, 0.782646]]]

pull_up_flag = False
max_angle = 0
min_angle = 180


def midpoint(points):
    x = 0
    y = 0
    n = len(points)
    for item in points:
        if item[2] > 0:
            x += item[0]
            y += item[1]
    x /= n
    y /= n
    return [x, y]


def point_distance(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# simplest version mappings
def push_up_mapping(skeleton, skt_idx):
    for index, skt in enumerate(skeleton):
        if push_up_pose(skt) is True:
            skeleton = skt
            break
        elif index == len(skeleton) - 1:
            return -1
    # if skeleton[1][11][2] * skeleton[1][14][2] > 0 and \
    #         point_distance(skeleton[1][11], skeleton[1][14]) * 4 < point_distance(skeleton[1][1], skeleton[1][8]):
    #     fulcrum = midpoint([skeleton[1][11], skeleton[1][14]])
    # else:
    #     fulcrum = skeleton[1][11] if skeleton[1][11][2] > skeleton[1][14][2] else skeleton[1][14]
    # h = abs(fulcrum[1] - skeleton[1][1][1])
    # k = point_distance(fulcrum, skeleton[1][1])
    # i = h / k
    # if i >= 0.35:
    #     print([skeleton[0], 0])
    #     return [skeleton[0], 0]
    # elif i <= 0.05:
    #     print([skeleton[0], 1])
    #     return [skeleton[0], 1]
    # else:
    #     print([skeleton[0], 1 - i / 0.35])
    #     return [skeleton[0], 1 - i / 0.35]

    # for index, skt in enumerate(skeleton):
    # if push_up_pose(skeleton[0], skt) is True:
    #     skeleton = [skeleton[0], skt]
    # elif index == len(skeleton[1]) - 1:
    #     return [skeleton[0], -1]
    # skeleton = skeleton[skt_idx]
    if skeleton[10][2] * skeleton[13][2] > 0 and \
            point_distance(skeleton[10], skeleton[13]) * 4 < point_distance(skeleton[1], midpoint([skeleton[8], skeleton[11]])):
        fulcrum = midpoint([skeleton[10], skeleton[13]])
    else:
        fulcrum = skeleton[10] if skeleton[10][2] > skeleton[13][2] else skeleton[13]
    h = abs(fulcrum[1] - skeleton[1][1])
    k = point_distance(fulcrum, skeleton[1])
    i = h / k
    if i >= 0.25:
        # print(0)
        return 0
    elif i <= 0.05:
        # print(1)
        return 1
    else:
        # print(1 - i / 0.35)
        return 1 - i / 0.25


def sit_up_mapping(skeleton, skt_idx):
    for index, skt in enumerate(skeleton):
        if sit_up_pose(skt) is True:
            skeleton = skt
            break
        elif index == len(skeleton) - 1:
            return -1
    # if skeleton[1][1][1] == 0:
    #     return [skeleton[0], -1]
    # h = abs(skeleton[1][8][1] - skeleton[1][1][1])
    # k = point_distance(skeleton[1][8], skeleton[1][1])
    # i = h / k
    # if i <= 0:
    #     return [skeleton[0], 0]
    # elif i >= 0.95:
    #     return [skeleton[0], 1]
    # else:
    #     return [skeleton[0], i]

    # for index, skt in enumerate(skeleton):
    #     if sit_up_pose(skt) is True:
    #         skeleton = skt
    #         break
    #     elif index == len(skeleton) - 1:
    #         return -1
    # if skt_idx < len(skeleton):
    #     skeleton = skeleton[skt_idx]
    else:
        # print("skeleton miss")
        return -1
    if skeleton[1][2] * (skeleton[8][2] + skeleton[11][2]) == 0:
        return -1
    if skeleton[8][2] != 0:
        if skeleton[11][2] == 0:
            h = abs(skeleton[8][1] - skeleton[1][1])
            k = point_distance(skeleton[8], skeleton[1])
        else:
            h = abs(midpoint([skeleton[8], skeleton[11]])[1] - skeleton[1][1])
            k = point_distance(midpoint([skeleton[8], skeleton[11]]), skeleton[1])
    else:
        h = abs(skeleton[11][1] - skeleton[1][1])
        k = point_distance(skeleton[11], skeleton[1])
    i = h / k
    if i <= 0:
        return 0
    elif i >= 0.95:
        return 1
    else:
        return i


def pull_up_mapping(skeleton, skt_idx):
    for index, skt in enumerate(skeleton):
        if pull_up_pose(skt):
            skeleton = skt
            break
        elif index == len(skeleton) - 1:
            return -1
    # skeleton = skeleton[skt_idx]
    global pull_up_flag
    angle = compute_angle(skeleton[3], skeleton[2], skeleton[3], skeleton[4])
    # print("p: ", p)
    if angle > 150:
        pull_up_flag = False
        # print('Initial position.')
        return 0
    elif angle <= 50:
        if pull_up_flag is False:
            pull_up_flag = True
            # print('Pulled up.')
            return 1
        else:
            return 1
    else:
        pull_up_flag = False
        # print(1 - angle / max_angle)
        return 1 - (angle - 50) / 100
    # print('Wrist distance too large.')
    # return -1


# pull_up_mapping(test_skeleton)
def pull_up_pose(skeleton):
    # print(skeleton[2], skeleton[3], skeleton[4])
    if (skeleton[4][1] < skeleton[2][1] or (skeleton[4][1] < skeleton[3][1] and skeleton[2][1] < skeleton[3][1])) \
            and skeleton[1][1] < skeleton[8][1] < skeleton[9][1]:
        return True
    return False


def sit_up_pose(skeleton):
    h = []
    for i in [skeleton[8], skeleton[11]]:
        if i[2] > 0:
            h.append(i)
    if len(h) == 0:
        return False
    h = midpoint(h)
    left = skeleton[12][2] * skeleton[13][2]
    right = skeleton[9][2] * skeleton[10][2]
    if left > right and left > 0:
        i = skeleton[12]
        j = skeleton[13]
    elif right > left and right > 0:
        i = skeleton[9]
        j = skeleton[10]
    else:
        return False
    if 45 < compute_angle(h, i, j, i) <= 135:
        return True
    return False


def push_up_pose(skeleton):
    if skeleton[1][2] == 0:
        return False
    elif (skeleton[8][2] + skeleton[11][2]) * (skeleton[10][2] + skeleton[13][2]) > 0:
        h = skeleton[1]
        j = []
        for i in [skeleton[8], skeleton[11]]:
            if i[2] > 0:
                j.append(i)
        if len(j) == 0:
            return False
        j = midpoint(j)
        k = []
        for i in [skeleton[10], skeleton[13]]:
            if i[2] > 0:
                k.append(i)
        if len(k) == 0:
            return False
        k = midpoint(k)
        if 160 < compute_angle(h, j, k, j) <= 180:
            return True
        return False


def compute_angle(p1, p2, p3, p4):
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p4[0] - p3[0]
    dy2 = p4[1] - p3[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle
