# Preset waveform function for different action processes
import json
import numpy as np
import math
import matplotlib.pyplot as plt

Pi = math.pi


def basic_sine(x):
    return (math.cos(-2 * Pi * x) + 1) / 2


def pull_up_poly(x):
    if x == 1 or x == 0:
        return 0
    return 59.37 * x ** 6 - 202.2 * x ** 5 + 256.2 * x ** 4 - 140.6 * x ** 3 + 23.61 * x ** 2 + 3.681 * x - 0.01616


def push_up_poly(x):
    if x == 1 or x == 0:
        return 0
    return x ** 6 * -160.2 + x ** 5 * 488.3 + x ** 4 * -553.4 + x ** 3 * 283 + x ** 2 * -64.73 + x * 7.051 + 0.01927


def sit_up_poly(x):
    if x == 1 or x == 0:
        return 0
    r = x ** 6 * 9.664 + x ** 5 * -29.19 + x ** 4 * 28.05 + x ** 3 * -7.352 + x ** 2 * -5.963 + x * 4.738 + 0.06725
    return r


action_dic = {}
pull_up_feature = [[336.6470947265625, 196.22607421875, 0.3569912612438202],
                   [311.70069885253906, 244.87152862548828, 0.8174419105052948],
                   [341.6363525390625, 243.62420654296875, 0.8108536601066589],
                   [364.0881042480469, 173.7743377685547, 0.8831974267959595],
                   [381.5505676269531, 123.88156127929688, 0.9775703549385071],
                   [281.7650451660156, 246.1188507080078, 0.8240301609039307],
                   [256.81866455078125, 176.2689666748047, 0.8581821918487549],
                   [244.345458984375, 123.88156127929688, 0.8922082185745239],
                   [334.1524353027344, 350.8936462402344, 0.7836722135543823],
                   [339.1417236328125, 338.42047119140625, 0.08282695710659027],
                   [324.17388916015625, 340.91510009765625, 0.020623071119189262],
                   [291.74359130859375, 348.3990173339844, 0.8669601678848267],
                   [311.70068359375, 335.92584228515625, 0.020525753498077393],
                   [276.7757568359375, 268.5705871582031, 0.03976046293973923],
                   [334.1524353027344, 196.22607421875, 0.4285731911659241],
                   [291.74359130859375, 188.74215698242188, 0.37375015020370483],
                   [329.1631774902344, 203.70999145507812, 0.9933605194091797],
                   [289.24896240234375, 203.70999145507812, 0.7576113939285278]]
push_up_feature = [[568.2879638671875, 105.42707061767578, 0.9009503126144409],
                   [490.95738220214844, 68.79681015014648, 0.8224585056304932],
                   [486.8873596191406, 81.00689697265625, 0.8539057970046997],
                   [470.60723876953125, 194.96771240234375, 0.8544957637786865],
                   [486.8873596191406, 292.6484069824219, 0.944424033164978],
                   [495.02740478515625, 56.58672332763672, 0.7910112142562866],
                   [462.4671936035156, 170.5475311279297, 0.5828210115432739],
                   [486.8873596191406, 243.8080596923828, 0.41814079880714417],
                   [299.666015625, 97.2870101928711, 0.8923789262771606],
                   [169.42510986328125, 162.407470703125, 0.9211053252220154],
                   [55.46429443359375, 203.10775756835938, 0.828125536441803],
                   [307.80609130859375, 89.14695739746094, 0.6710587739944458],
                   [185.70521545410156, 154.2674102783203, 0.7285734415054321],
                   [79.88446807861328, 170.5475311279297, 0.634394109249115],
                   [576.427978515625, 89.1469497680664, 0.9668546319007874],
                   [584.5680541992188, 97.2870101928711, 0.9550734758377075],
                   [560.1478881835938, 56.58672332763672, 0.9672874212265015],
                   [552.0078125, 56.58672332763672, 0.6332676410675049]]
sit_up_feature = [[274.1896057128906, 519.4857177734375, 0.8671795129776001], [422.24945068359375, 614.6670532226562, 0.564810186624527], [379.9466247558594, 646.3941650390625, 0.5431965589523315], [104.97833251953125, 688.6969604492188, 0.679581880569458], [168.43255615234375, 625.2427368164062, 0.46004629135131836], [464.5522766113281, 582.93994140625, 0.5864238142967224], [379.9466247558594, 540.6370849609375, 0.367825984954834], [802.9747924804688, 413.7286682128906, 0.31849855184555054], [929.8832397460938, 646.3941650390625, 0.7793871164321899], [1268.3057861328125, 329.1230163574219, 0.8322270512580872], [1479.81982421875, 646.3941650390625, 0.8703421950340271], [929.8832397460938, 561.7885131835938, 0.6263630390167236], [1268.3057861328125, 329.1230163574219, 0.8047410249710083], [1500.97119140625, 625.2427368164062, 0.6259809732437134], [253.03819274902344, 561.7885131835938, 0.8474107384681702], [274.1896057128906, 519.4857177734375, 0.7959244847297668], [274.1896057128906, 646.3941650390625, 0.9005277752876282], [295.34100341796875, 582.93994140625, 0.5401751399040222]]


action_dic['pull_up'] = {'feature': pull_up_feature, 'mapper': 'pull_up_trapezoidal', 'model': 'pull_up_mapping',
                         'judging': 'ave_error_judging1d',
                         'method': {'3': [[0, 0], [0.5, 1.0], [1, 0]],
                                    '5': [[0, 0], [0.25, 0.736], [0.5, 1.0], [0.75, 0.704], [1, 0]],
                                    'dense': [[0.0, 0], [0.05, 0.012], [0.1, 0.099], [0.15, 0.287],
                                              [0.2, 0.513], [0.25, 0.736], [0.3, 0.917],
                                              [0.35, 1.0], [0.4, 1.0], [0.45, 1.0], [0.5, 1.0],
                                              [0.55, 1.0], [0.6, 1.0], [0.65, 1.0],
                                              [0.7, 0.896], [0.75, 0.704], [0.8, 0.486],
                                              [0.85, 0.276], [0.9, 0.103], [0.95, 0.001],
                                              [1.0, 0]]
                                    },
                         # w, h, Δx1, Δx2, Δy1, Δy2
                         'roi': [130, 30, -75, 5, -20, 30]
                         }
action_dic['push_up'] = {'feature': push_up_feature, 'mapper': 'push_up_trapezoidal', 'model': 'push_up_mapping',
                         'judging': 'ave_error_judging1d',
                         'method': {'3': [[0, 0], [0.5, 0.906], [1, 0]],
                                    '5': [[0, 0], [0.25, 0.466], [0.5, 1.0], [0.75, 0.584], [1, 0]],
                                    'dense': [[0.0, 0], [0.05, 0.25], [0.1, 0.293], [0.15, 0.315],
                                              [0.2, 0.374], [0.25, 0.466], [0.3, 0.563],
                                              [0.35, 0.645], [0.4, 0.714], [0.45, 0.806], [0.5, 1],
                                              [0.55, 1], [0.6, 0.802], [0.65, 0.718],
                                              [0.7, 0.665], [0.75, 0.584], [0.8, 0.478],
                                              [0.85, 0.388], [0.9, 0.348], [0.95, 0.307],
                                              [1.0, 0]]
                                    },
                         'roi': [0, 0, 0, 0, 0, 0]}
action_dic['sit_up'] = {'feature': sit_up_feature, 'mapper': 'sit_up_trapezoidal', 'model': 'sit_up_mapping',
                        'judging': 'ave_error_judging1d',
                        'method': {'3': [[0, 0], [0.5, 1.0], [1, 0]],
                                   '5': [[0, 0], [0.1, 0.449], [0.5, 1.0], [0.9, 0.431], [1, 0]],
                                   'dense': [[0.0, 0], [0.05, 0.374], [0.1, 0.449], [0.15, 0.623],
                                             [0.2, 0.741], [0.25, 0.814], [0.3, 0.949],
                                             [0.35, 1.0], [0.4, 1.0], [0.45, 1.0], [0.5, 1.0],
                                             [0.55, 1.0], [0.6, 1.0], [0.65, 0.962],
                                             [0.7, 0.913], [0.75, 0.849], [0.8, 0.747],
                                             [0.85, 0.601], [0.9, 0.431], [0.95, 0.261],
                                             [1.0, 0]]
                                   },
                        'roi': [0, 0, 0, 0, 0, 0]}

# 生成动作字典
# action_json = json.dumps(action_dic)
# with open('action_dic.json', 'w') as json_file:
#     json_file.write(action_json)


def pull_up_trapezoidal(x=-1):
    if x == -1:
        return []
    if x == 0 or x == 1:
        return 0
    elif 0.337209 <= x <= 0.662790:
        return 1
    elif 0 < x < 0.337209:
        return -53.21 * x ** 3 + 31.46 * x ** 2 - 1.574 * x - 0.005411
    else:
        return 42.17 * x ** 3 - 99.87 * x ** 2 + 74.44 * x - 16.74


def push_up_trapezoidal(x=-1):
    if x == -1:
        return []
    if x == 0 or x == 1:
        return 0
    elif 0.5 <= x <= 5 / 9:
        return 1
    elif 0 < x < 0.5:
        return x ** 6 * -892.6 + x ** 5 * 1973 + x ** 4 * -1580 + x ** 3 * 585.5 + x ** 2 * -101.6 + x * 8.453 + 0.01701
    else:
        return x ** 6 * -2595.86 + x ** 5 * 10783.28 + x ** 4 * -18231.43 + x ** 3 * 15981.01 + x ** 2 * -7603.91 + x * 1838.06 + -171.127


def sit_up_trapezoidal(x=-1):
    if x == -1:
        return []
    if x == 0 or x == 1:
        return 0
    elif 0.304348 <= x <= 0.617647:
        return 1
    elif 0 < x < 0.304348:
        return x ** 6 * -71303.5 + x ** 5 * 72999.67 + x ** 4 * -28683.69 + x ** 3 * 5391.68 + x ** 2 * -492.011 + x * 21.9812 + -0.011
    else:
        return x ** 6 * -4665.52 + x ** 5 * 22197.315 + x ** 4 * -43730.55 + x ** 3 * 45662.286 + x ** 2 * -26659.22 + x * 8253.5 + -1057.787

# x = np.arange(0, 1.01, 0.02)
# y = [round(sit_up_trapezoidal(p), 3) for p in x]
# print(y)
#
# plt.plot(x, y, linewidth=2, color='red')
# plt.plot(x, y, 'o', ms=3)
# plt.savefig("mapper.png", dpi=600)
# # for a, b in zip(x, y):
# #     plt.text(a, b, b)
# plt.show()

