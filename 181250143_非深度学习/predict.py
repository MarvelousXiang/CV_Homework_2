import json
from math import sqrt
from random import random

import joblib
import cv2 as cv
import os
import numpy as np

middle_point_number = 25
frames = [[16, 80], [32, 80], [48, 80], [64, 80], [80, 80], [80, 16], [80, 32], [80, 48], [80, 64]]
model = joblib.load("SimpleClassifier.pkl")


# 目标检测入口，传入灰度图进行标注
def detect(gray_pic):
    # 根据上述初始化结果生成这225个窗口，格式为[xmin, ymin, xmax, ymax]
    windows = []
    # cv2图片格式是先y坐标后x坐标
    height = np.shape(gray_pic)[0]
    width = np.shape(gray_pic)[1]
    for y in range(int(sqrt(middle_point_number))):
        for x in range(int(sqrt(middle_point_number))):
            y_loc = y * (height // int(sqrt(middle_point_number)))
            x_loc = x * (width // int(sqrt(middle_point_number)))
            middle_point_frames = []
            for frame in frames:
                y_min = max([0, y_loc - frame[0] // 2])
                y_max = min([height, y_loc + frame[0] // 2])
                x_min = max([0, x_loc - frame[1] // 2])
                x_max = min([width, x_loc + frame[1] // 2])
                middle_point_frames.append([x_min, y_min, x_max, y_max])
            # 用于探测的窗口已经准备好，截取被框取的图片，对其进行数据归一化
            to_predict = []
            for frame in middle_point_frames:
                cut = gray_pic[frame[1]:frame[3], frame[0]:frame[2]]
                try:
                    new_cut = cv.resize(cut, [32, 32])
                except:
                    continue
                temp = []
                for line in list(new_cut):
                    for item in list(line):
                        temp.append(item / 255)
                to_predict.append(temp)
            # 预测每一个窗口的标签和对应的后验概率
            predict_class = list(model.predict(to_predict))
            predict_proba = list(model.predict_proba(to_predict))
            # 对这一部分窗口取交集
            # 筛除分类不合理的窗口，依据：输出的后验概率差别不大
            remove_list = []
            for i in range(len(predict_proba)):
                proba = list(predict_proba[i])
                proba.sort(reverse=True)
                if proba[0] - proba[1] <= 0.001:
                    remove_list.append(i)
            # 不反转下面会报错（索引出问题）
            remove_list.reverse()
            for item in remove_list:
                del predict_class[item]
                del predict_proba[item]
                del to_predict[item]
                del middle_point_frames[item]
            # 对剩下的窗口取交集，依据：按照最多的分类标签取
            # 取标签
            if len(predict_class) == 0:
                continue
            class_dict = dict.fromkeys(predict_class, 0)
            for item in predict_class:
                class_dict[item] += 1
            sorted_dict = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
            if not len(sorted_dict) == 1 and sorted_dict[0][1] == sorted_dict[1][1]:
                continue
            # 取交集
            aim_class = sorted_dict[0][0]
            min_x = 0
            max_x = width
            min_y = 0
            max_y = height
            for i in range(len(predict_class)):
                if predict_class[i] == aim_class:
                    min_x = max(min_x, middle_point_frames[i][0])
                    min_y = max(min_y, middle_point_frames[i][1])
                    max_x = min(max_x, middle_point_frames[i][2])
                    max_y = min(max_y, middle_point_frames[i][3])
            windows.append([aim_class, [min_x, min_y, max_x, max_y]])
    # 每一个窗口处理完毕之后还要对这25个窗口取交集（属于同一类别的有交集的窗口合并）
    final_result = []
    while True:
        # 用来存储所有的窗口是否存在相交的情况
        cross_index = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                if windows[i][0] == windows[j][0]:
                    # 同类别，判断有无相交
                    # j的任何一个角点落在i框中就算相交
                    first = windows[i][1]
                    second = windows[j][1]
                    if second[0] in range(first[0], first[2]) and second[1] in range(first[1], first[3]):
                        # 取交集加入结果，索引存留在相交的索引列表中
                        min_x = max([first[0], second[0]])
                        min_y = max([first[1], second[1]])
                        max_x = min([first[2], second[2]])
                        max_y = min(first[3], second[3])
                        # 多次含有同一个元素会造成死循环
                        result = [windows[i][0], [min_x, min_y, max_x, max_y]]
                        if not final_result.__contains__(result):
                            final_result.append(result)
                        if not cross_index.__contains__(i):
                            cross_index.append(i)
                        if not cross_index.__contains__(j):
                            cross_index.append(j)
                    elif second[0] in range(first[0], first[2]) and second[3] in range(first[1], first[3]):
                        min_x = max([first[0], second[0]])
                        min_y = max([first[1], second[1]])
                        max_x = min([first[2], second[2]])
                        max_y = min(first[3], second[3])
                        result = [windows[i][0], [min_x, min_y, max_x, max_y]]
                        if not final_result.__contains__(result):
                            final_result.append(result)
                        if not cross_index.__contains__(i):
                            cross_index.append(i)
                        if not cross_index.__contains__(j):
                            cross_index.append(j)
                    elif second[2] in range(first[0], first[2]) and second[1] in range(first[1], first[3]):
                        min_x = max([first[0], second[0]])
                        min_y = max([first[1], second[1]])
                        max_x = min([first[2], second[2]])
                        max_y = min(first[3], second[3])
                        result = [windows[i][0], [min_x, min_y, max_x, max_y]]
                        if not final_result.__contains__(result):
                            final_result.append(result)
                        if not cross_index.__contains__(i):
                            cross_index.append(i)
                        if not cross_index.__contains__(j):
                            cross_index.append(j)
                    elif second[2] in range(first[0], first[2]) and second[3] in range(first[1], first[3]):
                        min_x = max([first[0], second[0]])
                        min_y = max([first[1], second[1]])
                        max_x = min([first[2], second[2]])
                        max_y = min(first[3], second[3])
                        result = [windows[i][0], [min_x, min_y, max_x, max_y]]
                        if not final_result.__contains__(result):
                            final_result.append(result)
                        if not cross_index.__contains__(i):
                            cross_index.append(i)
                        if not cross_index.__contains__(j):
                            cross_index.append(j)
                    else:
                        pass
        for i in range(len(windows)):
            if not i in cross_index:
                # 不相交的加进来，相交的已经处理过了，留了交集
                final_result.append(windows[i])
        windows = final_result
        final_result = []
        # 数组里不存在相交窗口了，退出循环
        if len(cross_index) == 0:
            break
    objects = {}
    for item in windows:
        objcet_info = {}
        object_id = str(windows.index(item))
        category = item[0]
        bbox = item[1]
        objcet_info["category"] = category
        objcet_info["bbox"] = bbox
        objects[object_id] = objcet_info

    return objects


# 标注主程序
predict_path = "./test/"
img_dir = os.listdir(predict_path)
result_dict = {}
for file_name in img_dir:
    print(str(img_dir.index(file_name) + 1) + "/" + str(len(img_dir)))
    location = predict_path + file_name
    img = cv.imread(location)
    picture = {}
    # 标注图片的长宽和通道数
    shape = np.shape(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    picture["height"] = shape[0]
    picture["width"] = shape[1]
    picture["depth"] = shape[2]
    picture["objects"] = detect(gray)
    result_dict[file_name] = picture
json_str = json.dumps(result_dict)
with open("test.json", mode="w") as f:
    f.write(json_str)

