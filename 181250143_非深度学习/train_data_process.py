import json
import cv2 as cv

object_list = []
train_path = "./train/"
signal = int(input("This will update your train test, continue? 1/0\n"))
if signal == 1:
    with open("train.json", mode="r") as f:
        src = json.load(f)
        f.close()
    with open("classify.csv", mode="w") as f:
        f.close()
    i = 0
    for key in src.keys():
        i += 1
        print(str((i / len(src)) * 100) + "%")
        img_info = src[key]
        img = cv.imread(train_path + key)
        objects = img_info["objects"]
        for object_id in objects.keys():
            object_info = objects[object_id]
            if not object_list.__contains__(object_info["category"]):
                object_list.append(object_info["category"])
            bbox = object_info["bbox"]
            cut = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            try:
                new_cut = cv.resize(cut, [32, 32])
            except:
                continue
            gray = cv.cvtColor(new_cut, cv.COLOR_BGR2GRAY)
            write = ""
            for line in list(gray):
                for item in list(line):
                    write += str(item) + ","
            write += object_info["category"] + "\n"
            with open("classify.csv", mode="a") as f:
                f.write(write)
                f.close()
