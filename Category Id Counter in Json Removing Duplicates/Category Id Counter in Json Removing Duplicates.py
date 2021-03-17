import json
import glob
import os
from collections import defaultdict


class Parsing:

    @staticmethod
    def parsing(path_to_json):
        shirt = 0
        tshirt = 0
        pants = 0
        image_set = defaultdict(set)
        json_files = os.path.join(path_to_json, '*.json')
        for file_name in glob.glob(json_files):
            with open(file_name) as file:
                data = json.load(file)
            for key1 in data["annotations"]:
                img_id = key1["image_id"]
                v = key1["category_id"]
                image_set[img_id].add(v)
            for k, i in image_set.items():
                for f in i:
                    if f == 1:
                        shirt = shirt + 1
                    if f == 2:
                        tshirt = tshirt + 1
                    if f == 3:
                        pants = pants + 1
            image_set.clear()
        print("Shirts Number:", shirt)
        print("T-Shirts Number:", tshirt)
        print("Pants Number:", pants)


path = 'C:\\Users\\Vargha\\Desktop\\Symo\\Json Files\\'
Parsing.parsing(path)
