import os
import json
from collections import defaultdict


class Parsing:

    @staticmethod
    def json_files_address(path_to_json):
        for root, DIR, json_files in os.walk(path_to_json):
            return json_files, root

    @staticmethod
    def get_files_name(json_files, root):
        if json_files.endswith(".json"):
            file_path = os.path.join(root, json_files)
            with open(file_path) as file:
                data_load = json.load(file)
                return data_load
        else:
            print("There Is No Json Files!!")

    @staticmethod
    def label_count(root):
        shirt = 0
        tshirt = 0
        pants = 0
        for file in json_file:
            data1 = Parsing.get_files_name(file, root)
            shirt1, tshirt1, pants1 = Parsing.block_calculator(data1)
            shirt += shirt1
            tshirt += tshirt1
            pants += pants1
        return shirt, tshirt, pants

    @staticmethod
    def block_calculator(data1):
        shirt = 0
        tshirt = 0
        pants = 0
        image_set = defaultdict(set)
        for key1 in data1["annotations"]:
            img_id = key1['image_id']
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
        return shirt, tshirt, pants

    @staticmethod
    def print_labels(shirt, tshirt, pants):
        print("Shirts Number:", shirt)
        print("T-Shirts Number:", tshirt)
        print("Pants Number:", pants)


path = 'C:\\Users\\Vargha\\Desktop\\Symo\\Json Files\\'
json_file, root = Parsing.json_files_address(path)
shirt, tshirt, pants = Parsing.label_count(root)
Parsing.print_labels(shirt, tshirt, pants)
