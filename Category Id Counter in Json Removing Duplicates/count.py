import json
import os
from collections import defaultdict


def parsing(path_to_json):
    name_dict = {1: 'shirt', 2: 't-shirt', 3: 'pants'}
    img_dict = defaultdict(set)
    count_dict = defaultdict(int)
    for root, dirs, files in os.walk(path_to_json):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path) as file_:
                    data = json.load(file_)
                for key1 in data["annotations"]:
                    img_id = key1['image_id']
                    v = key1["category_id"]
                    img_dict[img_id].add(v)
                for image_id, set_ in img_dict.items():
                    for i in set_:
                        count_dict[i] += 1
                img_dict.clear()
    for id_, count in count_dict.items():
        print(name_dict[id_], f": {count}")


if __name__ == '__main__':
    path = 'C:\\Users\\Vargha\\Desktop\\Symo\\Json Files\\'
    parsing(path)
