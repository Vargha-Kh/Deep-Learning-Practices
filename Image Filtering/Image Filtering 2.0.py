from os.path import getsize

from PIL import Image
import os
import shutil
import imagehash
import hashlib
from PIL.Image import core as _imaging

os.chdir('C:\\Users\\Vargha\\Desktop\\Striped Shirt')


def filtering(width_limit, height_limit, bytes_limit, target):
    file_list = os.listdir()
    duplicates = []
    hash_keys = dict()
    max_size = 0
    for index, filename in enumerate(os.listdir()):
        image = Image.open(filename)
        width, height = image.size
        size = os.path.getsize(filename)
        hash = imagehash.average_hash(image)
        if width < width_limit or height < height_limit:
            shutil.copy(filename, target)
            os.remove(filename)
        if size < bytes_limit:
            shutil.copy(filename, target)
            os.remove(filename)
        if hash not in hash_keys:
            hash_keys[hash] = index
        else:
            if size > max_size:
                max_size = size
            else:
                duplicates.append((index, hash_keys[hash]))
    for index in duplicates:
        try:
            shutil.copy(file_list[index[0]], target)
            os.remove(file_list[index[0]])
        except:
            pass


saving_path = r'C:\\Users\\Vargha\\Desktop\\New folder (2)\\'
filtering(200, 200, 2000, saving_path)
