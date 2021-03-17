from PIL import Image
import os
import shutil
import imagehash
from PIL.Image import core as _imaging

os.chdir('C:\\Users\\Vargha\\Desktop\\Hawai Shirt')


def filtering(width_limit, height_limit, bytes_limit, target):
    hash_list = []
    for filename2 in (os.listdir()):
        image2 = Image.open(filename2)
        hash2 = imagehash.average_hash(image2)
        hash_list.append(hash2)

    count = 0
    for filename in (os.listdir()):
        image = Image.open(filename)
        width, height = image.size
        size = os.path.getsize(filename)
        hash = imagehash.average_hash(image)
        try:
            if hash != hash_list[count] and hash in hash_list:
                shutil.copy(filename, target)
                os.remove(filename)
                count += 1
        except IndexError:
            pass

        # if width < width_limit or height < height_limit:
        #     shutil.copy(filename, target)
        # if size < bytes_limit:
        #     shutil.copy(filename, target)


saving_path = r'C:\\Users\\Vargha\\Desktop\\New folder (2)\\'
filtering(100, 100, 2000, saving_path)
