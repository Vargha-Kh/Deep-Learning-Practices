from PIL import Image
import os
os.chdir('C:\\Users\\Vargha\\Desktop\\New folder')
target = 'C:\\Users\\Vargha\\Desktop\\New folder (2)'

count = 1
for filename in (os.listdir()):
    image = Image.open(filename)
    image = image.convert('RGB')
    image_name = "{fcount}.png".format(fcount=count)
    image.save(os.path.join(target, image_name), 'png')
    count += 1

