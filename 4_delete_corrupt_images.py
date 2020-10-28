from pathlib import Path
from PIL import Image, ImageFile
import math
import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

def check_image(fn):
    try:
        im = Image.open(fn)
        im.verify()
        im.close()
    except Exception as e:
        return False
    return True


images_count = 0
images = config.DIRPATH_IMAGE_DOWNLOAD.rglob('*.jpg')
for img in images:
    images_count += 1

i = 0
last_percentage = 0
images = config.DIRPATH_IMAGE_DOWNLOAD.rglob('*.jpg')
for img in images:
    percentage = int(math.floor(i / images_count * 100))
    if percentage != last_percentage:
        print(f'{percentage}%')
    last_percentage = percentage

    valid = check_image(img)
    if not valid:
        img.unlink()
        print(f'DELETED {img}')

    i += 1

print("-------------- DONE --------------")
