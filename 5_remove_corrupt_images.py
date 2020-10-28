from pathlib import Path
from PIL import Image, ImageFile
import math

folder_path = Path('/media/linus/ML/open_images/trainable')
extension = 'jpg'

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
images = folder_path.rglob(f'*.{extension}')
for img in images:
    images_count += 1
i = 0

last_percent = 0
images = folder_path.rglob(f'*.{extension}')
for img in images:
    percent = int(math.floor(i / images_count * 100))
    if percent != last_percent:
        print(f'{percent}%')
    last_percent = percent

    valid = check_image(img)
    if not valid:
        img.unlink()
        print("DELETED " + str(img))

    i += 1

print("-------------- DONE --------------")
