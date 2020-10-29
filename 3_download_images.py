from pathlib import Path
import csv
from PIL import Image
import io
import warnings
import requests
import concurrent.futures
import config

# ignore verbose warnings from PIL
warnings.filterwarnings('ignore')


print('reading class ids and names')

id_files = config.DIRPATH_IDS.glob('*.csv')
class_name_to_id = {}
class_id_to_name = {}
with open(config.FILEPATH_CLASS_NAMES) as f:
    next(f)
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        class_id = row[0]
        class_name = row[1].lower()
        class_name_to_id[class_name] = class_id
        class_id_to_name[class_id] = class_name


print('reading bounding boxes')

box_for_image = {}
boxes_files = config.DIRPATH_BOUNDING_BOXES.glob('*.csv')
for filename in boxes_files:
    with open(filename) as f:
        next(f)
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            image_id = row[0]
            class_id = row[2]
            class_name = class_id_to_name[class_id]
            x_min = float(row[4])
            x_max = float(row[5])
            y_min = float(row[6])
            y_max = float(row[7])

            if not class_name in box_for_image:
                box_for_image[class_name] = {}

            box = (x_min, y_min, x_max, y_max)
            box_for_image[class_name][image_id] = box



print('reading image urls')

# set url index depending on resolution
image_url_index = 10
if config.DOWNLAD_IMAGES_HIGH_RESOLUTION:
    image_url_index = 2

image_id_to_url = {}
for filename in id_files:
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        next(f)
        for row in reader:
            image_id = row[0]
            image_url = row[image_url_index]
            image_id_to_url[image_id] = image_url

print('reading class id to image ids mapping')

class_id_to_image_ids = {}
with open(config.FILEPATH_CLASS_ID_TO_IMAGE_IDS) as f:
    for line in f:
        line_arr = line.rstrip().split(',')
        class_id = line_arr[0]
        image_ids = line_arr[1].split(';')
        class_id_to_image_ids[class_id] = image_ids


threaded_executor = concurrent.futures.ThreadPoolExecutor()


def download_url_to_file(url, filename, box):
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        if (response.status_code == 200):
            if box:
                # crop image to bounding box
                img = Image.open(io.BytesIO(response.content))
                width, height = img.size
                box = (box[0] * width, box[1] * height,
                       box[2] * width, box[3] * height)
                img = img.crop(box)
                img.save(filename)
            else:
                open(filename, 'wb').write(response.content)

    except Exception as e:
        print(e)


def download_images_for_class_name(class_name, class_path, image_ids):
    if not class_path.exists():
        class_path.mkdir()

    class_has_boxes = class_name in box_for_image

    futures = []

    def download_images(path, image_ids):
        for image_id in image_ids:
            image_url = image_id_to_url[image_id]
            if image_url == '':
                continue
            image_filename = path / f'{image_id}.jpg'

            if not image_filename.exists():
                box = None
                if class_has_boxes and image_id in box_for_image[class_name]:
                    box = box_for_image[class_name][image_id]
                futures.append(
                    threaded_executor.submit(
                        download_url_to_file, image_url, image_filename, box))

    download_images(class_path, image_ids)

    concurrent.futures.wait(futures, timeout=None,
                            return_when=concurrent.futures.ALL_COMPLETED)


def download_images_for_class_list(filename):
    category_name = filename.stem
    print('--- ' + category_name)
    category_path = config.DIRPATH_IMAGE_DOWNLOAD / category_name
    if not category_path.exists():
        category_path.mkdir()

    lines = []
    with open(filename) as f:
        lines = [line.rstrip() for line in f]

    for i in range(len(lines)):
        class_name = lines[i]
        class_id = class_name_to_id[class_name]
        class_path = category_path / class_name

        print(f'class {i+1}/{len(lines)} ({class_name})')

        image_ids = class_id_to_image_ids[class_id][0:config.MAXIMUM_DOWNLOADED_IMAGE_COUNT_PER_CLASS]

        download_images_for_class_name(class_name, class_path, image_ids)


# download images for all class lists

lists_to_train = config.DIRPATH_CLASS_LISTS_TO_DOWNLOAD.glob('*.txt')

for filename in lists_to_train:
    download_images_for_class_list(filename)

print('-------------- DONE --------------')