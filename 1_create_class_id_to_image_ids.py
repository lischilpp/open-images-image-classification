import csv
import config
from pathlib import Path

print('reading class ids')

class_ids = []
with open(config.FILEPATH_CLASS_NAMES, encoding='utf-8') as f:
    next(f)
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        class_id = row[0]
        class_ids.append(class_id)


print('matching class ids to image ids')

class_id_to_image_ids = {}
for class_id in class_ids:
    class_id_to_image_ids[class_id] = []

label_files = config.DIRPATH_LABELS.glob('*.csv')
for filename in label_files:
    with open(filename, encoding='utf-8') as f:
        next(f)
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            image_id = row[0]
            class_id = row[2]
            confidence = float(row[3])
            if  confidence >= config.MINIMUM_CONFIDENCE_FOR_LABEL \
            and class_id in class_ids:
                class_id_to_image_ids[class_id].append(image_id)

print('writing result to file')

if not config.DIRPATH_PROCESSING.exists():
    config.DIRPATH_PROCESSING.mkdir()

f = open(config.FILEPATH_CLASS_ID_TO_IMAGE_IDS, "w", encoding='utf-8')
for class_id in class_ids:
    image_ids = class_id_to_image_ids[class_id]
    image_ids_str = ';'.join(image_ids)
    f.write(f'{class_id},{image_ids_str}\n')
f.close()

print('-------------- DONE --------------')