import csv
import config

# create class_id_to_name

class_ids = []
with open(config.FILEPATH_CLASS_ID_TO_NAME) as f:
    next(f)
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        class_id = row[0]
        class_ids.append(class_id)


# match each image id to a list of class ids

class_id_to_image_ids = {}
for class_id in class_ids:
    class_id_to_image_ids[class_id] = []

label_files = config.DIRPATH_LABELS.glob('*.csv')
for filename in label_files:
    with open(filename) as f:
        next(f)
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            image_id = row[0]
            class_id = row[2]
            confidence = float(row[3])
            if confidence >= config.MINIMUM_CONFIDENCE_FOR_LABEL:
                if class_id in class_ids:
                    class_id_to_image_ids[class_id].append(image_id)

f = open('processing/class_id_to_image_ids.csv', "w")

for class_id in class_ids:
    image_ids = class_id_to_image_ids[class_id]
    image_ids_str = ';'.join(image_ids)
    f.write(f'{class_id},{image_ids_str}\n')

f.close()
