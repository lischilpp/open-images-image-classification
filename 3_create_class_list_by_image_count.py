import functools
import csv

class_ids = []
class_id_to_name = {}
with open('in/class_id_to_name.csv') as f:
    next(f)
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        class_id = row[0]
        class_name = row[1].lower()
        class_ids.append(class_id)
        class_id_to_name[class_id] = class_name

class_id_to_image_ids = {}
with open('processing/class_id_to_image_ids.csv') as f:
    for line in f:
        line_arr = line.rstrip().split(',')
        class_id = line_arr[0]
        image_ids = line_arr[1].split(';')
        class_id_to_image_ids[class_id] = image_ids


# sort trainable_class_ids by training data size

def compare(x1, x2):
    return len(class_id_to_image_ids[x2]) - len(class_id_to_image_ids[x1])


class_ids = sorted(
    class_ids, key=functools.cmp_to_key(compare))


f = open('processing/class_list_by_image_count.csv', "w")

for class_id in class_ids:
    class_name = class_id_to_name[class_id]
    if ',' in class_name:
        class_name = f'"{class_name}"'
    count = len(class_id_to_image_ids[class_id])
    f.write(f'{class_name}, {count}\n')

f.close()
