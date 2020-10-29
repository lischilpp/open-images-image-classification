from pathlib import Path
from math import *
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config


def format_percentage2(n):
    return floor(n * 10000) / 100

# init variables

IMAGE_SIZE = (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)
print(f'Using {config.MODEL_URL} with input size {IMAGE_SIZE}')

# load test dataset

ds = tf.keras.preprocessing.image_dataset_from_directory(
  config.DIRPATH_DATASET,
  seed=123,
  image_size=IMAGE_SIZE,
  batch_size=config.DATASET_BATCH_SIZE)

class_names = ds.class_names
class_indices = range(len(class_names))
class_count = len(class_names)

ds_count = ds.cardinality().numpy()
test_count = int(ds_count * 0.2)
test_ds = ds.take(test_count)

print('datasets initialized')

# evaluate model on test images

model = keras.models.load_model(config.FILEPATH_SAVED_MODEL)

predicted_indices = []
actual_indices = []
num_batches = sum([1 for _ in test_ds])
i=1

for images, labels in test_ds:
    labels_list = list(labels.numpy())
    pred = model.predict(images)
    predicted_list = list(np.argmax(pred, axis=1))
    for predicted, actual in zip(predicted_list, labels_list):
        predicted_indices.append(predicted)
        actual_indices.append(actual)
    print(f'batch {i}/{num_batches}')
    i += 1

# count correct guesses for each class

classification_counts = [[0 for _ in class_names]
                         for _ in class_names]

correct_predictions = 0
total_predictions = 0
for predicted, actual in zip(actual_indices, predicted_indices):
    classification_counts[actual][predicted] += 1
    if predicted == actual:
        correct_predictions += 1
    total_predictions += 1

# calculate percentages for all classes

accuracies = []
class_index = 0
for class_counts in classification_counts:
    total_for_class = sum(class_counts)

    # calculate accuracy for class
    actual_score = class_counts[class_index]
    accuracy = format_percentage2(actual_score / total_for_class)

    # calculate top n guessed classes for actual class
    percentages = []
    for j in range(5):
        # pick class with highest classification count for the actual class
        class_index_highest_count = np.argmax(class_counts)
        highest_count = class_counts[class_index_highest_count]
        if highest_count < 0:
            break
        percentage = format_percentage2(highest_count / total_for_class)
        percentages.append({'class_index': class_index_highest_count,
                            'percentage': percentage})
        class_counts[class_index_highest_count] = -1

    # create string for top n guessed classes
    percentage_str = ''
    j = 0
    for percentage_entry in percentages:
        if j != 0:
            percentage_str += ', '
        class_name = class_names[percentage_entry['class_index']]
        percentage_str += f'{percentage_entry["percentage"]}% {class_name}'
        j += 1

    accuracies.append({
        'class_name': class_names[class_index],
        'accuracy': accuracy,
        'percentages': percentage_str})

    class_index += 1

# sort accuracies in DESC order

def compare(x1, x2):
    return x2["accuracy"] - x1["accuracy"]


accuracies = sorted(
    accuracies, key=functools.cmp_to_key(compare))

# write result to output file

print(f'accuracy: {format_percentage2(correct_predictions / total_predictions)}%')

f = open(config.FILEPATH_CLASS_ACCURACIES, "w")

for entry in accuracies:
    f.write(f'{entry["accuracy"]}%,{entry["class_name"]},"{entry["percentages"]}"\n')

f.close()

print('-------------- DONE --------------')
