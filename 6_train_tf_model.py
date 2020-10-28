import os
from pathlib import Path
from PIL import ImageFile
import numpy as np


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Path('/Volumes/ML/open_images/animals')
img_dir = Path('/media/linus/ML/open_images/animals/training')

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
if len(tf.config.list_physical_devices('GPU')) >= 1:
    print("GPU is available")
else:
    print("GPU not available")

# model_url, pixels = ("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4", 299)
# model_url, pixels = ("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4", 224)
model_url, pixels = (
    "https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2", 380)

IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(model_url, IMAGE_SIZE))

BATCH_SIZE = 32

ds = tf.keras.preprocessing.image_dataset_from_directory(
    img_dir,
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

class_names = ds.class_names
class_count = len(class_names)

ds_count = ds.cardinality().numpy()
test_count = int(ds_count * 0.2)
val_count = int((ds_count - test_count) * 0.2)

train_val_ds = ds.skip(test_count)
train_ds = train_val_ds.skip(val_count)
val_ds = train_val_ds.take(val_count)

print('datasets initialized')

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(*IMAGE_SIZE,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

do_fine_tuning = False
print("Building model with", model_url)
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(model_url, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(class_count,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

earlystop_callback = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.0001,
    patience=3)

checkpoint_dir = Path('./checkpoint')
checkpoint_path = checkpoint_dir / 'cp.ckpt'

if checkpoint_dir.exists():
    model.load_weights(checkpoint_path)
    print('loaded from checkpoint')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)

# manually calculate; equivalent to train_generator.classes
class_indices = []
i = 0
for class_name in sorted(os.listdir(img_dir)):
    image_count = len([f for f in os.listdir(img_dir / class_name)])
    class_indices += [i] * image_count
    i += 1


# calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_indices),
    y=class_indices)


class_weights = {i: class_weights[i] for i in range(len(class_weights))}


hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights,
    callbacks=[earlystop_callback, cp_callback]
).history

model.save("out/saved_model")
