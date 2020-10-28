from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight

img_dir = Path('/media/linus/ML/open_images/animals/training')

model_url, pixels = (
    "https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2", 380)

IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(model_url, IMAGE_SIZE))

BATCH_SIZE = 32


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    img_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

class_names = val_ds.class_names
class_count = len(class_names)

do_fine_tuning = False
print("Building model with", model_url)
model = tf.keras.Sequential([
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

checkpoint_dir = Path('./checkpoint')
checkpoint_path = checkpoint_dir / 'cp.ckpt'

if checkpoint_dir.exists():
    model.load_weights(checkpoint_path)
    print('loaded from checkpoint')
else:
    exit('checkpoint does not exist!')

model.save("out/saved_model")
