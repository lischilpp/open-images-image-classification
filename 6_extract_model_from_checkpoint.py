from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight

import config

# init variables

IMAGE_SIZE = (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)
print(f'Using {config.MODEL_URL} with input size {IMAGE_SIZE}')

class_paths = list(config.DIRPATH_DATASET.glob('*/'))
class_count = len(class_paths)


# build model

print("Building model with", config.MODEL_URL)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(config.MODEL_URL, trainable=config.DO_FINE_TUNING),
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

# load weights from checkpoint

checkpoint_dir = Path('./checkpoint')
checkpoint_path = checkpoint_dir / 'cp.ckpt'

if checkpoint_dir.exists():
    model.load_weights(checkpoint_path)
    print('loaded from checkpoint')
else:
    exit('checkpoint does not exist!')

# save model

model.save(config.FILEPATH_SAVED_MODEL)

print('-------------- DONE --------------')
