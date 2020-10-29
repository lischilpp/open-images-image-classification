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
import config


# enable to prevent PIL warning
ImageFile.LOAD_TRUNCATED_IMAGES = True

# debug information

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
if len(tf.config.list_physical_devices('GPU')) >= 1:
    print("GPU is available")
else:
    print("GPU not available")

# load dataset

IMAGE_SIZE = (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)
print(f'Using {config.MODEL_URL} with input size {IMAGE_SIZE}')

ds = tf.keras.preprocessing.image_dataset_from_directory(
    config.DIRPATH_DATASET,
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=config.DATASET_BATCH_SIZE)

# split dataset into train, validation and test

class_names = ds.class_names
class_count = len(class_names)

ds_count = ds.cardinality().numpy()
test_count = int(ds_count * config.TEST_DATA_PERCENTAGE)
val_count = int((ds_count - test_count) * config.VALIDATION_DATA_PERCENTAGE)

train_val_ds = ds.skip(test_count)
train_ds = train_val_ds.skip(val_count)
val_ds = train_val_ds.take(val_count)

print('datasets initialized')

# enable prefetching

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# build model

print("Building model with", config.MODEL_URL)
model_layers = [
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(config.MODEL_URL, trainable=config.DO_FINE_TUNING),
    tf.keras.layers.Dropout(rate=config.DROPOUT_RATE),
    tf.keras.layers.Dense(class_count,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
]
if config.DO_DATA_AUGMENTATION:
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                        input_shape=(*IMAGE_SIZE,
                                                                    3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    # add data augmentation as first layer
    model_layers.insert(0, data_augmentation)
model = tf.keras.Sequential(model_layers)

model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# checkpoint and early stopping

callbacks = []

if config.ENABLE_CHECKPOINTS:
    # load from checkpoint if it exists
    checkpoint_dir = Path('./checkpoint')
    checkpoint_path = checkpoint_dir / 'cp.ckpt'

    if checkpoint_dir.exists():
        model.load_weights(checkpoint_path)
        print('loaded from checkpoint')

    # add checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)
    callbacks.append(cp_callback)

if config.ENABLE_EARLY_STOPPING:
    earlystop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0001,
        patience=3)

    callbacks.append(earlystop_callback)


# calculate class weights

class_weights = None

if config.ENABLE_AUTOMATIC_CLASS_WEIGHTS:
    class_indices = []
    i = 0
    for class_name in sorted(os.listdir(config.DIRPATH_DATASET)):
        image_count = len([f for f in os.listdir(config.DIRPATH_DATASET / class_name)])
        class_indices += [i] * image_count
        i += 1

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_indices),
        y=class_indices)

    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# train model on dataset

hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.TRAINING_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
).history

model.save("out/saved_model")

print("-------------- DONE --------------")
