from pathlib import Path
from PIL import ImageFile
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Path('/media/linus/ML/open_images/words')
image_directory = Path('/home/linus/Code/datasets/words')

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

# load training data
BATCH_SIZE = 32
data_dir = image_directory / 'training'

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = True
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        **datagen_kwargs)
else:
    train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=False, **dataflow_kwargs)

do_fine_tuning = False

# build model
print("Building model with", model_url)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(model_url, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'])
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size

# early stopping
earlystop_callback = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.0001,
    patience=3)

# checkpoints
checkpoint_dir = Path('./checkpoint')
checkpoint_path = checkpoint_dir / 'cp.ckpt'

if checkpoint_dir.exists():
    model.load_weights(checkpoint_path)
    print('loaded from checkpoint')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)
# calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes)


class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# train the model
hist = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=[earlystop_callback]
).history

model.save("out/saved_model")
