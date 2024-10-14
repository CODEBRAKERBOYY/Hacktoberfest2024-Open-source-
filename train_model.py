# File: train_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Define paths (Ensure these paths match your Google Drive setup)
image_path = '/content/drive/derive/nameofdataset/train/images'
mask_path = '/content/drive/Mderive/nameofdataset/train/mask'

# Define image size and batch size
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16

# Define a simple UNet model
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up1)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up2)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    model = Model(inputs, outputs)
    return model

# Create an instance of the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data generators for images and masks
image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
mask_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training image generator
train_image_generator = image_datagen.flow_from_directory(
    image_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,
    subset='training',
    seed=42
)

# Training mask generator
train_mask_generator = mask_datagen.flow_from_directory(
    mask_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,
    subset='training',
    color_mode='grayscale',
    seed=42
)

# Validation image generator
val_image_generator = image_datagen.flow_from_directory(
    image_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,
    subset='validation',
    seed=42
)

# Validation mask generator
val_mask_generator = mask_datagen.flow_from_directory(
    mask_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,
    subset='validation',
    color_mode='grayscale',
    seed=42
)

# Function to create tf.data.Dataset
def create_dataset(image_gen, mask_gen):
    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_generator(lambda: image_gen, output_signature=tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)),
        tf.data.Dataset.from_generator(lambda: mask_gen, output_signature=tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32))
    ))
    dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Create train and validation datasets
train_dataset = create_dataset(train_image_generator, train_mask_generator)
val_dataset = create_dataset(val_image_generator, val_mask_generator)

# Callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/unet_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_image_generator),
    epochs=10,
    validation_data=val_dataset,
    validation_steps=len(val_image_generator),
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model after training
model.save('/content/drive/MyDrive/final_unet_model.keras')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
