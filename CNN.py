"""
Layer Count: 159 (~155 ImageNet, ~4 custom)

Freezing pretrained layers for now

Need to add image evaluation code
"""

class_count_constant = 25

# import base model - MobileNetV2
from tensorflow.keras.applications import MobileNetV2

# imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# load base model into object
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # must match your image preprocessing
    # exclude ImageNet’s head
    include_top=False,         
    weights="imagenet"         # load weights trained on ImageNet
)

# freeze pretrained layers
base_model.trainable = False

print("********************************************************************************")
print("ImageNet model loaded successfully.")
print("********************************************************************************")

# my model
model = Sequential([
    base_model,
    # compressees into single vector
    GlobalAveragePooling2D(),
    # fully connected layer with 64 neurons using ReLU activation function
    Dense(64, activation='relu'),
    # dropout layer with 50% dropout rate to prevent overfitting - relying to heavily on one feature
    Dropout(0.5),
    # constant neurons, 1 per class, 1 for each class - change as needed - softmax activation function for multiclass classification
    Dense(class_count_constant, activation='softmax')  # ← change this number to your actual class count
])

print("********************************************************************************")
print("custom model created successfully.")
print("********************************************************************************")


# compile model
model.compile(
    # 'adaptive moment estimation' - Adam optimizer - consider changing
    optimizer='adam',
    # loss function for multiclass classification, fits with softmax activation function
    loss='categorical_crossentropy',
    # track and display accuracy during training
    metrics=['accuracy']
)

print("********************************************************************************")
print("custom model compiled successfully.")
print("********************************************************************************")


# data generator
datagen = ImageDataGenerator(
    # normalize pixel values to [0, 1] range from [0, 255] range 
    rescale=1./255,
    # randomly rotate images in the range (degrees, 0 to 180) 
    rotation_range=20,
    # random zoom
    zoom_range=0.2,
    # random flip
    horizontal_flip=True,
    # reserve 20% for validation
    validation_split=0.2
)

# load training dataset
train_generator = datagen.flow_from_directory(
    # path to training data
    r'C:\Users\Dawson\Documents\GitHub\CNN-Draft\pest\train',
    # size of images, must match input shape of model - consider a constant               
    target_size=(128, 128),
    # batch size tells generator how many images to return at a time        
    batch_size=32,
    # need for multiclass classification softmax activation function
    class_mode='categorical',
    # 80% of data for training, 20% for validation
    subset='training'
)

# split into validation set
val_generator = datagen.flow_from_directory(
    # path to training data
    r'C:\Users\Dawson\Documents\GitHub\CNN-Draft\pest\train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    # 20% of data for validation, 80% for training
    subset='validation'
)

print("********************************************************************************")
print("Training and validation data loaded.")
print("********************************************************************************")

print("********************************************************************************")
print("Beginning Training")
print("********************************************************************************")

# begin training
model.fit(
    # training data
    train_generator,
    # monitor overfitting
    validation_data=val_generator,
    # how many times to go through the entire training dataset
    epochs=10 # tweak this number as needed
)

print("********************************************************************************")
print("Training complete.")
print("********************************************************************************")   

print("********************************************************************************")
print("Saving model...")
print("********************************************************************************")

# save model
model.save("insect_model.h5")
print("Model saved successfully.")

# print class index
print("Class indices:")
print(train_generator.class_indices)

# note to self - add model evaluation code here

