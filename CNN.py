
# import base model - MobileNetV2
from tensorflow.keras.applications import MobileNetV2

# imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

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
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(25, activation='softmax')  # ← change this number to your actual class count
])

print("********************************************************************************")
print("custom model created successfully.")
print("********************************************************************************")



# compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("********************************************************************************")
print("custom model compiled successfully.")
print("********************************************************************************")
