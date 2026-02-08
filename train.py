import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. SETUP PATHS
data_path = 'data/train'
model_dir = 'models'
model_path = os.path.join(model_dir, 'deepfake_detector.h5')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 2. DATA PREPARATION (Alphabetical order: fake=0, real=1)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=10,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# 3. BUILD THE AI (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Use pre-trained "eyes"

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid') # 0 to 1 probability
])

# 4. TRAIN THE AI
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("AI is learning... please wait.")
model.fit(train_gen, epochs=5) # Increased to 5 for better accuracy

# 5. SAVE THE BRAIN
model.save(model_path)
print(f"Success! Model saved at: {model_path}")
print(f"Class Indices: {train_gen.class_indices}") # Should show {'fake': 0, 'real': 1}