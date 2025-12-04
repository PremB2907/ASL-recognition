from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os

imgSize = 300
batch = 32
base_epochs = 5       # frozen base
finetune_epochs = 10  # unfrozen last layers

path = "Data"
class_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
num_classes = len(class_names)
print("Classes:", class_names)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    path,
    target_size=(imgSize, imgSize),
    batch_size=batch,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    path,
    target_size=(imgSize, imgSize),
    batch_size=batch,
    class_mode='categorical',
    subset='validation'
)

# 1) Base model
base = MobileNet(weights='imagenet', include_top=False, input_shape=(imgSize, imgSize, 3))
base.trainable = False   # start frozen

model = Sequential([
    base,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ModelCheckpoint("sign_model.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-5)
]

print("\n--- Stage 1: Train top layers (base frozen) ---\n")
history1 = model.fit(
    train,
    validation_data=val,
    epochs=base_epochs,
    callbacks=callbacks
)

# 2) Fine-tune last few layers of MobileNet
print("\n--- Stage 2: Fine-tune last MobileNet layers ---\n")
# Unfreeze last N layers
base.trainable = True
fine_tune_at = len(base.layers) - 30   # last ~30 layers trainable

for i, layer in enumerate(base.layers):
    layer.trainable = i >= fine_tune_at

model.compile(
    optimizer=Adam(learning_rate=1e-4),  # smaller LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train,
    validation_data=val,
    epochs=finetune_epochs,
    callbacks=callbacks
)

print("\nTraining completed. Best model saved as sign_model.h5")
