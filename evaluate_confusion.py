from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

IMG_SIZE = 300
BATCH = 32
DATA_DIR = "Data"    # ya alag test dir
MODEL_PATH = "sign_model.h5"

model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

y_true = val_gen.classes
class_indices = val_gen.class_indices
idx2label = {v: k for k, v in class_indices.items()}

pred_probs = model.predict(val_gen)
y_pred = np.argmax(pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Classes order:", [idx2label[i] for i in range(len(idx2label))])
print("Confusion matrix:\n", cm)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))]))
