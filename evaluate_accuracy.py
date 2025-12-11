import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL_PATH = "sign_model.h5"
TEST_DIR = "Data"
BATCH_SIZE = 32

# Load model
model = load_model(MODEL_PATH)

# Prepare data generator
datagen = ImageDataGenerator(rescale=1./255)
test_flow = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(300, 300),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# Ground truth
y_true = test_flow.classes
class_indices = test_flow.class_indices
class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

# Predictions
y_prob = model.predict(test_flow, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# Overall accuracy
acc = accuracy_score(y_true, y_pred)
print("Overall accuracy: {:.2%}".format(acc))

# Per-class accuracy
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
support = cm.sum(axis=1)
diag = np.diag(cm)
per_class_acc = diag / support

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    print(f"{name:>3s}: support={support[i]:4d}, accuracy={per_class_acc[i]:.3f}")