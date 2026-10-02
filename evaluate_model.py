import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from asl_utils import load_label_mapping

MODEL_PATH = "sign_model.h5"
LABELS_PATH = "labels.json"
DATA_DIR = "Data"
IMG_SIZE = 300
BATCH_SIZE = 32

def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train model first.")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found.")

    print(f"[evaluate_model] Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    idx_to_label, label_to_idx = load_label_mapping(DATA_DIR, LABELS_PATH)
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]

    print(f"[evaluate_model] Loading validation dataset from {DATA_DIR}...")
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    val_flow = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    y_true = val_flow.classes
    print(f"[evaluate_model] Evaluating {len(y_true)} validation samples...")

    pred_probs = model.predict(val_flow, verbose=1)
    y_pred = np.argmax(pred_probs, axis=1)

    # Metrics
    overall_acc = accuracy_score(y_true, y_pred)
    top5_acc = tf.keras.metrics.top_k_categorical_accuracy(
        tf.keras.utils.to_categorical(y_true, num_classes=len(class_names)),
        pred_probs,
        k=min(5, len(class_names))
    ).numpy().mean()

    print("\n" + "=" * 60)
    print(f" 📊 MODEL EVALUATION SUMMARY")
    print(f" Overall Top-1 Accuracy: {overall_acc * 100:.2f}%")
    print(f" Overall Top-5 Accuracy: {top5_acc * 100:.2f}%")
    print("=" * 60 + "\n")

    print("--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)

    # Save Confusion Matrix visual plot if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('ASL Model Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("\n[evaluate_model] Saved confusion matrix heatmap to 'confusion_matrix.png'.")
    except Exception as err:
        print(f"[evaluate_model] Could not save heatmap plot: {err}")

if __name__ == "__main__":
    evaluate()
