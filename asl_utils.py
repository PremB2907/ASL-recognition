import os
import sys
import glob

# Dynamically bind NVIDIA CUDA & cuDNN pip package libraries to LD_LIBRARY_PATH and XLA_FLAGS
try:
    venv_site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    nvidia_dirs = glob.glob(os.path.join(venv_site_packages, "nvidia", "*", "lib"))
    if nvidia_dirs:
        curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(nvidia_dirs + ([curr_ld] if curr_ld else []))
        os.environ["LD_LIBRARY_PATH"] = new_ld

    # Point XLA compiler to libdevice.10.bc for NVIDIA GPU JIT kernel execution
    nvcc_dir = os.path.join(venv_site_packages, "nvidia", "cuda_nvcc")
    if os.path.exists(nvcc_dir):
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={nvcc_dir}"
except Exception:
    pass

import json
import math
import cv2
import numpy as np

def preprocess_hand_crop(img, hand_bbox, img_size=300, offset=20):
    """
    Safely crop hand region from image with bounding box, pad with white background,
    and resize maintaining aspect ratio.
    
    Args:
        img: Input BGR image (numpy array)
        hand_bbox: Hand bounding box, either dict with key "bbox": [x, y, w, h] or tuple/list (x, y, w, h)
        img_size: Target square canvas dimension (default: 300)
        offset: Padding margin around bounding box (default: 20)
        
    Returns:
        Processed uint8 (img_size, img_size, 3) image centered on white background, or None if crop invalid.
    """
    if img is None or img.size == 0:
        return None

    if isinstance(hand_bbox, dict):
        x, y, w, h = hand_bbox["bbox"]
    elif isinstance(hand_bbox, (list, tuple)) and len(hand_bbox) >= 4:
        x, y, w, h = hand_bbox[:4]
    else:
        return None

    if w <= 0 or h <= 0:
        return None

    img_h, img_w = img.shape[:2]

    # Clamp bounding box coordinates safely within image dimensions
    y1 = max(0, int(y - offset))
    y2 = min(img_h, int(y + h + offset))
    x1 = max(0, int(x - offset))
    x2 = min(img_w, int(x + w + offset))

    if y2 <= y1 or x2 <= x1:
        return None

    img_crop = img[y1:y2, x1:x2]
    if img_crop.size == 0:
        return None

    crop_h, crop_w = img_crop.shape[:2]
    img_white = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    aspect = crop_h / crop_w if crop_w > 0 else 1.0

    try:
        if aspect > 1:  # Height > Width
            scale = img_size / crop_h
            w_cal = min(img_size, max(1, math.ceil(scale * crop_w)))
            img_resize = cv2.resize(img_crop, (w_cal, img_size))
            gap = (img_size - w_cal) // 2
            img_white[:, gap:gap + w_cal] = img_resize
        else:  # Width >= Height
            scale = img_size / crop_w if crop_w > 0 else 1.0
            h_cal = min(img_size, max(1, math.ceil(scale * crop_h)))
            img_resize = cv2.resize(img_crop, (img_size, h_cal))
            gap = (img_size - h_cal) // 2
            img_white[gap:gap + h_cal, :] = img_resize
    except Exception as err:
        print(f"[asl_utils] Preprocessing error: {err}")
        return None

    return img_white


def load_label_mapping(data_dir="Data", labels_path="labels.json"):
    """
    Load or generate class label mappings (index -> letter name).
    First checks `labels_path`. If missing, scans directory names in `data_dir` in alphabetical order,
    matching Keras ImageDataGenerator default ordering.
    
    Returns:
        idx_to_label (dict): {0: 'A', 1: 'B', ...}
        label_to_idx (dict): {'A': 0, 'B': 1, ...}
    """
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                idx_to_label = {int(k): str(v) for k, v in raw_data.items()}
                label_to_idx = {v: k for k, v in idx_to_label.items()}
                return idx_to_label, label_to_idx
        except Exception as err:
            print(f"[asl_utils] Failed to read {labels_path}: {err}. Falling back to directory scan.")

    # Fallback: scan data directory
    if os.path.exists(data_dir):
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        idx_to_label = {i: name for i, name in enumerate(class_names)}
        label_to_idx = {name: i for i, name in enumerate(class_names)}

        # Save generated labels for future instant loading
        save_label_mapping(idx_to_label, labels_path)
        return idx_to_label, label_to_idx

    # Return default fallback alphabet if data_dir is not found
    default_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    idx_to_label = {i: name for i, name in enumerate(default_classes)}
    label_to_idx = {name: i for i, name in enumerate(default_classes)}
    return idx_to_label, label_to_idx


def save_label_mapping(idx_to_label, labels_path="labels.json"):
    """Save label mapping dictionary to a JSON file."""
    try:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in idx_to_label.items()}, f, indent=2)
        print(f"[asl_utils] Saved label mapping to {labels_path}")
    except Exception as err:
        print(f"[asl_utils] Failed to save labels to {labels_path}: {err}")


class FastPredictor:
    """
    Optimized inference wrapper for TensorFlow/Keras models to minimize overhead
    in real-time loops.
    """
    def __init__(self, model_path="sign_model.h5", labels_path="labels.json", data_dir="Data"):
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"[asl_utils] Loading Keras model from {model_path}...")
        self.model = load_model(model_path)
        self.idx_to_label, self.label_to_idx = load_label_mapping(data_dir, labels_path)

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[asl_utils] GPU Acceleration ACTIVE: {gpus}")
        else:
            print("[asl_utils] Running on CPU.")

        # Create tf.function for fast forward pass execution without graph overhead
        @tf.function(experimental_relax_shapes=True)
        def _predict_fn(x):
            return self.model(x, training=False)

        self._predict_fn = _predict_fn

        # Warm up model execution
        dummy_input = tf.zeros((1, 300, 300, 3), dtype=tf.float32)
        _ = self._predict_fn(dummy_input)
        print("[asl_utils] Model loaded and warmed up successfully.")

    def predict(self, img_white):
        """
        Run forward pass on a 300x300 uint8 BGR hand crop image.
        
        Returns:
            predicted_letter (str)
            confidence (float)
            margin (float) - top1 vs top2 probability gap
            probs (np.ndarray) - softmax output
        """
        if img_white is None:
            return None, 0.0, 0.0, None

        # Convert OpenCV BGR to RGB to match ImageDataGenerator training color space
        img_rgb = cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] float32 array
        norm_img = (img_rgb.astype(np.float32) / 255.0)[np.newaxis, ...]
        
        probs_tensor = self._predict_fn(norm_img)
        probs = probs_tensor.numpy()[0]

        top_indices = np.argsort(probs)[::-1]
        top1_idx = top_indices[0]
        top2_idx = top_indices[1] if len(top_indices) > 1 else top1_idx

        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        margin = top1_prob - top2_prob

        letter = self.idx_to_label.get(top1_idx, f"Class_{top1_idx}")
        return letter, top1_prob, margin, probs
