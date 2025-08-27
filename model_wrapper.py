# model_wrapper.py
# Contains a helper class that (a) loads the saved model only once,
# (b) offers a predict() method that takes an image (PIL or ndarray/path).

import numpy as np
from pathlib import Path
from typing import Union
from PIL import Image

import tensorflow as tf

CLASS_IDX = {0: "normal", 1: "corroded"}

class AssetClassifier:
    def __init__(self, model_path: Union[str, Path],
                 img_size: int | None = None):
        self.model = tf.keras.models.load_model(model_path)
        # Infer expected input size from the model if not provided
        inferred_size = None
        try:
            shape = getattr(self.model, "input_shape", None)
            # shape like: (None, H, W, C)
            if isinstance(shape, tuple) and len(shape) >= 3:
                inferred_size = shape[1]
        except Exception:
            inferred_size = None
        self.img_size = img_size or inferred_size or 224

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.resize((self.img_size, self.img_size))
        arr = np.asarray(img).astype("float32") / 255.0
        if arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[..., :3]
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image: Union[str, Path, Image.Image]):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        arr = self._preprocess(image)
        prob_corroded = float(self.model.predict(arr)[0, 0])
        prob_normal   = 1.0 - prob_corroded
        label         = CLASS_IDX[int(prob_corroded >= 0.5)]
        return {
            "label": label,
            "probabilities": {
                "corroded": prob_corroded,
                "normal":   prob_normal
            }
        }