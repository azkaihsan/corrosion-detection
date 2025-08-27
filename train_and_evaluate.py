"""
train_and_evaluate.py
Train a CNN (transfer-learning on top of a pretrained backbone)
to classify images as either “normal” or “corroded”.
"""

import os, json, argparse, time
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0   # light, modern backbone
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)

# ------------------------------------------------------------------ #
# 1. Parse CLI arguments (epochs, batch_size, etc.)                  #
# ------------------------------------------------------------------ #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset", type=str,
                    help="Root directory that contains train/ and test/")
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=10,  type=int)
    ap.add_argument("--lr", default=1e-4,   type=float)
    ap.add_argument("--model_out", default="model_asset_classifier.h5")
    return ap.parse_args()

# ------------------------------------------------------------------ #
# 2. Prepare the data loaders                                        #
# ------------------------------------------------------------------ #
def build_generators(root, img_size, batch_size):
    train_dir = os.path.join(root, "train")
    test_dir  = os.path.join(root, "test")

    # Basic augmentation for training; only rescaling for test.
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    test_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",   # normal = 0, corroded = 1
        color_mode="rgb"       # ensure RGB format
    )
    test_gen = test_aug.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        color_mode="rgb",      # ensure RGB format
        shuffle=False          # keep order to later map predictions
    )
    return train_gen, test_gen

# ------------------------------------------------------------------ #
# 3. Build the model (transfer learning)                             #
# ------------------------------------------------------------------ #
def build_model(img_size, lr):
    # Build EfficientNetB0 without pretrained weights to avoid shape mismatch
    base = EfficientNetB0(weights=None,
                          include_top=False,
                          input_shape=(img_size, img_size, 3))
    base.trainable = True  # Allow training from scratch

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ------------------------------------------------------------------ #
# 4. Training routine                                                #
# ------------------------------------------------------------------ #
def train(model, train_gen, test_gen, epochs, model_out):
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(model_out,
                        monitor="val_accuracy",
                        save_best_only=True,
                        verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history

# ------------------------------------------------------------------ #
# 5. Evaluation + metrics                                            #
# ------------------------------------------------------------------ #
def evaluate(model, test_gen):
    test_gen.reset()
    probs = model.predict(test_gen, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    y_true = test_gen.classes

    metrics = {
        "accuracy":  accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall":    recall_score(y_true, preds),
        "f1":        f1_score(y_true, preds),
        # For ROC-AUC we need positive & negative samples in test set
        "roc_auc":   roc_auc_score(y_true, probs)
    }
    return metrics, probs, y_true

# ------------------------------------------------------------------ #
# 6. Main                                                            #
# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    train_gen, test_gen = build_generators(args.data_dir,
                                           args.img_size,
                                           args.batch_size)
    model = build_model(args.img_size, args.lr)
    print(model.summary())

    train(model, train_gen, test_gen,
          args.epochs, args.model_out)

    # Reload the best checkpoint just saved
    best_model = tf.keras.models.load_model(args.model_out)
    metrics, probs, y_true = evaluate(best_model, test_gen)

    print("\nEvaluation metrics on the test set:")
    for k,v in metrics.items():
        print(f"{k:9s}: {v:.4f}")

    # Optional: save metrics to JSON for later inspection
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()