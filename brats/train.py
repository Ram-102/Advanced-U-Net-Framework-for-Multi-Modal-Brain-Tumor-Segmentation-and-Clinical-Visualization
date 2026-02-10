import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping

from .config import (
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_DIR,
    CHECKPOINT_PATTERN,
    TRAIN_LOG_PATH,
)
from .data import make_generators
from .model import build_compiled_unet


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    training_generator, valid_generator, _, train_ids, val_ids, _ = make_generators()

    model = build_compiled_unet(learning_rate=LEARNING_RATE)

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATTERN,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        ),
        CSVLogger(TRAIN_LOG_PATH, separator=",", append=False),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        training_generator,
        epochs=EPOCHS,
        steps_per_epoch=len(train_ids),
        callbacks=callbacks,
        validation_data=valid_generator,
    )

    model.save("my_model.keras")


if __name__ == "__main__":
    main()


