from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

from .config import IMG_SIZE, PRETRAINED_MODEL_PATH, USE_PRETRAINED
from .metrics import compile_metrics


def build_unet(inputs, ker_init="he_normal", dropout=0.2):
    conv1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv1)

    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool)
    conv = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv3)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(drop5)
    )
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = Concatenate(axis=3)([conv, up9])
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv9)

    up = Conv2D(32, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(conv9)
    )
    merge = Concatenate(axis=3)([conv1, up])
    conv = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge)
    conv = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv)

    conv10 = Conv2D(4, (1, 1), activation="softmax")(conv)

    return Model(inputs=inputs, outputs=conv10)


def build_compiled_unet(learning_rate: float = 1e-3):
    inputs = Input((IMG_SIZE, IMG_SIZE, 2))
    model = build_unet(inputs, "he_normal", 0.2)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=compile_metrics(),
    )
    return model


def load_pretrained_model():
    """Load pre-trained model from Colab if available"""
    if USE_PRETRAINED and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading pre-trained model from {PRETRAINED_MODEL_PATH}")
        import tensorflow as tf
        from .metrics import dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing
        
        model = tf.keras.models.load_model(
            PRETRAINED_MODEL_PATH,
            custom_objects={
                "accuracy": tf.keras.metrics.MeanIoU(num_classes=4),
                "dice_coef": dice_coef,
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "dice_coef_necrotic": dice_coef_necrotic,
                "dice_coef_edema": dice_coef_edema,
                "dice_coef_enhancing": dice_coef_enhancing
            },
            compile=False
        )
        # Recompile with current metrics
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=1e-3),
            metrics=compile_metrics(),
        )
        return model
    else:
        print("No pre-trained model found, building new model...")
        return build_compiled_unet()


