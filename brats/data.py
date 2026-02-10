import os
import glob
import random
import numpy as np
import cv2
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

from .config import (
    TRAIN_DATASET_PATH,
    APPLY_355_RENAME_FIX,
    IMG_SIZE,
    VOLUME_SLICES,
    VOLUME_START_AT,
    NUM_CHANNELS,
    RANDOM_SEED,
)


def maybe_fix_355_filename():
    if not APPLY_355_RENAME_FIX:
        return
    old_name = os.path.join(TRAIN_DATASET_PATH, "BraTS20_Training_355", "W39_1998.09.19_Segm.nii")
    new_name = os.path.join(TRAIN_DATASET_PATH, "BraTS20_Training_355", "BraTS20_Training_355_seg.nii")
    if os.path.exists(old_name) and not os.path.exists(new_name):
        try:
            os.rename(old_name, new_name)
            print("Renamed misnamed segmentation for BraTS20_Training_355")
        except OSError as e:
            print(f"Could not rename 355 seg file: {e}")


def list_case_dirs(root: str):
    return [f.path for f in os.scandir(root) if f.is_dir()]


def path_list_into_ids(dir_list):
    ids = []
    for p in dir_list:
        ids.append(p[p.rfind('/') + 1 :])
    return ids


def split_ids(all_ids, val_size: float = 0.2, test_size: float = 0.15):
    random.seed(RANDOM_SEED)
    train_and_test_ids, val_ids = train_test_split(all_ids, test_size=val_size, random_state=RANDOM_SEED)
    train_ids, test_ids = train_test_split(train_and_test_ids, test_size=test_size, random_state=RANDOM_SEED)
    return train_ids, val_ids, test_ids


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_ids, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=NUM_CHANNELS, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_ids = [self.list_ids[k] for k in indexes]
        X, Y = self.__data_generation(batch_ids)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240), dtype=np.float32)

        scaler = MinMaxScaler()

        for c, case_id in enumerate(batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, case_id)

            flair = nib.load(os.path.join(case_path, f"{case_id}_flair.nii")).get_fdata()
            t1ce = nib.load(os.path.join(case_path, f"{case_id}_t1ce.nii")).get_fdata()
            seg = nib.load(os.path.join(case_path, f"{case_id}_seg.nii")).get_fdata()

            # MinMax per volume (channel-wise)
            flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)
            t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES * c, :, :, 0] = cv2.resize(
                    flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
                )
                X[j + VOLUME_SLICES * c, :, :, 1] = cv2.resize(
                    t1ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
                )
                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START_AT]

        y[y == 4] = 3
        mask = tf.one_hot(tf.cast(y, tf.int32), 4)
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
        X = X / np.maximum(np.max(X), 1e-6)
        return X, Y


def make_generators():
    maybe_fix_355_filename()
    all_dirs = list_case_dirs(TRAIN_DATASET_PATH)
    all_ids = path_list_into_ids(all_dirs)
    train_ids, val_ids, test_ids = split_ids(all_ids)

    training_generator = DataGenerator(train_ids)
    valid_generator = DataGenerator(val_ids)
    test_generator = DataGenerator(test_ids)

    return training_generator, valid_generator, test_generator, train_ids, val_ids, test_ids


