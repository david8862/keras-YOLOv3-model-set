#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob
from tensorflow.keras.callbacks import Callback


class CheckpointCleanCallBack(Callback):
    def __init__(self, checkpoint_dir, max_val_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_val_keep = max_val_keep

    def on_epoch_end(self, epoch, logs=None):
        # filter out val checkpoints
        val_checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ep*.h5')))

        # keep latest val checkpoints
        for val_checkpoint in val_checkpoints[:-(self.max_val_keep)]:
            os.remove(val_checkpoint)
