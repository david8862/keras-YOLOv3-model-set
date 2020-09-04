#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    return BatchNorm(*args, **kwargs)

