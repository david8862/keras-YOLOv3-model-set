#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""custom model callbacks."""
import os, sys, random, tempfile
import numpy as np
import glob
from tensorflow_model_optimization.sparsity import keras as sparsity
#from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from yolo5.model import get_yolo5_model
from yolo3.model import get_yolo3_model
from yolo2.model import get_yolo2_model
from eval import eval_AP


class DatasetShuffleCallBack(Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        np.random.shuffle(self.dataset)


class CheckpointCleanCallBack(Callback):
    def __init__(self, checkpoint_dir, max_val_keep=5, max_eval_keep=2):
        self.checkpoint_dir = checkpoint_dir
        self.max_val_keep = max_val_keep
        self.max_eval_keep = max_eval_keep

    def on_epoch_end(self, epoch, logs=None):

        # filter out eval checkpoints and val checkpoints
        all_checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ep*.h5')), reverse=False)
        eval_checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ep*-mAP*.h5')), reverse=False)
        val_checkpoints = sorted(list(set(all_checkpoints) - set(eval_checkpoints)), reverse=False)

        # keep latest val checkpoints
        for val_checkpoint in val_checkpoints[:-(self.max_val_keep)]:
            os.remove(val_checkpoint)

        # keep latest eval checkpoints
        for eval_checkpoint in eval_checkpoints[:-(self.max_eval_keep)]:
            os.remove(eval_checkpoint)


class EvalCallBack(Callback):
    def __init__(self, model_type, annotation_lines, anchors, class_names, model_input_shape, model_pruning, log_dir, eval_epoch_interval=10, save_eval_checkpoint=False, elim_grid_sense=False):
        self.model_type = model_type
        self.annotation_lines = annotation_lines
        self.anchors = anchors
        self.class_names = class_names
        self.model_input_shape = model_input_shape
        self.model_pruning = model_pruning
        self.log_dir = log_dir
        self.eval_epoch_interval = eval_epoch_interval
        self.save_eval_checkpoint = save_eval_checkpoint
        self.elim_grid_sense = elim_grid_sense
        self.best_mAP = 0.0
        self.eval_model = self.get_eval_model()

    def get_eval_model(self):
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        if self.model_type.startswith('scaled_yolo4_') or self.model_type.startswith('yolo5_'):
            # Scaled-YOLOv4 & YOLOv5 entrance
            eval_model, _ = get_yolo5_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_input_shape + (3,), model_pruning=self.model_pruning)
            self.v5_decode = True
        elif self.model_type.startswith('yolo3_') or self.model_type.startswith('yolo4_') or \
             self.model_type.startswith('tiny_yolo3_') or self.model_type.startswith('tiny_yolo4_'):
            # YOLOv3 & v4 entrance
            eval_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_input_shape + (3,), model_pruning=self.model_pruning)
            self.v5_decode = False
        elif self.model_type.startswith('yolo2_') or self.model_type.startswith('tiny_yolo2_'):
            # YOLOv2 entrance
            eval_model, _ = get_yolo2_model(self.model_type, num_anchors, num_classes, input_shape=self.model_input_shape + (3,), model_pruning=self.model_pruning)
            self.v5_decode = False
        else:
            raise ValueError('Unsupported model type')

        return eval_model


    def update_eval_model(self, train_model):
        # create a temp weights file to save training result
        tmp_weights_path = os.path.join(tempfile.gettempdir(), str(random.randint(10, 1000000)) + '.h5')
        train_model.save_weights(tmp_weights_path)

        # load the temp weights to eval model
        self.eval_model.load_weights(tmp_weights_path)
        os.remove(tmp_weights_path)

        if self.model_pruning:
            eval_model = sparsity.strip_pruning(self.eval_model)
        else:
            eval_model = self.eval_model

        return eval_model


    #def update_eval_model(self, model):
        ## We strip the extra layers in training model to get eval model
        #num_anchors = len(self.anchors)

        #if num_anchors == 9:
            ## YOLOv3 use 9 anchors and 3 prediction layers.
            ## Has 7 extra layers (including metrics) in training model
            #y1 = model.layers[-10].output
            #y2 = model.layers[-9].output
            #y3 = model.layers[-8].output

            #eval_model = Model(inputs=model.input[0], outputs=[y1,y2,y3])
        #elif num_anchors == 6:
            ## Tiny YOLOv3 use 6 anchors and 2 prediction layers.
            ## Has 6 extra layers in training model
            #y1 = model.layers[-8].output
            #y2 = model.layers[-7].output

            #eval_model = Model(inputs=model.input[0], outputs=[y1,y2])
        #elif num_anchors == 5:
            ## YOLOv2 use 5 anchors and 1 prediction layer.
            ## Has 6 extra layers in training model
            #eval_model = Model(inputs=model.input[0], outputs=model.layers[-7].output)
        #else:
            #raise ValueError('Invalid anchor set')

        #return eval_model


    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.eval_epoch_interval == 0:
            # Do eval every eval_epoch_interval epochs
            eval_model = self.update_eval_model(self.model)
            mAP = eval_AP(eval_model, 'H5', self.annotation_lines, self.anchors, self.class_names, self.model_input_shape, eval_type='VOC', iou_threshold=0.5, conf_threshold=0.001, elim_grid_sense=self.elim_grid_sense, v5_decode=self.v5_decode, save_result=False)
            if self.save_eval_checkpoint and mAP > self.best_mAP:
                # Save best mAP value and model checkpoint
                self.best_mAP = mAP
                self.model.save(os.path.join(self.log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-mAP{mAP:.3f}.h5'.format(epoch=(epoch+1), loss=logs.get('loss'), val_loss=logs.get('val_loss'), mAP=mAP)))
