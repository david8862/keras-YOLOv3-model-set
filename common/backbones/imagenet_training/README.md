# Training backbone classification network on imagenet

This is a small submodule to train some tf.keras classification network with Imagenet2012 dataset, and export headless model for object detection backbone

### Dataset prepare
1. Download & extract Imagenet 2012 train/val/test dataset (Update@2019.11.22: free download link is invalid now, may need to get the dataset with official website account):

```
# mkdir data
# cd data
# wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar (138GB)
# wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar (6.3GB)
# wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar (12.7GB)
# mkdir ILSVRC2012_img_train ILSVRC2012_img_val ILSVRC2012_img_test
# tar xvf ILSVRC2012_img_train.tar -C ILSVRC2012_img_train
# tar xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val
# tar xvf ILSVRC2012_img_test.tar -C ILSVRC2012_img_test
```

It may cost some time since the dataset is quite large

2. preprocess train & validation dataset to moves JPEG files into synset label dirs:

```
# cd imagenet_preprocess
# python preprocess_imagenet_train_data.py --train_data_path=../data/ILSVRC2012_img_train
# python preprocess_imagenet_validation_data.py ../data/ILSVRC2012_img_val imagenet_2012_validation_synset_labels.txt
```

The validation preprocess script and synset labels file are from [tensorflow inception model](https://github.com/tensorflow/models/tree/master/research/inception/inception/data)

### Train
1. [train_imagenet.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/common/backbones/imagenet_training/train_imagenet.py)
```
# python train_imagenet.py -h
usage: train_imagenet.py [-h] --model_type
                         {shufflenet,shufflenet_v2,nanonet,darknet53,cspdarknet53,mobilevit_s,mobilevit_xs,mobilevit_xxs}
                         [--weights_path WEIGHTS_PATH]
                         [--train_data_path TRAIN_DATA_PATH]
                         [--val_data_path VAL_DATA_PATH]
                         [--batch_size BATCH_SIZE]
                         [--optimizer {adam,rmsprop,sgd}]
                         [--learning_rate LEARNING_RATE]
                         [--decay_type {None,cosine,exponential,polynomial,piecewise_constant}]
                         [--label_smoothing LABEL_SMOOTHING]
                         [--init_epoch INIT_EPOCH] [--total_epoch TOTAL_EPOCH]
                         [--gpu_num GPU_NUM] [--evaluate]
                         [--verify_with_image] [--dump_headless]
                         [--output_model_file OUTPUT_MODEL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --model_type {shufflenet,shufflenet_v2,nanonet,darknet53,cspdarknet53,mobilevit_s,mobilevit_xs,mobilevit_xxs}
                        backbone model type
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --train_data_path TRAIN_DATA_PATH
                        path to Imagenet train data
  --val_data_path VAL_DATA_PATH
                        path to Imagenet validation dataset
  --batch_size BATCH_SIZE
                        batch size for train, default=128
  --optimizer {adam,rmsprop,sgd}
                        optimizer for training (adam/rmsprop/sgd), default=sgd
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.01
  --decay_type {None,cosine,exponential,polynomial,piecewise_constant}
                        Learning rate decay type, default=None
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing factor (between 0 and 1) for
                        classification loss, default=0
  --init_epoch INIT_EPOCH
                        Initial training epochs for fine tune training,
                        default=0
  --total_epoch TOTAL_EPOCH
                        Total training epochs, default=200
  --gpu_num GPU_NUM     Number of GPU to use, default=1
  --evaluate            Evaluate a trained model with validation dataset
  --verify_with_image   Verify trained model with image
  --dump_headless       Dump out classification model to headless backbone
                        model
  --output_model_file OUTPUT_MODEL_FILE
                        output headless backbone model file
```

For example, following cmd will start training shufflenet_v2 with the Imagenet train/val data we prepared before:

```
# python train_imagenet.py --model_type=shufflenet_v2 --train_data_path=data/ILSVRC2012_img_train/ --val_data_path=data/ILSVRC2012_img_val/ --batch_size=128 --optimizer=sgd --learning_rate=0.01 --decay_type=cosine --label_smoothing=0.1
```

Currently it support shufflenet/shufflenet_v2/nanonet/darknet53/cspdarknet53/mobilevit_s/mobilevit_xs/mobilevit_xxs which is implement under [backbones](https://github.com/david8862/keras-YOLOv3-model-set/tree/master/common/backbones) with fixed hyperparam.

Checkpoints during training could be found at logs/. Choose a best one as result

MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).


### Evaluate trained model

```
# python train_imagenet.py --model_type=shufflenet_v2 --weights_path=logs/<checkpoint file> --val_data_path=data/ILSVRC2012_img_val/ --batch_size=64 --evaluate
```

### Verify trained model

```
# python train_imagenet.py --model_type=shufflenet_v2 --weights_path=logs/<checkpoint file> --verify_with_image
```

### Export headless backbone model

```
# python train_imagenet.py --model_type=shufflenet_v2 --weights_path=logs/<checkpoint file> --dump_headless --output_model_file=shufflenet_v2_headless.h5
```

### TODO
- [ ] use [tensorflow_datasets](https://github.com/tensorflow/datasets) for image data feeding, to improve training efficiency on large dataset like Imagenet 2012


### Environment
- Ubuntu 16.04/18.04
- Python 3.6.7
- tensorflow 1.14.0
- tf.keras 2.2.4-tf


# Citation
```
@article{YOLO Nano,
     title={YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection},
     author={Alexander Wong, Mahmoud Famuori, Mohammad Javad Shafiee, Francis Li, Brendan Chwyl, and Jonathan Chung},
     journal = {arXiv},
     year={2019}
}
@article{ShuffleNet,
     title={ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices},
     author={Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun},
     journal = {arXiv},
     year={2017}
}
@article{ShuffleNet V2,
     title={ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design},
     author={Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun},
     journal = {arXiv},
     year={2018}
}
@article{MobileViT,
     title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
     author={Sachin Mehta, Mohammad Rastegari},
     journal = {arXiv},
     year={2021}
}
```
