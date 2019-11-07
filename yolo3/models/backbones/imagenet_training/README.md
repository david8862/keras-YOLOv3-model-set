# Training backbone classification network on imagenet

This is a small submodule to train some tf.keras classification network with Imagenet2012 dataset

### Dataset prepare
1. Download & extract Imagenet 2012 train/val/test dataset:

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

2. preprocess validation dataset to moves JPEG files into synset label dirs:

```
# cd imagenet_preprocess
# python preprocess_imagenet_validation_data.py ../data/ILSVRC2012_img_val imagenet_2012_validation_synset_labels.txt
```

The script and synset labels file are from [tensorflow inception model](https://github.com/tensorflow/models/tree/master/research/inception/inception/data)

### Train
1. [train_imagenet.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/yolo3/models/backbones/imagenet_training/train_imagenet.py)
```
# python train_imagenet.py -h
usage: train_imagenet.py [-h] [--model_type MODEL_TYPE] --train_data_path
                         TRAIN_DATA_PATH --val_data_path VAL_DATA_PATH
                         [--weights_path WEIGHTS_PATH]
                         [--batch_size BATCH_SIZE] [--optim_type OPTIM_TYPE]
                         [--learning_rate LEARNING_RATE]
                         [--init_epoch INIT_EPOCH] [--total_epoch TOTAL_EPOCH]
                         [--gpu_num GPU_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        backbone model type: shufflenet/shufflenet_v2/nanonet,
                        default=shufflenet_v2
  --train_data_path TRAIN_DATA_PATH
                        path to Imagenet train data
  --val_data_path VAL_DATA_PATH
                        path to Imagenet validation dataset
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --batch_size BATCH_SIZE
                        batch size for train, default=128
  --optim_type OPTIM_TYPE
                        optimizer type: sgd/rmsprop/adam, default=sgd
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.05
  --init_epoch INIT_EPOCH
                        Initial training epochs for fine tune training,
                        default=0
  --total_epoch TOTAL_EPOCH
                        Total training epochs, default=200
  --gpu_num GPU_NUM     Number of GPU to use, default=1
```

For example, following cmd will start training shufflenet_v2 with the Imagenet train/val data we prepared before:

```
# python train_imagenet.py --model_type=shufflenet_v2 --train_data_path=data/ILSVRC2012_img_train/ --val_data_path=data/ILSVRC2012_img_val/
```


Currently it support shufflenet/shufflenet_v2/nanonet which is implement under [backbones](https://github.com/david8862/keras-YOLOv3-model-set/tree/master/yolo3/models/backbones) with fixed hyperparam.

Checkpoints during training could be found at logs/. Choose a best one as result

MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).


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
```
