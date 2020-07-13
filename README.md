# Falco
Deep Learning on Cornell's NABirds Dataset

This repository contains code to accompany the article
[This Model is For the Birds](https://towardsdatascience.com/this-model-is-for-the-birds-6d55060d9074)

The article deals with experiments performed using classification models build with
Cornell's NABirds dataset.

The code consists of three parts:

* Creating the tfrecord files 

* Training the model.

* Model evaluation.

In the article, three sizes of [EfficientNet](https://github.com/qubvel/efficientnet) and the corresponding image resolutions are considered.
Any one of the eight EfficientNet sizes can be trained from this respository, and each size model
can be trained with any of the eight image resolutions.

I recommend running code in a clean environment. My own experiments were done in Google Colab;
if you don't require a TPU a virtual environment with TensorFlow should be sufficient. I would not
recommend trying to train or test without at least a GPU.

Samples of how to run the three components are below.

## Making records.

This code converts the images in NABIRDS_DIR to tfrecord shards containing up to FILE_SIZE records
each, writing progress to the console every 1000 records.

```bash
python make_tfrecord.py -l --dir ${NABIRDS_DIR} \
                        --size ${FILE_SIZE} \                    
```

## Training the Model

This code trains an EfficientNetB0 model using the default image size for EfficientNetB3 (higher
resoluton improves accuracy.) This uses the default learning rate of 1e-3, decayed 0.94 every 4
epochs, the default Adam optimizer, and the default pretrained ImageNet weights.

```bash
python train_model.py --dir ${NABIRDS_DIR}\
                      --enet 0 \
                      --res 3 \
                      --epoch 300
```

## Model Evaluation

This code evaluates the model trained above. There are four possible ways to choose testing data,
here I choose to use the bounding box crop to evaluate, but rescaled to match the training input
size.

```bash
python test_model.py --dir ${NABIRDS_DIR}\
                      --enet 0 \
                      --res 3 \
                      --crop
```