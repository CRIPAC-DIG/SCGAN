# SCGAN
code for "Semi-supervised Compatibility Learning Across Categories for Clothing Matching" in ICME2019

## Paper data and code

This is the code for the ICME 2019 Paper: [Semi-supervised Compatibility Learning Across Categories for Clothing Matching](https://arxiv.org/pdf/1907.13304.pdf). We have implemented our methods in **Tensorflow**.

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `data/`:

- Taobao: <https://tianchi.aliyun.com/dataset/dataDetail?dataId=52> the released dataset is been changed a litte by Alibaba. If you want the totally same dataset as the paper used, you can click [here](http://).

- Amazon: <http://jmcauley.ucsd.edu/data/amazon/links.html>

## Usage

You need to run the file  `data/data_preprocess.py` first to preprocess the data.

`cd data; python data_preprocess.py`

Then use vgg-16 to generate the image feature of each items from their images. Click [here]() if you don't want to generate image feature again, we will upload our extracted feature on Google Drive. 

`python convert_image.py`


```bash

Then you can run the file `./clothing_matching.py` to train the model.

For example: `cd pytorch_code; python clothing_matching.py`

You can also change other parameters according to the usage in the file './config.py':

```bash

## Requirements

- Python 2.7
- Tensorflow 1.5.0


