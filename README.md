# SCGAN
Code and data for "Semi-supervised Compatibility Learning Across Categories for Clothing Matching" in ICME2019
<div align=center>
  <img src="https://github.com/CRIPAC-DIG/SCGAN/blob/faec6e65cde7d25c5b5f15e853222dff36979dc1/figures/UCLANs2.png" width = 50% height = 50% />
</div>

## Paper data and code

This is the data and code for the ICME 2019 Paper: [Semi-supervised Compatibility Learning Across Categories for Clothing Matching](https://arxiv.org/pdf/1907.13304.pdf). We have implemented our method in **Tensorflow**.

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `data/`:

- Taobao: <https://tianchi.aliyun.com/dataset/dataDetail?dataId=52> the released dataset is been changed a litte by Alibaba. If you want the totally same dataset as the paper used, you can click [here](http://).

- Amazon: <http://jmcauley.ucsd.edu/data/amazon/links.html>

## Usage

You need to run the file  `data/data_preprocess.py` first to preprocess the data.

`cd data; python data_preprocess.py`

Then use vgg-16 to generate the image feature of each items from their images. Click [here]() if you don't want to generate image feature again, we will upload our extracted feature on Google Drive. 

`python convert_image.py`




Then you can run the file `./clothing_matching.py` to train the model.

For example: `cd pytorch_code; python clothing_matching.py`

You can also change other parameters according to the usage in the file './config.py':



## Requirements

- Python 2.7
- Tensorflow 1.5.0

## Citation
If you find SCGAN useful for your research, please consider citing the following paper:
```
@inproceedings{li2019semi,
  title={Semi-Supervised Compatibility Learning Across Categories for Clothing Matching},
  author={Li, Zekun and Cui, Zeyu and Wu, Shu and Zhang, Xiaoyu and Wang, Liang},
  booktitle={2019 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={484--489},
  year={2019},
  organization={IEEE}
}
```

