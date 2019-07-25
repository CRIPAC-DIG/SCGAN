# SCGAN
code for "Semi-supervised Compatibility Learning Across Categories for Clothing Matching" in ICME2019

## Paper data and code

This is the code for the ICME 2019 Paper: [Semi-supervised Compatibility Learning Across Categories for Clothing Matching](https://). We have implemented our methods in **Tensorflow**.

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `data/`:

- Taobao: <http://>

- Amazon: <http://>

## Usage

You need to run the file  `data/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=sample` ('preprocess.py' is still on arrangement.)

```bash
usage: preprocess.py [-h] [--dataset DATASET]

Then you can run the file `./clothing_matching.py` to train the model.

For example: `cd pytorch_code; python main.py --dataset=sample`

You can add the suffix `--nonhybrid` to use the global preference of a session graph to recommend instead of the hybrid preference.

You can also change other parameters according to the usage in the file './config.py':

```bash

## Requirements

- Python 2.7
- Tensorflow 1.5.0


