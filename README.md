# Long-Tail-GAN

The repository contains training and testing codes for an adversarial learning framework for Neural Collaborative Filtering (NCF) models which aims to enhance long-tail item recommendations. 

This model was proposed in the following short paper:

> Adit Krishnan, Ashish Sharma, Aravind Sankar, Hari Sundaram, "An Adversarial Approach to Tackle Long-Tail Recommendation in Neural Collaborative Filtering", 
> 27th ACM International Conference on Information and Knowledge Management (CIKM'18)

## Getting Started

These instructions will get you a copy of the model up and running on your local machine.

### Platforms Supported

- Unix-Like

### Prerequisites

For using our framework you will need Python 2.7+ with the following modules installed:
- [tensorflow](https://www.tensorflow.org/)
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [bottleneck](https://pypi.org/project/Bottleneck/)


## Creating the Input Files

TODO

## Running the Model


### Configure

The model can be configured using the file [config.ini](Codes/config.ini) present inside the [Codes](Codes/) folder. The major parameters to be configured are:

```
GANLAMBDA:       Weight provided to the Adversary's Loss Term (Default = 1.0)
NUM_EPOCH:       Number of Epochs for training (Default = 80)
BATCH_SIZE:      Size of each batch (Default = 100)
LEARNING_RATE:   Learning Rate of the Model (Default = 0.0001)
model_name:      Name with with model is saved (Default = "LT_GAN")
```

### Train

For training the model, run the following command: 

```
$ python2.7 train <path/to/input/folder>
```

By default, the model gets saved to **path/to/input/folder/chkpt** after every epoch.

### Test

For testing the model, run the following command:

```
$ python2.7 test.py <path/to/saved/folder>
```

The mean NDCG, and Recall are reported for the test dataset provided.

### Train

For training the model, run the following command: 

```
$ python2.7 train <path/to/input/folder>
```

By default, the model gets saved to **path/to/input/folder/chkpt/** after every epoch.

### Test

For testing the model, run the following command:

```
$ python2.7 test.py <path/to/saved/folder>
```

The mean NDCG, and Recall are reported for the test dataset provided.