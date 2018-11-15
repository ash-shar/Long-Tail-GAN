# Long-Tail-GAN

The repository contains training and testing codes for the generative adversarial learning framework for Neural Collaborative Filtering (NCF) models which aims to enhance long-tail item recommendations. 

If this code helps you in your research, please cite the following publication:

> Krishnan, Adit, et al. "An Adversarial Approach to Improve Long-Tail Performance in Neural Collaborative Filtering." Proceedings of the 27th ACM International Conference on Information and Knowledge Management. ACM, 2018.
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


## Input Files

You will need the following files for running our model:

```
tag_counts.csv:         CSV file containing userId, itemId, and rating (given by user to item) separated by comma (,) 
item_list.txt:          List of item ids.
unique_item_id.txt:     Items to use for training and testing (say, only use items rated by atleast 5 users)
item2id.txt:            Mapping which makes item ids in unique_item_id sequential (0 to num_item), tab-separated
profile2id.txt:         Mapping which makes user ids sequential (0 to num_user), tab-separated
niche_items.txt:        Items which are niche (original ids) 
train_GAN.csv:          CSV file containing pairs of userId (mapped), itemId (mapped) with rating greater than an application-specific threshold
train_GAN_popular.csv:  userId (mapped), itemId (mapped) pairs of niche items
train_GAN_niche.csv:    userId (mapped), itemId (mapped) pairs of popular items (unique_items - niche items)
validation_tr.csv:      Training data for Validation (userId (mapped), itemId (mapped) pairs)
validation_te.csv:      Test Data for Validation (userId (mapped), itemId (mapped) pairs)
test_tr.csv:            Training data for Testing (userId (mapped), itemId (mapped) pairs)
test_te.csv:            Test Data for Testing (userId (mapped), itemId (mapped) pairs)
```

Sample set of input files for Askubuntu dataset is present in the [Dataset](Dataset/) folder. Refer to the following ipython notebook for details regarding creation of these files for movielens dataset: [ml-parse-vaecf](https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb) 

## Running the Model


### Configure

The model can be configured using the file [config.ini](Codes/config.ini) present inside the [Codes](Codes/) folder. The major parameters to be configured are:

```
GANLAMBDA:       Weight provided to the Adversary's Loss Term (Default = 1.0)
NUM_EPOCH:       Number of Epochs for training (Default = 80)
BATCH_SIZE:      Size of each batch (Default = 100)
LEARNING_RATE:   Learning Rate of the Model (Default = 0.0001)
model_name:      Name by which model is saved (Default = "LT_GAN")
```

### Train

For training the model, run the following command: 

```
$ python2.7 train.py <path/to/input/folder>
```

By default, the model gets saved to **path/to/input/folder/chkpt/** after every epoch.

### Test

For testing the model, run the following command:

```
$ python2.7 test.py <path/to/saved/folder>
```
