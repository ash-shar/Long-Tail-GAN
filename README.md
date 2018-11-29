# Long-Tail-GAN

This repository contains the training and testing codes for the Generative Adversarial learning framework for Neural Collaborative Filtering (NCF) models, which aims to enhance long-tail item recommendations. 

If this code helps you in your research, please cite the following publication:

> Krishnan, Adit, et al. "An Adversarial Approach to Improve Long-Tail Performance in Neural Collaborative Filtering." Proceedings of the 27th ACM International Conference on Information and Knowledge Management. ACM, 2018.
## Getting Started

These instructions will help you setup the proposed model on your local machine.

### Platforms Supported

- Unix, MacOS, Windows (with appropriate Python and Tensorflow environment)

### Prerequisites

Our framework can be compiled on Python 2.7+ environments with the following modules installed:
- [tensorflow](https://www.tensorflow.org/)
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [bottleneck](https://pypi.org/project/Bottleneck/)

These requirements may be satisified with an updated Anaconda environment as well - https://www.anaconda.com/


## Input Files

You will need the following files for running our model:

```
item_counts.csv:        CSV file containing userId, itemId, and rating (given by user to item) separated by comma (,) 
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

A set of input files for a sampled version of Askubuntu dataset are present in the [Dataset](Dataset/) folder. Note that we use the set of tags assigned to the posts of a user as items; the posts correspond to the questions asked by the user, the answers given by the user, the posts liked by the user, and the posts to which the user commented.

Refer to the following ipython notebook for details regarding creation of these files for movielens dataset: [ml-parse-vaecf](https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb). The movies rated by the users are the items.

## Running the Model


### Configure

The model can be configured using the file [config.ini](Codes/config.ini) present inside the [Codes](Codes/) folder. The parameters h0_size, h1_size, h2_size, and h3_size are the sizes of the hidden layers as defined in the architecture of our discriminator in the GAN framework (see figure).

![Architecture](architecture.JPG?raw=true "Title") 


The other parameters to be configured are:

```
GANLAMBDA:       Weight provided to the Adversary's Loss Term (Default = 1.0)
NUM_EPOCH:       Number of Epochs for training (Default = 80)
BATCH_SIZE:      Size of each batch (Default = 100)
LEARNING_RATE:   Learning Rate of the Model (Default = 0.0001)
model_name:      Name by which model is saved (Default = "LT_GAN")
```

### Base Recommender

The repo uses [VAE-CF](https://arxiv.org/abs/1802.05814) as the base recommender (generator in our architecture) by default. You can also replace this with your own recommender models (or other recommenders) to be trained with the GAN loss and long-tail strategy proposed by us. Follow the below instructions:

1. Create a python class of your recommender. You can use the [VAECF class](Codes/Base_Recommender/MultiVAE.py) as a template.
2. Write a wrapper function for your recommender class in the [generator.py](Codes/generator.py) file. The function should take path to the dataset folder as input, irrespective of its usage. Eg., for Askubuntu dataset, it will take path to the [Askubuntu folder](Dataset/Askubuntu) as input. The function should return the following: object to the defined class, probability distribution over the set of items (recommender's output), loss function of the recommender, parameters of the recommender to learn, and hyperparamters used by the recommender. Again, refer the wrapper function of VAE-CF defined in the code.   
3. In the [train.py](Codes/train.py) file, import the wrapper function of your recommender instead of generator_VAECF (line 21).
4. If the set of hyperparameters of your recommender are similar to VAE-CF, then no more change would be needed. Otherwise, you might need to take care of them in the code, especially if some of them are updated over training iterations (like annealing). 

### Train

For training the model, run the following command: 

```
$ python2.7 train.py <path/to/input/folder>
```

Model parameters are set to the values provided in the config file. By default, the trained model is checkpointed and saved to **path/to/input/folder/chkpt/** after every epoch.

### Test

For testing the model, run the following command:

```
$ python2.7 test.py <path/to/input/folder> <path/to/saved/model>
```

where Path to saved model is the path to the saved model file inside chkpt folder (will be model_<last_epoch> by default).
