from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import shutil
import codecs
import psutil
from scipy import sparse
import pandas as pd
import bottleneck as bn

from math import isinf

import sys
import pickle
import ConfigParser


from Base_Recommender.MultiVAE import MultiVAE as BaseRecommender

from data_processing import load_tr_te_data, load_pop_niche_tags
from data_processing import load_item_one_hot_features as load_item_features

from generator import generator_VAECF as generator

from discriminator import discriminator

from eval_functions import NDCG_binary_at_k_batch, Recall_at_k_batch

def test_GAN(h0_size, h1_size, h2_size, h3_size, NUM_EPOCH, NUM_SUB_EPOCHS, BATCH_SIZE, DISPLAY_ITER, LEARNING_RATE, to_restore, model_name, dataset, GANLAMBDA, output_path):


	DATA_DIR = dataset+'/'

	show2id_path = DATA_DIR + "item2id.txt"
	niche_tags_path = DATA_DIR + "niche_items.txt"

	user_tag_matrix_path = DATA_DIR + "item_counts.csv"


	item_list_path = DATA_DIR + 'item_list.txt'

	pro_dir = DATA_DIR # os.path.join(DATA_DIR, 'pro_sg_tags_1k')


	unique_sid = list()
	with open(os.path.join(pro_dir, 'unique_item_id.txt'), 'r') as f:
		for line in f:
			unique_sid.append(line.strip())

	n_items = len(unique_sid)



	print('Loading Items...', end = '')
	SHOW2ID, IDs_present, NICHE_TAGS, ALL_TAGS, OTHER_TAGS = load_pop_niche_tags(show2id_path, item_list_path, niche_tags_path, n_items)
	print('Done.')


	# One Hot Vectors for Items
	print('Loading Item Features...', end = '')
	ITEM_FEATURE_DICT, FEATURE_LEN, ITEM_FEATURE_ARR = load_item_features(item_list_path, SHOW2ID, n_items)
	print('Done.')


	# Load Data for Testing
	print('Loading Test Matrix...', end = '')
	test_data_tr, test_data_te, uid_start_idx = load_tr_te_data(os.path.join(pro_dir, 'test_tr.csv'), 
															os.path.join(pro_dir, 'test_te.csv'), n_items)

	N_test = test_data_tr.shape[0]
	print('N_test:', N_test)

	idxlist_test = range(N_test)

	batch_size_test = 20000

	# Generator
	generator_network, generator_out, g_vae_loss, g_params, p_dims, total_anneal_steps, anneal_cap = generator(pro_dir)

	generated_tags = tf.placeholder(tf.float32, [None, n_items], name = "generated_tags")


	# Discriminator
	y_data, y_generated, d_params, x_generated_id, x_popular_n_id, x_popular_g_id, x_niche_id, item_feature_arr, keep_prob = discriminator(n_items, FEATURE_LEN, h0_size, h1_size, h2_size, h3_size)
	

	zero = tf.constant(0, dtype=tf.float32)


	# Loss Function

	d_loss = - tf.reduce_sum(tf.log(y_data)) - tf.reduce_sum(tf.log(1 - y_generated))
	d_loss_mean = tf.reduce_mean(d_loss)

	sampled_generator_out = tf.multiply(generator_out, generated_tags)

	sampled_generator_out = tf.reshape(sampled_generator_out, [-1])

	sampled_generator_out_non_zero = tf.gather_nd(sampled_generator_out ,tf.where(tf.not_equal(sampled_generator_out, zero)))

	sampled_cnt = tf.placeholder_with_default(1., shape=None)
	gen_lambda = tf.placeholder_with_default(1.0, shape=None)


	g_loss = g_vae_loss - (1.0 * gen_lambda / sampled_cnt) * tf.reduce_sum(tf.multiply(sampled_generator_out_non_zero, y_generated))
	g_loss_mean = tf.reduce_mean(g_loss)
	gan_loss = - (1.0 * gen_lambda / sampled_cnt) * tf.reduce_sum(tf.multiply(sampled_generator_out_non_zero, y_generated))

	# optimizer : AdamOptimizer
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

	# discriminator and generator loss
	d_trainer = optimizer.minimize(d_loss, var_list=d_params)
	g_trainer = optimizer.minimize(g_loss, var_list=g_params)


	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	curr_gen_lamda = GANLAMBDA

	update_count = 0.0

	n100_list, r20_list, r50_list = [], [], []

	user_li = []

	not_found_20_list, not_found_50_list = [], []


	with tf.Session() as sess:
		saver.restore(sess, output_path)

		print('Model Loaded')

		for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
			end_idx = min(st_idx + batch_size_test, N_test)
			X = test_data_tr[idxlist_test[st_idx:end_idx]]

			if sparse.isspmatrix(X):
				X = X.toarray()
			X = X.astype('float32')

			pred_val = sess.run(generator_out, feed_dict = {generator_network.input_ph: X})
			
			# exclude examples from training and validation (if any)
			pred_val[X.nonzero()] = -np.inf

			ndcg = NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100)

			n100_list.append(ndcg)

			recall_at_20, not_found_20 = Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20)

			recall_at_50, not_found_50 = Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50)

			r20_list.append(recall_at_20)

			r50_list.append(recall_at_50)

			not_found_20_list.append(not_found_20)
			not_found_50_list.append(not_found_50)

			curr_user_li = []

			for user_idx in idxlist_test[st_idx:end_idx]:
				curr_user_li.append(user_idx+uid_start_idx)

			user_li.append(curr_user_li)

	print(str(np.mean(n100_list)) + '\t' + str(np.mean(r20_list)) + '\t' + str(np.mean(r50_list)))



configParser = ConfigParser.RawConfigParser()   
configFilePath = r'config.ini'
configParser.read(configFilePath)

h0_size = int(configParser.get('Long-Tail-GAN', 'h0_size'))
h1_size = int(configParser.get('Long-Tail-GAN', 'h1_size'))
h2_size = int(configParser.get('Long-Tail-GAN', 'h2_size'))
h3_size = int(configParser.get('Long-Tail-GAN', 'h3_size'))

NUM_EPOCH = int(configParser.get('Long-Tail-GAN', 'NUM_EPOCH'))
NUM_SUB_EPOCHS = int(NUM_EPOCH/8)
BATCH_SIZE = int(configParser.get('Long-Tail-GAN', 'BATCH_SIZE'))

DISPLAY_ITER = int(configParser.get('Long-Tail-GAN', 'DISPLAY_ITER'))
LEARNING_RATE = float(configParser.get('Long-Tail-GAN', 'LEARNING_RATE'))
to_restore = int(configParser.get('Long-Tail-GAN', 'to_restore'))
GANLAMBDA = float(configParser.get('Long-Tail-GAN', 'GANLAMBDA'))

model_name = configParser.get('Long-Tail-GAN', 'model_name')


dataset = sys.argv[1]
output_path = sys.argv[2]


test_GAN(h0_size, h1_size, h2_size, h3_size, NUM_EPOCH, NUM_SUB_EPOCHS, BATCH_SIZE, DISPLAY_ITER, LEARNING_RATE, to_restore, model_name, dataset, GANLAMBDA, output_path)