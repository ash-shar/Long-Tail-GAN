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


def generator_VAECF(p_dims):

	vae = BaseRecommender(p_dims, lam=0.0, random_seed=98765)

	logits_var, loss_var, params = vae.build_graph()

	return vae, logits_var, loss_var, params


def test_GAN(h0_size, h1_size, h2_size, h3_size, NUM_EPOCH, NUM_SUB_EPOCHS, BATCH_SIZE, DISPLAY_ITER, LEARNING_RATE, GENERATOR_SAMPLE_TH, total_anneal_steps, anneal_cap, to_restore, model_name, dataset, GANLAMBDA, output_path):


	DATA_DIR = '../Dataset/'+dataset+'/'

	show2id_path = DATA_DIR + "item2id.txt"
	niche_tags_path = DATA_DIR + "niche_items.txt"

	user_tag_matrix_path = DATA_DIR + "tag_counts.csv"

	# output_path = "chkpt/"+dataset+"_"+model_name+"_"+str(GANLAMBDA)+"/"

	# if not os.path.exists(output_path):
	# 	os.makedirs(output_path)


	item_list_path = DATA_DIR + 'item_list.txt'

	pro_dir = DATA_DIR # os.path.join(DATA_DIR, 'pro_sg_tags_1k')


	unique_sid = list()
	with open(os.path.join(pro_dir, 'unique_item_id.txt'), 'r') as f:
		for line in f:
			unique_sid.append(line.strip())

	n_items = len(unique_sid)

	p_dims = [200, 600, n_items]



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
	generator_network, generator_out, g_vae_loss, g_params = generator_VAECF(p_dims)
	generated_tags = tf.placeholder(tf.float32, [None, n_items], name = "generated_tags")

	
	# Discriminator
	x_generated_id = tf.placeholder(tf.int32, [None], name = "x_generated")
	x_popular_n_id = tf.placeholder(tf.int32, [None], name="x_popular_n")
	x_popular_g_id = tf.placeholder(tf.int32, [None], name="x_popular_g")
	x_niche_id = tf.placeholder(tf.int32, [None], name="x_niche")

	item_feature_arr = tf.placeholder(tf.float32, [None, FEATURE_LEN], name="item_feature_arr") # num_tags x ...

	keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout

	emb_matrix = tf.Variable(tf.truncated_normal([FEATURE_LEN, h0_size], stddev=0.1), name="d_w1", dtype=tf.float32)

	x_generated = tf.nn.embedding_lookup(emb_matrix, x_generated_id) # [None, h0_size]
	x_popular_n = tf.nn.embedding_lookup(emb_matrix, x_popular_n_id) # [None, h0_size]
	x_popular_g = tf.nn.embedding_lookup(emb_matrix, x_popular_g_id) # [None, h0_size]
	x_niche = tf.nn.embedding_lookup(emb_matrix, x_niche_id) # [None, h0_size]

	#Discriminator
	# Popular Tags
	w1 = tf.Variable(tf.truncated_normal([h0_size, h1_size], stddev=0.1), name="d_w1", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([h1_size]), name="d_b1", dtype=tf.float32)
	h1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(x_popular_n, w1) + b1), keep_prob)

	# Niche Tags
	w2 = tf.Variable(tf.truncated_normal([h0_size, h2_size], stddev=0.1), name="d_w2", dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([h2_size]), name="d_b2", dtype=tf.float32)
	h2 = tf.nn.dropout(tf.nn.tanh(tf.matmul(x_niche, w2) + b2), keep_prob)


	h_in_data = tf.concat([h1, h2], 1)

	# Fully Connected Layer 1
	w3 = tf.Variable(tf.truncated_normal([h1_size + h2_size, h3_size], stddev=0.1), name="d_w3", dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([h3_size]), name="d_b3", dtype=tf.float32)

	# Fully Connected Layer 2
	w4 = tf.Variable(tf.truncated_normal([h3_size, 1], stddev=0.1), name="d_w4", dtype=tf.float32)
	b4 = tf.Variable(tf.zeros([1]), name="d_b4", dtype=tf.float32)


	y_data = tf.nn.dropout(tf.nn.tanh(tf.matmul(h_in_data, w3) + b3), keep_prob)
	y_data = tf.nn.sigmoid(tf.matmul(y_data, w4) + b4)

	d_params = [w1, b1, w2, b2, w3, b3, w4, b4]


	# Generated Tags
	h3 = tf.nn.dropout(tf.nn.tanh(tf.matmul(x_popular_g, w1) + b1), keep_prob)
	h4 = tf.nn.dropout(tf.nn.tanh(tf.matmul(x_generated, w2) + b2), keep_prob)
	h_in_gen = tf.concat([h3, h4], 1)
	y_generated = tf.nn.dropout(tf.nn.tanh(tf.matmul(h_in_gen, w3) + b3), keep_prob)
	y_generated = tf.nn.sigmoid(tf.matmul(y_generated, w4) + b4)
	

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
		

	# for idx, elem in enumerate(n100_list):
	# 	for inner_idx, inner_elem in enumerate(elem):
	# 		print(str(user_li[idx][inner_idx])+'\t'+str(n100_list[idx][inner_idx])+'\t'+str(r20_list[idx][inner_idx])+'\t'+str(r50_list[idx][inner_idx])+'\t'+str(not_found_20_list[idx][inner_idx])+'\t'+str(not_found_50_list[idx][inner_idx]))


	print(str(np.mean(n100_list)) + '\t' + str(np.mean(r20_list)) + '\t' + str(np.mean(r50_list)))



configParser = ConfigParser.RawConfigParser()   
configFilePath = r'config.ini'
configParser.read(configFilePath)

h0_size = int(configParser.get('Long-Tail-GAN', 'h0_size'))
h1_size = int(configParser.get('Long-Tail-GAN', 'h1_size'))
h2_size = int(configParser.get('Long-Tail-GAN', 'h2_size'))
h3_size = int(configParser.get('Long-Tail-GAN', 'h3_size'))

NUM_EPOCH = int(configParser.get('Long-Tail-GAN', 'NUM_EPOCH'))
NUM_SUB_EPOCHS = int(configParser.get('Long-Tail-GAN', 'NUM_SUB_EPOCHS'))
BATCH_SIZE = int(configParser.get('Long-Tail-GAN', 'BATCH_SIZE'))

DISPLAY_ITER = int(configParser.get('Long-Tail-GAN', 'DISPLAY_ITER'))
LEARNING_RATE = float(configParser.get('Long-Tail-GAN', 'LEARNING_RATE'))
GENERATOR_SAMPLE_TH = float(configParser.get('Long-Tail-GAN', 'GENERATOR_SAMPLE_TH'))
total_anneal_steps = int(configParser.get('Long-Tail-GAN', 'total_anneal_steps'))
anneal_cap = float(configParser.get('Long-Tail-GAN', 'anneal_cap'))
to_restore = int(configParser.get('Long-Tail-GAN', 'to_restore'))
GANLAMBDA = float(configParser.get('Long-Tail-GAN', 'GANLAMBDA'))

model_name = configParser.get('Long-Tail-GAN', 'model_name')


output_path = sys.argv[1]


test_GAN(h0_size, h1_size, h2_size, h3_size, NUM_EPOCH, NUM_SUB_EPOCHS, BATCH_SIZE, DISPLAY_ITER, LEARNING_RATE, GENERATOR_SAMPLE_TH, total_anneal_steps, anneal_cap, to_restore, model_name, dataset, GANLAMBDA, output_path)