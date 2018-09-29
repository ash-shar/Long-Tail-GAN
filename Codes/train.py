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

from data_processing import load_train_data, load_tr_te_data, load_user_items, load_overlap_coeff, load_pop_niche_tags
from data_processing import load_item_one_hot_features as load_item_features


from sample import sample_from_generator_new


def generator_VAECF(p_dims):

	vae = BaseRecommender(p_dims, lam=0.0, random_seed=98765)

	logits_var, loss_var, params = vae.build_graph()

	return vae, logits_var, loss_var, params

def train_GAN(h0_size, h1_size, h2_size, h3_size, NUM_EPOCH, NUM_SUB_EPOCHS, BATCH_SIZE, DISPLAY_ITER, LEARNING_RATE, GENERATOR_SAMPLE_TH, total_anneal_steps, anneal_cap, to_restore, model_name, dataset, GANLAMBDA):


	DATA_DIR = '../Dataset/'+dataset+'/'

	show2id_path = DATA_DIR + "item2id.txt"
	niche_tags_path = DATA_DIR + "niche_items.txt"

	user_tag_matrix_path = DATA_DIR + "tag_counts.csv"

	output_path = "chkpt/"+dataset+"_"+model_name+"_"+str(GANLAMBDA)+"/"

	if not os.path.exists(output_path):
		os.makedirs(output_path)


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


	# Load Binary Interaction Matrix X
	print('Loading Training Interaction Matrix...', end = '')
	train_data, uid_start_idx = load_train_data(os.path.join(pro_dir, 'train_GAN.csv'), n_items)
	print('Done.')
	

	# Load Data for Validation
	print('Loading Validation Matrix...', end = '')
	vad_data_tr, vad_data_te, uid_start_idx_vad = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
											   os.path.join(pro_dir, 'validation_te.csv'), n_items)
	print('Done.')



	# Load User's Popular and Niche Items
	print("Loading User's Popular and Niche Items...", end = '')
	user_popular_data = load_user_items(os.path.join(pro_dir,'train_GAN_popular.csv'))
	user_niche_data = load_user_items(os.path.join(pro_dir,'train_GAN_niche.csv'))
	print("Done.")



	print('Loading item overlap coefficients....', end = '')
	OVERLAP_COEFFS = load_overlap_coeff(show2id_path, user_tag_matrix_path)
	print('Done.')



	N = train_data.shape[0]
	idxlist = range(N)



	#changed USER_FEATURES, ITEM_FEATURE_ARR
	user_x_niche_vectors = {}
	user_x_popular_n_vectors = {}

	for user_idx in range(N):
		if user_idx not in user_popular_data or user_idx not in user_niche_data:
			continue

		curr_pop_vectors = user_popular_data[user_idx]
		curr_niche_vectors = user_niche_data[user_idx]

		curr_x_niche = []
		curr_x_popular_n = []

		for niche_tag in curr_niche_vectors:
			niche_tag_idx = niche_tag

			max_coeff = -1.0
			max_pop_tag_idx = -1


			for pop_tag in curr_pop_vectors:
				pop_tag_idx = pop_tag

				curr_coeff = OVERLAP_COEFFS[niche_tag_idx][pop_tag_idx]

				if curr_coeff > max_coeff:
					max_coeff = curr_coeff
					max_pop_tag_idx = pop_tag_idx

			if niche_tag_idx not in ITEM_FEATURE_DICT or max_pop_tag_idx not in ITEM_FEATURE_DICT:
				# print('Invalid Niche Tag Pair:', niche_tag_idx, max_pop_tag_idx)
				continue

			curr_x_niche.append(niche_tag_idx)
			curr_x_popular_n.append(max_pop_tag_idx)


		user_x_niche_vectors[user_idx] = curr_x_niche

		user_x_popular_n_vectors[user_idx] = curr_x_popular_n

	print('Vectors Loaded')

	#print(len(user_x_niche_vectors), len(user_x_popular_n_vectors))


	
	print('Loading Items to Sample....', end = '')
	USER_TAGS_TO_SAMPLE = {}

	for user_idx in range(N):
		if user_idx not in user_popular_data or user_idx not in user_niche_data:
			continue

		curr_pop_vectors = user_popular_data[user_idx]
		curr_niche_vectors = user_niche_data[user_idx]

		num_niche_tags = len(curr_niche_vectors)

		num_sample_tags = max(2 * len(curr_niche_vectors), 10 - num_niche_tags)

		curr_niche_tags = set()

		curr_sampling_tags = []

		for niche_tag in curr_niche_vectors:
			niche_tag_idx = niche_tag
			curr_niche_tags.add(niche_tag_idx)
			curr_sampling_tags.append(int(niche_tag_idx))

		other_niche_tags = list(NICHE_TAGS - curr_niche_tags)

		other_tags_corr = {}

		for inner_idx in range(len(other_niche_tags)):

			other_tag_idx = other_niche_tags[inner_idx]

			max_coeff = -1.0

			for niche_tag in curr_niche_vectors:
				niche_tag_idx = niche_tag

				curr_coeff = OVERLAP_COEFFS[niche_tag_idx][other_tag_idx]

				if curr_coeff > max_coeff:
					max_coeff = curr_coeff

			other_tags_corr[other_tag_idx] = max_coeff


		sorted_other_tags = sorted(other_tags_corr.items(), key = lambda x: x[1] , reverse = True)

		for inner_idx in range(min(num_sample_tags, len(sorted_other_tags))):
			curr_sampling_tags.append(sorted_other_tags[inner_idx][0])


		curr_sampling_tags.sort()

		USER_TAGS_TO_SAMPLE[user_idx] = np.asarray(curr_sampling_tags)

	print("Done")

	N_vad = vad_data_tr.shape[0]
	idxlist_vad = range(N_vad)

	print('Number of Users: ', N)

	batches_per_epoch = int(np.ceil(float(N) / BATCH_SIZE))

	print('Batches Per Epoch: ', batches_per_epoch)

	global_step = tf.Variable(0, name="global_step", trainable=False)

	tf.reset_default_graph()

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

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	sess.run(init)

	curr_gen_lamda = GANLAMBDA

	update_count = 0.0

	for i in range(NUM_EPOCH):

		batch_total_sampled_tags = []
		batch_curr_x_generated = []
		batch_curr_x_popular_g = []
		batch_curr_x_popular_n = []
		batch_curr_x_niche = []
		batch_X = []
		batch_total_sampled_cnt = []

		# train for each epoch
		user_err_cnt = 0
		for bnum, st_idx in enumerate(range(0, N, BATCH_SIZE)):
			end_idx = min(st_idx + BATCH_SIZE, N)
			X = train_data[idxlist[st_idx:end_idx]]

			# print(X)
			
			if sparse.isspmatrix(X):
				X = X.toarray()
			X = X.astype('float32')

			curr_generator_out = sess.run(generator_out, feed_dict = {generator_network.input_ph: X})

			

			curr_x_popular_n = []
			curr_x_niche = []

			curr_x_popular_g = []
			curr_x_generated = []

			total_sampled_cnt = 0
			total_sampled_tags = []

			for ii, user_idx in enumerate(idxlist[st_idx:end_idx]):
				if user_idx + uid_start_idx not in user_popular_data or user_idx + uid_start_idx not in user_niche_data:
					# print('Invalid User:', user_idx + uid_start_idx)
					user_err_cnt += 1
					total_sampled_tags.append([0]*n_items)
					continue

				curr_pop_vectors = user_popular_data[user_idx + uid_start_idx]
				curr_niche_vectors = user_niche_data[user_idx + uid_start_idx]


				curr_x_niche += user_x_niche_vectors[user_idx + uid_start_idx]
				curr_x_popular_n += user_x_popular_n_vectors[user_idx + uid_start_idx]


				# curr_sampled_tags_bin, curr_sampled_tags = sample_from_generator(range(n_items), curr_generator_out[ii], len(curr_niche_vectors), niche_only = sample_process)

				curr_sampled_tags_bin, curr_sampled_tags = sample_from_generator_new(USER_TAGS_TO_SAMPLE[user_idx + uid_start_idx], np.asarray(curr_generator_out)[ii, USER_TAGS_TO_SAMPLE[user_idx + uid_start_idx]], len(curr_niche_vectors), n_items)

				curr_cnt = 0
				curr_sampled_tags.sort()

				for generated_tag_idx in curr_sampled_tags:

					max_coeff = -1.0
					# max_pop_tag_idx = -1

					max_pop_tag_idx = np.random.choice(range(len(curr_pop_vectors)))

					max_pop_tag_idx = curr_pop_vectors[max_pop_tag_idx]

					if generated_tag_idx not in ITEM_FEATURE_DICT or max_pop_tag_idx not in ITEM_FEATURE_DICT:
						# print('Invalid Generated Tag Pair:', generated_tag_idx, max_pop_tag_idx)
						curr_sampled_tags_bin[generated_tag_idx] = 0
						continue

					curr_x_generated.append(generated_tag_idx)
					curr_x_popular_g.append(max_pop_tag_idx)

					curr_cnt += 1

				total_sampled_tags.append(curr_sampled_tags_bin)
				total_sampled_cnt += curr_cnt


			if curr_x_generated == []:
				continue

			total_sampled_tags = np.asarray(total_sampled_tags)
			curr_x_generated = np.asarray(curr_x_generated)
			curr_x_popular_g = np.asarray(curr_x_popular_g)
			curr_x_popular_n = np.asarray(curr_x_popular_n)
			curr_x_niche = np.asarray(curr_x_niche)

			batch_total_sampled_tags.append(total_sampled_tags)
			batch_curr_x_generated.append(curr_x_generated)
			batch_curr_x_popular_g.append(curr_x_popular_g)
			batch_curr_x_popular_n.append(curr_x_popular_n)
			batch_curr_x_niche.append(curr_x_niche)
			batch_X.append(X)
			batch_total_sampled_cnt.append(total_sampled_cnt)


		batch_total_sampled_tags = np.asarray(batch_total_sampled_tags)
		batch_curr_x_generated = np.asarray(batch_curr_x_generated)
		batch_curr_x_popular_g = np.asarray(batch_curr_x_popular_g)
		batch_curr_x_popular_n = np.asarray(batch_curr_x_popular_n)
		batch_curr_x_niche = np.asarray(batch_curr_x_niche)
		batch_X = np.asarray(batch_X)
		batch_total_sampled_cnt = np.asarray(batch_total_sampled_cnt)

		print("global-epoch:", i, "Data Creation Finished", "user_err_cnt:", user_err_cnt)

		# print(batch_total_sampled_cnt.tolist())

		indices = np.arange(batch_total_sampled_tags.shape[0])
		np.random.shuffle(indices)

		for j_disc in range(NUM_SUB_EPOCHS):
			
			for disc_batch_idx in indices:

				X = batch_X[disc_batch_idx]
				curr_x_popular_id_n = batch_curr_x_popular_n[disc_batch_idx]
				curr_x_popular_id_g = batch_curr_x_popular_g[disc_batch_idx]
				curr_x_niche_id = batch_curr_x_niche[disc_batch_idx]
				curr_x_generated_id = batch_curr_x_generated[disc_batch_idx]
				total_sampled_tags = batch_total_sampled_tags[disc_batch_idx]
				total_sampled_cnt = batch_total_sampled_cnt[disc_batch_idx]


				_, curr_d_loss = sess.run([d_trainer, d_loss_mean], feed_dict={generator_network.input_ph: X, x_popular_n_id: curr_x_popular_id_n, x_popular_g_id: curr_x_popular_id_g , x_niche_id: curr_x_niche_id, x_generated_id: curr_x_generated_id, generated_tags: total_sampled_tags, sampled_cnt: total_sampled_cnt, keep_prob: np.sum(0.7).astype(np.float32), item_feature_arr: ITEM_FEATURE_ARR})


			print("global-epoch:%s, discr-epoch:%s, d_loss:%.5f" % (i, j_disc, curr_d_loss))

		print('')

		for j_gen in range(NUM_SUB_EPOCHS):
			
			for gen_batch_idx in indices:
				X = batch_X[gen_batch_idx]
				curr_x_popular_id_n = batch_curr_x_popular_n[gen_batch_idx]
				curr_x_popular_id_g = batch_curr_x_popular_g[gen_batch_idx]
				curr_x_niche_id = batch_curr_x_niche[gen_batch_idx]
				curr_x_generated_id = batch_curr_x_generated[gen_batch_idx]
				total_sampled_tags = batch_total_sampled_tags[gen_batch_idx]
				total_sampled_cnt = batch_total_sampled_cnt[gen_batch_idx]


				if total_anneal_steps > 0:
						anneal = min(anneal_cap, 1. * ((update_count) / total_anneal_steps))
				else:
					anneal = anneal_cap

				update_count += 1

				_, curr_g_loss, curr_g_loss_term_1, curr_g_loss_term_2 = sess.run([g_trainer, g_loss_mean, g_vae_loss, gan_loss], feed_dict={generator_network.input_ph: X, x_popular_n_id: curr_x_popular_id_n, x_popular_g_id: curr_x_popular_id_g , x_niche_id: curr_x_niche_id, x_generated_id: curr_x_generated_id, generated_tags: total_sampled_tags, sampled_cnt: total_sampled_cnt, generator_network.keep_prob_ph: 0.75, generator_network.is_training_ph: 1, generator_network.anneal_ph: anneal, gen_lambda: curr_gen_lamda, keep_prob: np.sum(0.7).astype(np.float32)})


			print("global-epoch:%s, generator-epoch:%s, g_loss:%.5f (vae_loss: %.5f + gan_loss: %.5f, anneal: %.5f)" % (i, j_gen, curr_g_loss, curr_g_loss_term_1, curr_g_loss_term_2, anneal))

		print('')

		X_vad = vad_data_tr[idxlist_vad[0:N_vad]]

		if sparse.isspmatrix(X_vad):
			X_vad = X_vad.toarray()
		X_vad = X_vad.astype('float32')
		
		pred_vad = sess.run(generator_out, feed_dict={generator_network.input_ph: X_vad} )
		# exclude examples from training and validation (if any)
		pred_vad[X_vad.nonzero()] = -np.inf
		ndcg_vad = NDCG_binary_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]])
		
		recall_at_20, not_found_20 = Recall_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]], k=20)

		recall_at_50, not_found_50 = Recall_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]], k=50)

		print('global-epoch:', i , 'gen-epoch:', j_gen, 'Vad: NDCG:', np.mean(ndcg_vad), 'Recall@20:', np.mean(recall_at_20), 'Recall@50:', np.mean(recall_at_50), 'Num_users:', len(ndcg_vad), len(recall_at_20), len(recall_at_50))


		print('')

		
		saver.save(sess, os.path.join(output_path, "model_"+str(i)))

		print('Model saved at global-epoch', i)


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

dataset = sys.argv[1]

train_GAN(h0_size, h1_size, h2_size, h3_size, NUM_EPOCH, NUM_SUB_EPOCHS, BATCH_SIZE, DISPLAY_ITER, LEARNING_RATE, GENERATOR_SAMPLE_TH, total_anneal_steps, anneal_cap, to_restore, model_name, dataset, GANLAMBDA)
