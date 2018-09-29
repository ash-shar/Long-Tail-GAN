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



def test_GAN():



	curr_gen_lamda = GANLAMBDA

	update_count = 0.0

	n100_list, r20_list, r50_list = [], [], []

	user_li = []

	not_found_20_list, not_found_50_list = [], []


	with tf.Session() as sess:
		saver.restore(sess, os.path.join(output_path, "model_"+str(epoch_no)))

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

			# print(np.shape(pred_val))

			# print(np.shape(test_data_te[idxlist_test[st_idx:end_idx]]))

			# exit(-1)

			ndcg = NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100)
			# if ndcg != 0:
			n100_list.append(ndcg)

			recall_at_20, not_found_20 = Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20)

			recall_at_50, not_found_50 = Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50)

			r20_list.append(recall_at_20)

			r50_list.append(recall_at_50)

			not_found_20_list.append(not_found_20)
			not_found_50_list.append(not_found_50)


			curr_user_li = []


			for user_idx in idxlist_test[st_idx:end_idx]:
				# print(user_idx+uid_start_idx, test_data_te[user_idx].nonzero())
				curr_user_li.append(user_idx+uid_start_idx)

			# rows,cols = test_data_te[idxlist_test[st_idx:end_idx]].nonzero()
			# for row,col in zip(rows,cols):
			#   curr_user_li.append(row+uid_start_idx)


			# print(test_data_te[idxlist_test[st_idx:end_idx]], type(test_data_te[idxlist_test[st_idx:end_idx]]))

			# curr_user_li = test_data_te[idxlist_test[st_idx:end_idx]]
			user_li.append(curr_user_li)
		
	# n100_list = np.concatenate(n100_list)
	# r20_list = np.concatenate(r20_list)
	# r50_list = np.concatenate(r50_list)

	for idx, elem in enumerate(n100_list):
		for inner_idx, inner_elem in enumerate(elem):
			# print(str(user_li[idx][inner_idx])+'\t'+str(n100_list[idx][inner_idx])+'\t'+str(r20_list[idx][inner_idx])+'\t'+str(r50_list[idx][inner_idx]))
			print(str(user_li[idx][inner_idx])+'\t'+str(n100_list[idx][inner_idx])+'\t'+str(r20_list[idx][inner_idx])+'\t'+str(r50_list[idx][inner_idx])+'\t'+str(not_found_20_list[idx][inner_idx])+'\t'+str(not_found_50_list[idx][inner_idx]))


	print(str(np.mean(n100_list)) + '\t' + str(np.mean(r20_list)) + '\t' + str(np.mean(r50_list)))
