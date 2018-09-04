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

import MultiVAE
import sys
import pickle

model_name = sys.argv[1] # "GAN_POP_5"

sample_process = int(sys.argv[2])
GANLAMBDA = float(sys.argv[3])

if sample_process == 1:
	model_name = model_name + "_niche"


dataset = "Askubuntu"

show2id_path = "../Dataset/Full-Data-"+dataset+"/pro_sg_tags_1k/show2id.txt"
niche_tags_path = "../Dataset/Full-Data-"+dataset+"/niche_tags_1k.txt"

user_tag_matrix_path = "../Dataset/Full-Data-"+dataset+"/tag_counts_1k.csv"

output_path = "chkpt/"+dataset+"_"+model_name+"_"+str(h1_size)+"_"+str(h2_size)+"_"+str(h3_size)+"/"
if not os.path.exists(output_path):
        os.makedirs(output_path)


DATA_DIR = '../Dataset/Full-Data-'+dataset+'/'

tag_feature_vectors_path = DATA_DIR+'parsed_tag_features.txt'

pro_dir = os.path.join(DATA_DIR, 'pro_sg_tags_1k')


unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
	for line in f:
		unique_sid.append(line.strip())

n_items = len(unique_sid)

p_dims = [200, 600, n_items]



def generate_minibatches(input, output, k):
	indices = np.arange(input.shape[0])
	np.random.shuffle(indices)

	for i in range(0, input.shape[0] - k + 1, k):
		r = indices[i:i + k]
		# print(input[r].shape)
		
		yield input[r], output[r]


def load_overlap_coeff():

	SHOW2ID = {}

	show2id_file = codecs.open(show2id_path, 'r', 'utf-8')

	for row in show2id_file:
		s = row.strip().split('\t')

		SHOW2ID[s[0]] = s[1]


	show2id_file.close()


	TAG_SETS = {}

	TAG_IDs = set()

	# user_tag_matrix_file = codecs.open(user_tag_matrix_path, 'r', 'utf-8')

	# i = 0
	# for row in user_tag_matrix_file:
	# 	if i == 0:
	# 		i = 1
	# 		continue

	# 	s = row.strip().split(',')

	# 	user_id = s[0]
	# 	try:
	# 		tag_id = SHOW2ID[s[1]]
	# 	except:
	# 		print('Error:', tag_id)
	# 		continue

	# 	if tag_id not in TAG_SETS:
	# 		TAG_SETS[tag_id] = set()

	# 	TAG_SETS[tag_id].add(user_id)

	# 	TAG_IDs.add(tag_id)

	# user_tag_matrix_file.close()

	# print('TAG SETS:', len(TAG_SETS), len(TAG_IDs))

	# TAG_IDs = list(TAG_IDs)

	# pickle.dump(TAG_SETS, open(pro_dir+'/TAG_SETS.pkl', 'wb'))
	# pickle.dump(TAG_IDs, open(pro_dir+'/TAG_IDs.pkl', 'wb'))
	TAG_SETS = pickle.load(open(pro_dir+'/TAG_SETS.pkl', 'rb'))
	TAG_IDs = pickle.load(open(pro_dir+'/TAG_IDs.pkl', 'rb'))

	OVERLAP_COEFFS = {}

	# for idx in range(len(TAG_IDs)):
	# 	OVERLAP_COEFFS[int(TAG_IDs[idx])] = {}
	# 	for inner_idx in range(len(TAG_IDs)):
	# 		OVERLAP_COEFFS[int(TAG_IDs[idx])][int(TAG_IDs[inner_idx])] = overlap_cofficient(TAG_SETS[TAG_IDs[idx]], TAG_SETS[TAG_IDs[inner_idx]])

	# pickle.dump(OVERLAP_COEFFS, open(pro_dir+'/OVERLAP_COEFFS.pkl', 'wb'))
	OVERLAP_COEFFS = pickle.load(open(pro_dir+'/OVERLAP_COEFFS.pkl', 'rb'))
	return OVERLAP_COEFFS


def load_data_from_file():
	popular_vectors = []
	niche_vectors = []

	user_vectors_file = codecs.open(user_vectors_path, 'r', 'utf-8')

	idx = 0

	for row in user_vectors_file:
		s = row.strip().split('\t')

		pop_vec = eval(s[1])
		niche_vec = eval(s[2])

		popular_vectors.append(pop_vec)
		niche_vectors.append(niche_vec)

		idx += 1

		if idx == 5000:
			break

	user_vectors_file.close()


	popular_vectors = np.asarray(popular_vectors)
	niche_vectors = np.asarray(niche_vectors)

	return popular_vectors, niche_vectors


def overlap_cofficient(x,y):
	num = len(list(x & y))

	denom = min(len(x),len(y))

	overlap = (num*1.0)/(1.0*denom)

	return overlap

def load_data(USER_SET = None):

	POPULAR_TAGS = {}

	popular_tags_file = codecs.open(popular_tags_path, 'r', 'utf-8')

	tag_idx = 0

	for row in popular_tags_file:
		s = row.strip()

		if s not in POPULAR_TAGS:
			POPULAR_TAGS[s] = tag_idx
			tag_idx += 1

	popular_tags_file.close()

	NICHE_TAGS = {}

	niche_tags_file = codecs.open(niche_tags_path, 'r', 'utf-8')

	tag_idx = 0

	for row in niche_tags_file:
		s = row.strip()

		if s not in NICHE_TAGS:
			NICHE_TAGS[s] = tag_idx
			tag_idx += 1

	niche_tags_file.close()


	USER_POP_VECTOR = {}
	USER_NICHE_VECTOR = {}

	USER_NICHE_TAGS = {}

	USER_POP_TAGS = {}

	user_tag_matrix_file = codecs.open(user_tag_matrix_path, 'r', 'utf-8')

	idx = 0

	for row in user_tag_matrix_file:
		if idx == 0:
			idx = 1
			continue

		s = row.strip().split(',')

		user_id = s[0]

		if user_id not in USER_NICHE_VECTOR:
			USER_NICHE_VECTOR[user_id] = [0]*len(NICHE_TAGS)

		if user_id not in USER_POP_VECTOR:
			USER_POP_VECTOR[user_id] = [0]*len(POPULAR_TAGS)

		if user_id not in USER_POP_TAGS:
			USER_POP_TAGS[user_id] = set()
			USER_NICHE_TAGS[user_id] = set()

		tag_id = s[1]

		if tag_id in POPULAR_TAGS:
			USER_POP_VECTOR[user_id][POPULAR_TAGS[tag_id]] = 1.0

			USER_POP_TAGS[user_id].add(tag_id)

		elif tag_id in NICHE_TAGS:
			USER_NICHE_VECTOR[user_id][NICHE_TAGS[tag_id]] = 1.0
			USER_NICHE_TAGS[user_id].add(tag_id)

	user_tag_matrix_file.close()

	print('USER_MATRIX_Loaded:', len(USER_POP_VECTOR), len(USER_NICHE_VECTOR))



	TAG_SETS = {}

	TAG_IDs = set()

	user_tag_matrix_file = codecs.open(user_tag_matrix_path, 'r', 'utf-8')

	i = 0
	for row in user_tag_matrix_file:
		if i == 0:
			i = 1
			continue

		s = row.strip().split(',')

		user_id = s[0]
		tag_id = s[1]

		if tag_id not in TAG_SETS:
			TAG_SETS[tag_id] = set()

		TAG_SETS[tag_id].add(user_id)

		TAG_IDs.add(tag_id)

	user_tag_matrix_file.close()

	print('TAG SETS:', len(TAG_SETS), len(TAG_IDs))


	OVERLAP_COEFFS = {}

	for niche_tag in NICHE_TAGS:
		OVERLAP_COEFFS[niche_tag] = {}
		for pop_tag in POPULAR_TAGS:
			OVERLAP_COEFFS[niche_tag][pop_tag] = overlap_cofficient(TAG_SETS[niche_tag], TAG_SETS[pop_tag])


	# return OVERLAP_COEFFS

	discarded_users = 0

	popular_vectors = []
	niche_vectors = []

	idx = 0

	for user_id in USER_SET:

		pop_vector = USER_POP_VECTOR[user_id]
		niche_vector = USER_NICHE_VECTOR[user_id]

		if int(np.sum(pop_vector)) == 0 or int(np.sum(niche_vector)) == 0:
			discarded_users += 1
			continue

		niche_vector_new = [0.0]*len(NICHE_TAGS)

		SUM_NON_ZERO = 0.0

		for niche_tag in USER_NICHE_TAGS[user_id]:

			max_coeff = -1.0

			for pop_tag in USER_POP_TAGS[user_id]:

				curr_coeff = OVERLAP_COEFFS[niche_tag][pop_tag]

				if curr_coeff > max_coeff:
					max_coeff = curr_coeff

			niche_vector_new[NICHE_TAGS[niche_tag]] = max_coeff

			SUM_NON_ZERO += (max_coeff)

		NUM_ZEROS = len(NICHE_TAGS) - len(USER_POP_TAGS[user_id])

		zero_prob = 0.2/(1.0*NUM_ZEROS)
		non_zero_prob = 0.8/(SUM_NON_ZERO)

		niche_vector_norm = [0.0]*len(NICHE_TAGS)

		for jj in range(len(NICHE_TAGS)):
			if niche_vector_new[jj] == 0.0:
				niche_vector_norm[jj] = zero_prob

			else:
				niche_vector_norm[jj] = non_zero_prob*niche_vector_new[jj]


		idx += 1

		if idx % 10000 == 0:
			print('Loaded Users:', idx)

		popular_vectors.append(pop_vector)
		niche_vectors.append(niche_vector_norm)

	# user_popular_vector_file.close()

	# print('discarded_users:', discarded_users)

	popular_vectors = np.asarray(popular_vectors)
	niche_vectors = np.asarray(niche_vectors)

	return popular_vectors, niche_vectors




def load_train_data(csv_file):
	tp = pd.read_csv(csv_file)
	n_users = tp['uid'].max() + 1

	rows, cols = tp['uid'], tp['sid']

	
	data = sparse.csr_matrix((np.ones_like(rows),
							 (rows, cols)), dtype='float32',
							 shape=(n_users, n_items))

	return data, tp['uid'].min()


def load_tr_te_data(csv_file_tr, csv_file_te):
	tp_tr = pd.read_csv(csv_file_tr)
	tp_te = pd.read_csv(csv_file_te)

	start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
	end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
	# print(start_idx, end_idx)

	rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
	rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

	data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
							 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
	data_te = sparse.csr_matrix((np.ones_like(rows_te),
							 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))


	return data_tr, data_te, start_idx


def load_tag_features():
	# global tag_feature_len

	TAG_FEATURES = {}

	tag_feature_len = 0

	tag_feature_file = codecs.open(tag_feature_vectors_path, 'r', 'utf-8')

	for row in tag_feature_file:
		s = row.strip().split('\t')

		li = eval(s[1])

		try:
			TAG_FEATURES[int(SHOW2ID[s[0]])] = li
		except:
			continue

		tag_feature_len = len(li)

	tag_feature_file.close()

	return TAG_FEATURES, tag_feature_len

def load_one_hot_features():
	# global tag_feature_len
	TAG_FEATURES_ARR = []
	TAG_FEATURES = {}

	tag_feature_len = 0

	tag_feature_file = codecs.open(tag_feature_vectors_path, 'r', 'utf-8')

	for row in tag_feature_file:
		s = row.strip().split('\t')

		li = eval(s[1])

		try:
			curr_li = [0]*n_items
			curr_li[int(SHOW2ID[s[0]])] = 1
			TAG_FEATURES[int(SHOW2ID[s[0]])] = curr_li
		except:
			continue

		tag_feature_len = len(curr_li)

	tag_feature_file.close()
	for id1 in range(len(TAG_FEATURES)):
		try:
			TAG_FEATURES_ARR.append(TAG_FEATURES[id1])
		except:
			print("error")
	return TAG_FEATURES, tag_feature_len, np.array(TAG_FEATURES_ARR)




# Evaluation Functions

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
	'''
	normalized discounted cumulative gain@k for binary relevance
	ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
	'''
	batch_users = X_pred.shape[0]
	idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
	topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
					   idx_topk_part[:, :k]]
	idx_part = np.argsort(-topk_part, axis=1)
	# print('topk-part:', topk_part)
	# print('idx_part:', idx_part)
	# X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
	# topk predicted score
	idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
	# build the discount template
	tp = 1. / np.log2(np.arange(2, k + 2))

	# print('ratings:', heldout_batch[np.arange(batch_users)[:, np.newaxis],
	#                    idx_topk].toarray())

	DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
						 idx_topk].toarray() * tp).sum(axis=1)
	IDCG = np.array([(tp[:min(n, k)]).sum()
					 for n in heldout_batch.getnnz(axis=1)])

	# if IDCG[0] == 0:
	#   return 0

	output = []

	for idx in range(np.shape(DCG)[0]):
		if IDCG[idx] != 0:
			output.append(DCG[idx]/IDCG[idx])
		# else:
		#   output.append(0)

	# print("NDCG",np.shape(DCG), np.shape(IDCG))

	# print("NDCG:",len(output))

	# print(DCG/IDCG)
	return output

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
	batch_users = X_pred.shape[0]

	idx = bn.argpartition(-X_pred, k, axis=1)
	X_pred_binary = np.zeros_like(X_pred, dtype=bool)
	X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

	X_true_binary = (heldout_batch > 0).toarray()

	# print('X_true_binary:', X_true_binary)

	tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
		np.float32)

	# print("Recall", np.shape(tmp), np.shape(np.minimum(k, X_true_binary.sum(axis=1))))

	# if np.minimum(k, X_true_binary.sum(axis=1)) == 0:
	#   return 0

	denom = np.minimum(k, X_true_binary.sum(axis=1))

	output = []

	misclassified_tags = []

	for idx in range(np.shape(tmp)[0]):
		if denom[idx] != 0:
			output.append(tmp[idx]/denom[idx])

			# curr_tags = []
			# for inner_idx in range(len(X_true_binary[idx])):
			#   if X_true_binary[idx][inner_idx] == True and X_pred_binary[idx][inner_idx] == False:
			#       curr_tags.append(inner_idx)

			# misclassified_tags.append(curr_tags)

	# if isinf(recall):
	#   print('inf')
	#   return 0

	# print("Recall:",len(output))

	return output, misclassified_tags


def generator_VAECF(p_dims):

	vae = BaseRecommender(p_dims, lam=0.0, random_seed=98765)

	logits_var, loss_var, params = vae.build_graph()

	return vae, logits_var, loss_var, params


def sample_from_generator(elements, probabilities_li, to_sample, niche_only = False):

	sampled_li_bin = np.zeros([len(elements)], dtype = float)
	probabilities_li = np.asarray(probabilities_li)

	while True:
		try:
			if niche_only == True:
				try:
					probabilities_li[OTHER_TAGS] = 0.0
					if probabilities_li.sum() != 0.0:
						probabilities_li = probabilities_li/(1.0*probabilities_li.sum())
					else:
						probabilities_li = [(1.0/len(elements))]*len(elements)
						probabilities_li = np.asarray(probabilities_li)
				except Exception as e:
					print('Error:', str(e))

			sampled_li = np.random.choice(elements, to_sample, p = probabilities_li, replace = False)

			break
		except:
			# print('Error Sampling: Reducing to_sample')
			to_sample -= 1
			if to_sample == 0:
				break


	# for idx in range(np.shape(probabilities_li)[0]):
	sampled_li_bin[sampled_li] = 1

	return np.asarray(sampled_li_bin), np.asarray(sampled_li)



def sample_from_generator_new(elements, probabilities_li, to_sample, num_elements):

	sampled_li_bin = np.zeros([num_elements], dtype = float)
	probabilities_li = np.asarray(probabilities_li)

	if probabilities_li.sum() != 0.0:
		probabilities_li = probabilities_li/(1.0*probabilities_li.sum())
	else:
		probabilities_li = [(1.0/num_elements)]*num_elements
		probabilities_li = np.asarray(probabilities_li)

	while True:
		try:

			sampled_li = np.random.choice(elements, to_sample, p = probabilities_li, replace = False)

			break
		except:
			# print('Error Sampling: Reducing to_sample')
			to_sample -= 1
			if to_sample == 0:
				break


	# for idx in range(np.shape(probabilities_li)[0]):
	sampled_li_bin[sampled_li] = 1

	return np.asarray(sampled_li_bin), np.asarray(sampled_li)



train_IRGAN()