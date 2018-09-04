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
h0_size = 100
h1_size = 150
h2_size = 250
h3_size = 300


NUM_EPOCH = 80
NUM_SUB_EPOCHS = 10
BATCH_SIZE = 100

DISPLAY_ITER = 50

LEARNING_RATE = 0.0001

GENERATOR_SAMPLE_TH = 20

total_anneal_steps = 20000
anneal_cap = 0.2

to_restore = False

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


SHOW2ID = {}

show2id_file = codecs.open(show2id_path, 'r', 'utf-8')

for row in show2id_file:
	s = row.strip().split('\t')

	SHOW2ID[s[0]] = s[1]

show2id_file.close()


IDs_present = set()

parsed_tag_vector_file = codecs.open(tag_feature_vectors_path, 'r', 'utf-8')

for row in parsed_tag_vector_file:
	s = row.strip().split('\t')

	try:
		IDs_present.add(SHOW2ID[s[0]])
	except:
		pass

parsed_tag_vector_file.close()


NICHE_TAGS = set()

niche_tags_file = codecs.open(niche_tags_path, 'r', 'utf-8')

tag_idx = 0

for row in niche_tags_file:
	s = row.strip()

	try:
		if SHOW2ID[s] in IDs_present:
			NICHE_TAGS.add(int(SHOW2ID[s]))

	except Exception as e:
		print('Error:', str(e))

niche_tags_file.close()

ALL_TAGS = []

for i in range(n_items):
	ALL_TAGS.append(int(i))

# ALL_TAGS = range(n_items)

OTHER_TAGS_1 = list(set(ALL_TAGS) - set(NICHE_TAGS))

OTHER_TAGS = []
for elem in OTHER_TAGS_1:
	OTHER_TAGS.append(int(elem))

OTHER_TAGS.sort()

OTHER_TAGS = np.asarray(OTHER_TAGS)

print('OTHER_TAGS:', len(OTHER_TAGS), OTHER_TAGS)


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


def load_tag_feature_vectors(csv_file_path, TAG_FEATURES):
	csv_file = codecs.open(csv_file_path, 'r', 'utf-8')

	USER_FEATURES = {}

	idx = 0

	for row in csv_file:
		if idx == 0:
			idx = 1
			continue

		s = row.strip().split(',')

		user_id = int(s[0])
		tag_id = int(s[1])

		if user_id not in USER_FEATURES:
			USER_FEATURES[user_id] = []

		USER_FEATURES[user_id].append(tag_id)

	csv_file.close()

	return USER_FEATURES

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

	vae = MultiVAE.MultiVAE(p_dims, lam=0.0, random_seed=98765)

	saver, logits_var, loss_var, train_op_var, merged_var, params = vae.build_graph()

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

def train_IRGAN():
	# global tag_feature_len
	# TAG_FEATURES, tag_feature_len = load_tag_features()
	TAG_FEATURES, tag_feature_len, TAG_FEATURES_ARR = load_one_hot_features()

	print('TAG features Loaded: ', np.shape(TAG_FEATURES_ARR), tag_feature_len)
	print('Loading Train Data')
	train_data, uid_start_idx = load_train_data(os.path.join(pro_dir, 'train_GAN.csv'))
	print('Train data loaded')
	print('Creating Vad Sets')
	vad_data_tr, vad_data_te, uid_start_idx_vad = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
											   os.path.join(pro_dir, 'validation_te.csv'))
	print('Done 1')
	vad_data_tr_20_50, vad_data_te_20_50, uid_start_idx_vad_20_50 = load_tr_te_data(os.path.join(pro_dir, 'validation_tr_20_50.csv'),
											   os.path.join(pro_dir, 'validation_te_20_50.csv'))
	print('Done 2')
	vad_data_tr_20, vad_data_te_20, uid_start_idx_vad_20 = load_tr_te_data(os.path.join(pro_dir, 'validation_tr_20.csv'),
											   os.path.join(pro_dir, 'validation_te_20.csv'))
	print('Done 3')

	user_popular_data = load_tag_feature_vectors(os.path.join(pro_dir,'train_GAN_popular.csv'), TAG_FEATURES)
	user_niche_data = load_tag_feature_vectors(os.path.join(pro_dir,'train_GAN_niche.csv'), TAG_FEATURES)
	print('Tag feature vectors loaded')
	OVERLAP_COEFFS = load_overlap_coeff()
	print('Overlap Coeffs loaded')

	N = train_data.shape[0]
	idxlist = range(N)



	#changed USER_FEATURES, TAG_FEATURES_ARR
	user_x_niche_vectors = {}
	user_x_popular_n_vectors = {}


	# for user_idx in range(N):
	# 	if user_idx not in user_popular_data or user_idx not in user_niche_data:
	# 		continue

	# 	curr_pop_vectors = user_popular_data[user_idx]
	# 	curr_niche_vectors = user_niche_data[user_idx]

	# 	curr_x_niche = []
	# 	curr_x_popular_n = []

	# 	for niche_tag in curr_niche_vectors:
	# 		niche_tag_idx = niche_tag

	# 		max_coeff = -1.0
	# 		max_pop_tag_idx = -1


	# 		for pop_tag in curr_pop_vectors:
	# 			pop_tag_idx = pop_tag

	# 			curr_coeff = OVERLAP_COEFFS[niche_tag_idx][pop_tag_idx]

	# 			if curr_coeff > max_coeff:
	# 				max_coeff = curr_coeff
	# 				max_pop_tag_idx = pop_tag_idx

	# 		if niche_tag_idx not in TAG_FEATURES or max_pop_tag_idx not in TAG_FEATURES:
	# 			# print('Invalid Niche Tag Pair:', niche_tag_idx, max_pop_tag_idx)
	# 			continue

	# 		curr_x_niche.append(niche_tag_idx)
	# 		curr_x_popular_n.append(max_pop_tag_idx)


	# 	user_x_niche_vectors[user_idx] = curr_x_niche

	# 	user_x_popular_n_vectors[user_idx] = curr_x_popular_n

	# 	# print(curr_x_niche)
	# 	# print(curr_x_popular_n)

	# 	# exit(1)
	# pickle.dump(user_x_niche_vectors, open(pro_dir+'/user_x_niche_vectors.pkl', 'wb'))
	# pickle.dump(user_x_popular_n_vectors, open(pro_dir+'/user_x_popular_n_vectors.pkl', 'wb'))


	user_x_niche_vectors = pickle.load(open(pro_dir+'/user_x_niche_vectors.pkl', 'rb'))
	user_x_popular_n_vectors = pickle.load(open(pro_dir+'/user_x_popular_n_vectors.pkl', 'rb'))


	print('Vectors Loaded')

	print(len(user_x_niche_vectors), len(user_x_popular_n_vectors))


	
	print('User tags to sample being computed')
	USER_TAGS_TO_SAMPLE = {}

	# for user_idx in range(N):
	# 	if user_idx not in user_popular_data or user_idx not in user_niche_data:
	# 		continue

	# 	curr_pop_vectors = user_popular_data[user_idx]
	# 	curr_niche_vectors = user_niche_data[user_idx]

	# 	num_niche_tags = len(curr_niche_vectors)

	# 	num_sample_tags = max(2 * len(curr_niche_vectors), 10 - num_niche_tags)

	# 	curr_niche_tags = set()

	# 	curr_sampling_tags = []

	# 	for niche_tag in curr_niche_vectors:
	# 		niche_tag_idx = niche_tag
	# 		curr_niche_tags.add(niche_tag_idx)
	# 		curr_sampling_tags.append(int(niche_tag_idx))

	# 	other_niche_tags = list(NICHE_TAGS - curr_niche_tags)

	# 	# shuffle(othe)

	# 	other_tags_corr = {}

	# 	for inner_idx in range(len(other_niche_tags)):

	# 		other_tag_idx = other_niche_tags[inner_idx]

	# 		max_coeff = -1.0

	# 		for niche_tag in curr_niche_vectors:
	# 			niche_tag_idx = niche_tag

	# 			curr_coeff = OVERLAP_COEFFS[niche_tag_idx][other_tag_idx]

	# 			if curr_coeff > max_coeff:
	# 				max_coeff = curr_coeff

	# 		other_tags_corr[other_tag_idx] = max_coeff


	# 	sorted_other_tags = sorted(other_tags_corr.items(), key = lambda x: x[1] , reverse = True)

	# 	for inner_idx in range(min(num_sample_tags, len(sorted_other_tags))):
	# 		curr_sampling_tags.append(sorted_other_tags[inner_idx][0])


	# 	curr_sampling_tags.sort()

	# 	# print(curr_sampling_tags)

	# 	USER_TAGS_TO_SAMPLE[user_idx] = np.asarray(curr_sampling_tags)

	# 	# print(curr_sampling_tags, len(curr_niche_vectors), len(curr_sampling_tags))
	# print("User tags to sample done")
	# pickle.dump(USER_TAGS_TO_SAMPLE, open(pro_dir+'/USER_TAGS_TO_SAMPLE.pkl', 'wb'))
	USER_TAGS_TO_SAMPLE = pickle.load(open(pro_dir+'/USER_TAGS_TO_SAMPLE.pkl', 'rb'))


	N_vad = vad_data_tr.shape[0]
	idxlist_vad = range(N_vad)

	N_vad_20_50 = vad_data_tr_20_50.shape[0]
	idxlist_vad_20_50 = range(N_vad_20_50)

	N_vad_20 = vad_data_tr_20.shape[0]
	idxlist_vad_20 = range(N_vad_20)

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

	tag_features_arr = tf.placeholder(tf.float32, [None, tag_feature_len], name="tag_features_arr") # num_tags x ...

	keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout

	emb_matrix = tf.Variable(tf.truncated_normal([tag_feature_len, h0_size], stddev=0.1), name="d_w1", dtype=tf.float32)

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


	d_loss = - tf.reduce_sum(tf.log(y_data)) - tf.reduce_sum(tf.log(1 - y_generated))
	d_loss_mean = tf.reduce_mean(d_loss)
	# g_loss = -tf.reduce_mean(tf.log(self.prob) *self.reward)

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

		# train for one epoch
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

					# for pop_tag in curr_pop_vectors:
					# 	pop_tag_idx = pop_tag[0]

					# 	curr_coeff = OVERLAP_COEFFS[generated_tag_idx][pop_tag_idx]

					# 	if curr_coeff > max_coeff:
					# 		max_coeff = curr_coeff
					# 		max_pop_tag_idx = pop_tag_idx

					if generated_tag_idx not in TAG_FEATURES or max_pop_tag_idx not in TAG_FEATURES:
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


				# curr_x_popular_n = []
				# curr_x_niche = []

				# for tag_idx,jjj in enumerate(curr_x_popular_id_n):
				# 	curr_x_niche.append(TAG_FEATURES[curr_x_niche_id[tag_idx]])
				# 	curr_x_popular_n.append(TAG_FEATURES[curr_x_popular_id_n[tag_idx]])


				# curr_x_popular_n = np.asarray(curr_x_popular_n)
				# curr_x_niche = np.asarray(curr_x_niche)


				# curr_x_popular_g = []
				# curr_x_generated = []

				# for tag_idx,jjj in enumerate(curr_x_popular_id_g):
				# 	curr_x_generated.append(TAG_FEATURES[curr_x_generated_id[tag_idx]])
				# 	curr_x_popular_g.append(TAG_FEATURES[curr_x_popular_id_g[tag_idx]])


				# curr_x_popular_g = np.asarray(curr_x_popular_g)
				# curr_x_generated = np.asarray(curr_x_generated)



				_, curr_d_loss = sess.run([d_trainer, d_loss_mean], feed_dict={generator_network.input_ph: X, x_popular_n_id: curr_x_popular_id_n, x_popular_g_id: curr_x_popular_id_g , x_niche_id: curr_x_niche_id, x_generated_id: curr_x_generated_id, generated_tags: total_sampled_tags, sampled_cnt: total_sampled_cnt, keep_prob: np.sum(0.7).astype(np.float32), tag_features_arr: TAG_FEATURES_ARR})


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


				# curr_x_popular_n = []
				# curr_x_niche = []

				# for tag_idx,jjj in enumerate(curr_x_popular_id_n):
				# 	curr_x_niche.append(TAG_FEATURES[curr_x_niche_id[tag_idx]])
				# 	curr_x_popular_n.append(TAG_FEATURES[curr_x_popular_id_n[tag_idx]])


				# curr_x_popular_n = np.asarray(curr_x_popular_n)
				# curr_x_niche = np.asarray(curr_x_niche)


				# curr_x_popular_g = []
				# curr_x_generated = []

				# for tag_idx,jjj in enumerate(curr_x_popular_id_g):
				# 	curr_x_generated.append(TAG_FEATURES[curr_x_generated_id[tag_idx]])
				# 	curr_x_popular_g.append(TAG_FEATURES[curr_x_popular_id_g[tag_idx]])


				# curr_x_popular_g = np.asarray(curr_x_popular_g)
				# curr_x_generated = np.asarray(curr_x_generated)

				if total_anneal_steps > 0:
						anneal = min(anneal_cap, 1. * ((update_count) / total_anneal_steps))
				else:
					anneal = anneal_cap

				update_count += 1

				_, curr_g_loss, curr_g_loss_term_1, curr_g_loss_term_2 = sess.run([g_trainer, g_loss_mean, g_vae_loss, gan_loss], feed_dict={generator_network.input_ph: X, x_popular_n_id: curr_x_popular_id_n, x_popular_g_id: curr_x_popular_id_g , x_niche_id: curr_x_niche_id, x_generated_id: curr_x_generated_id, generated_tags: total_sampled_tags, sampled_cnt: total_sampled_cnt, generator_network.keep_prob_ph: 0.75, generator_network.is_training_ph: 1, generator_network.anneal_ph: anneal, gen_lambda: curr_gen_lamda, keep_prob: np.sum(0.7).astype(np.float32)})


			print("global-epoch:%s, generator-epoch:%s, g_loss:%.5f (vae_loss: %.5f + gan_loss: %.5f, anneal: %.5f)" % (i, j_gen, curr_g_loss, curr_g_loss_term_1, curr_g_loss_term_2, anneal))


			# print("g-epoch:%s, d-epoch:%s, d_loss:%.5f" % (i, j_disc, curr_d_loss))

		print("")


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

		X_vad_20 = vad_data_tr_20[idxlist_vad_20[0:N_vad_20]]

		if sparse.isspmatrix(X_vad_20):
			X_vad_20 = X_vad_20.toarray()
		X_vad_20 = X_vad_20.astype('float32')
		
		pred_vad_20 = sess.run(generator_out, feed_dict={generator_network.input_ph: X_vad_20} )
		# exclude examples from training and validation (if any)
		pred_vad_20[X_vad_20.nonzero()] = -np.inf
		ndcg_vad = NDCG_binary_at_k_batch(pred_vad_20, vad_data_te_20[idxlist_vad_20[0:N_vad_20]])
		
		recall_at_20, not_found_20 = Recall_at_k_batch(pred_vad_20, vad_data_te_20[idxlist_vad_20[0:N_vad_20]], k=20)

		recall_at_50, not_found_50 = Recall_at_k_batch(pred_vad_20, vad_data_te_20[idxlist_vad_20[0:N_vad_20]], k=50)


		print('global-epoch:', i , 'gen-epoch:', j_gen, 'Vad>=20: NDCG:', np.mean(ndcg_vad), 'Recall@20:', np.mean(recall_at_20), 'Recall@50:', np.mean(recall_at_50), 'Num_users:', len(ndcg_vad), len(recall_at_20), len(recall_at_50))


		X_vad_20_50 = vad_data_tr_20_50[idxlist_vad_20_50[0:N_vad_20_50]]

		if sparse.isspmatrix(X_vad_20_50):
			X_vad_20_50 = X_vad_20_50.toarray()
		X_vad_20_50 = X_vad_20_50.astype('float32')
		
		pred_vad_20_50 = sess.run(generator_out, feed_dict={generator_network.input_ph: X_vad_20_50} )
		# exclude examples from training and validation (if any)
		pred_vad_20_50[X_vad_20_50.nonzero()] = -np.inf
		ndcg_vad = NDCG_binary_at_k_batch(pred_vad_20_50, vad_data_te_20_50[idxlist_vad_20_50[0:N_vad_20_50]])

		recall_at_20, not_found_20 = Recall_at_k_batch(pred_vad_20_50, vad_data_te_20_50[idxlist_vad_20_50[0:N_vad_20_50]], k=20)

		recall_at_50, not_found_50 = Recall_at_k_batch(pred_vad_20_50, vad_data_te_20_50[idxlist_vad_20_50[0:N_vad_20_50]], k=50)


		print('global-epoch:', i , 'gen-epoch:', j_gen,  'Vad-20-50: NDCG:', np.mean(ndcg_vad), 'Recall@20:', np.mean(recall_at_20), 'Recall@50:', np.mean(recall_at_50), 'Num_users:', len(ndcg_vad), len(recall_at_20), len(recall_at_50))


		print('')
			
		# print('')

		# curr_generator_out, curr_disc_data_out, curr_disc_gen_out = sess.run([x_generated, y_data, y_generated], feed_dict={x_popular: POPULAR_VECTORS, x_niche: NICHE_VECTORS, keep_prob: np.sum(0.7).astype(np.float32)})

		# # for idx in range(int(np.shape(NICHE_VECTORS)[0]*0.01)):
		# #     print(USER_SET[idx], file = epoch_out_file)
		# #     print(NICHE_VECTORS[idx].tolist(), file = epoch_out_file)
		# #     print(curr_generator_out[idx].tolist(), file = epoch_out_file)
		# #     print(curr_disc_data_out[idx], file = epoch_out_file)
		# #     print(curr_disc_gen_out[idx], file = epoch_out_file)
		# #     print('', file = epoch_out_file)

		# # epoch_out_file.close()
		
		saver.save(sess, os.path.join(output_path, "model_"+str(i)))

		print('Model saved at global-epoch', i)


def test_IRGAN(epoch_no):
	TAG_FEATURES, tag_feature_len = load_tag_features()

	print('TAG features Loaded: ', len(TAG_FEATURES), tag_feature_len)

	test_data_tr, test_data_te, uid_start_idx = load_tr_te_data(
		os.path.join(pro_dir, 'test_tr.csv'),
		os.path.join(pro_dir, 'test_te.csv'))

	N_test = test_data_tr.shape[0]
	print('N_test:', N_test)

	# print(test_data_tr)
	# print(test_data_te)

	idxlist_test = range(N_test)

	batch_size_test = 20000

	# OVERLAP_COEFFS = load_overlap_coeff()


	generator_network, generator_out, g_vae_loss, g_params = generator_VAECF(p_dims)

	x_generated = tf.placeholder(tf.float32, [None, tag_feature_len], name = "x_generated")

	generated_tags = tf.placeholder(tf.float32, [None, n_items], name = "generated_tags")

	# Discriminator

	x_popular_n = tf.placeholder(tf.float32, [None, tag_feature_len], name="x_popular_n")
	x_popular_g = tf.placeholder(tf.float32, [None, tag_feature_len], name="x_popular_g")
	x_niche = tf.placeholder(tf.float32, [None, tag_feature_len], name="x_niche")

	keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout



	# Popular Tags
	w1 = tf.Variable(tf.truncated_normal([tag_feature_len, h1_size], stddev=0.1), name="d_w1", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([h1_size]), name="d_b1", dtype=tf.float32)
	h1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(x_popular_n, w1) + b1), keep_prob)

	# Niche Tags
	w2 = tf.Variable(tf.truncated_normal([tag_feature_len, h2_size], stddev=0.1), name="d_w2", dtype=tf.float32)
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
	# y_data, d_params = discriminator_QACNN(x_popular, x_niche, keep_prob)

	zero = tf.constant(0, dtype=tf.float32)


	d_loss = - tf.reduce_sum(tf.log(y_data)) - tf.reduce_sum(tf.log(1 - y_generated))
	# g_loss = -tf.reduce_mean(tf.log(self.prob) *self.reward)

	sampled_generator_out = tf.multiply(generator_out, generated_tags)

	sampled_generator_out = tf.reshape(sampled_generator_out, [-1])

	sampled_generator_out_non_zero = tf.gather_nd(sampled_generator_out ,tf.where(tf.not_equal(sampled_generator_out, zero)))

	sampled_cnt = tf.placeholder_with_default(1., shape=None)
	gen_lambda = tf.placeholder_with_default(1.0, shape=None)


	g_loss = g_vae_loss - (1.0 * gen_lambda / sampled_cnt) * tf.reduce_sum(tf.multiply(tf.log(sampled_generator_out_non_zero), tf.log(y_generated)))


	# optimizer : AdamOptimizer
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

	# discriminator and generator loss
	d_trainer = optimizer.minimize(d_loss, var_list=d_params)
	g_trainer = optimizer.minimize(g_loss, var_list=g_params)


	saver = tf.train.Saver()

	sess = tf.Session()

	curr_gen_lamda = GANLAMBDA

	update_count = 0.0

	n100_list, r20_list, r50_list = [], [], []

	user_li = []

	not_found_20_list, not_found_50_list = [], []


	with tf.Session() as sess:
		saver.restore(sess, os.path.join(output_path, "model_"+str(epoch_no)))

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





# test_IRGAN(33)



train_IRGAN()