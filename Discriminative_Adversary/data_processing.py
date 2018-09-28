import pandas as pd
from scipy import sparse


def load_train_data(csv_file, n_items):
	tp = pd.read_csv(csv_file)
	n_users = tp['uid'].max() + 1

	rows, cols = tp['uid'], tp['sid']

	
	data = sparse.csr_matrix((np.ones_like(rows),
							 (rows, cols)), dtype='float32',
							 shape=(n_users, n_items))

	return data, tp['uid'].min()


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
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


def load_item_one_hot_features(item_list_path, SHOW2ID, n_items):
	# global FEATURE_LEN
	ITEM_OH_ARR = []
	ITEM_OH_DICT = {}

	FEATURE_LEN = 0

	tag_feature_file = codecs.open(item_list_path, 'r', 'utf-8')

	for row in tag_feature_file:
		s = row.strip().split('\t')

		# li = eval(s[1])

		try:
			curr_li = [0]*n_items
			curr_li[int(SHOW2ID[s[0]])] = 1
			ITEM_OH_DICT[int(SHOW2ID[s[0]])] = curr_li
		except:
			continue

		FEATURE_LEN = len(curr_li)

	tag_feature_file.close()
	for id1 in range(len(ITEM_OH_DICT)):
		try:
			ITEM_OH_ARR.append(ITEM_OH_DICT[id1])
		except:
			continue
			# print("error")
	return ITEM_OH_DICT, FEATURE_LEN, np.array(ITEM_OH_ARR)

def load_user_items(csv_file_path):
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



def overlap_cofficient(x,y):
	num = len(list(x & y))

	denom = min(len(x),len(y))

	overlap = (num*1.0)/(1.0*denom)

	return overlap


def load_overlap_coeff(show2id_path, user_tag_matrix_path):

	SHOW2ID = {}

	show2id_file = codecs.open(show2id_path, 'r', 'utf-8')

	for row in show2id_file:
		s = row.strip().split('\t')

		SHOW2ID[s[0]] = s[1]


	show2id_file.close()


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
		try:
			tag_id = SHOW2ID[s[1]]
		except:
			print('Error:', tag_id)
			continue

		if tag_id not in TAG_SETS:
			TAG_SETS[tag_id] = set()

		TAG_SETS[tag_id].add(user_id)

		TAG_IDs.add(tag_id)

	user_tag_matrix_file.close()

	print('TAG SETS:', len(TAG_SETS), len(TAG_IDs))

	TAG_IDs = list(TAG_IDs)

	OVERLAP_COEFFS = {}

	for idx in range(len(TAG_IDs)):
		OVERLAP_COEFFS[int(TAG_IDs[idx])] = {}
		for inner_idx in range(len(TAG_IDs)):
			OVERLAP_COEFFS[int(TAG_IDs[idx])][int(TAG_IDs[inner_idx])] = overlap_cofficient(TAG_SETS[TAG_IDs[idx]], TAG_SETS[TAG_IDs[inner_idx]])

	# OVERLAP_COEFFS = pickle.load(open(pro_dir+'/OVERLAP_COEFFS.pkl', 'rb'))
	return OVERLAP_COEFFS


def load_pop_niche_tags(show2id_path, item_list_path, niche_tags_path, n_items):
	SHOW2ID = {}

	show2id_file = codecs.open(show2id_path, 'r', 'utf-8')

	for row in show2id_file:
		s = row.strip().split('\t')

		SHOW2ID[s[0]] = s[1]

	show2id_file.close()


	IDs_present = set()

	parsed_tag_vector_file = codecs.open(item_list_path, 'r', 'utf-8')

	for row in parsed_tag_vector_file:
		s = row.strip()#.split('\t')

		try:
			IDs_present.add(SHOW2ID[s)
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

	return SHOW2ID, IDs_present, NICHE_TAGS, ALL_TAGS, OTHER_TAGS