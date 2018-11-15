import tensorflow as tf

def discriminator(n_items, FEATURE_LEN, h0_size, h1_size, h2_size, h3_size):
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


	return y_data, y_generated, d_params, x_generated_id, x_popular_n_id, x_popular_g_id, x_niche_id, item_feature_arr, keep_prob
