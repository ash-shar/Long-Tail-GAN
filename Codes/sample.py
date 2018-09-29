import numpy as np



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
