from Base_Recommender.MultiVAE import MultiVAE
import os

def generator_VAECF(pro_dir):

	unique_sid = list()
	with open(os.path.join(pro_dir, 'unique_item_id.txt'), 'r') as f:
		for line in f:
			unique_sid.append(line.strip())

	n_items = len(unique_sid)

	p_dims = [200, 600, n_items] # VAECF recommended values

	total_anneal_steps = 20000 # VAECF recommended values
	anneal_cap = 0.2 # VAECF recommended values

	vae = MultiVAE(p_dims, lam=0.0, random_seed=98765)

	logits_var, loss_var, params = vae.build_graph()

	return vae, logits_var, loss_var, params, p_dims, total_anneal_steps, anneal_cap