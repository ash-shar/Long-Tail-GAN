import tensorflow as tf
import numpy as np
import os
import shutil
import codecs
import psutil
from scipy import sparse
import pandas as pd
import bottleneck as bn

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

	# topk predicted score
	idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
	# build the discount template
	tp = 1. / np.log2(np.arange(2, k + 2))

	DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
						 idx_topk].toarray() * tp).sum(axis=1)
	IDCG = np.array([(tp[:min(n, k)]).sum()
					 for n in heldout_batch.getnnz(axis=1)])

	output = []

	for idx in range(np.shape(DCG)[0]):
		if IDCG[idx] != 0:
			output.append(DCG[idx]/IDCG[idx])

	return output

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
	batch_users = X_pred.shape[0]

	idx = bn.argpartition(-X_pred, k, axis=1)
	X_pred_binary = np.zeros_like(X_pred, dtype=bool)
	X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

	X_true_binary = (heldout_batch > 0).toarray()

	tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
		np.float32)

	denom = np.minimum(k, X_true_binary.sum(axis=1))

	output = []

	misclassified_tags = []

	for idx in range(np.shape(tmp)[0]):
		if denom[idx] != 0:
			output.append(tmp[idx]/denom[idx])

	return output, misclassified_tags
