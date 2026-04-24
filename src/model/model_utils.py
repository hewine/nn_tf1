import numpy as np


def decomposeReturn(w, dl):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		R_reshape = R[mask]
		splits = np.sum(mask, axis=1).cumsum()[:-1]
		w_list = np.split(w, splits)
		R_list = np.split(R_reshape, splits)
		R_hat_list = []
		residual_list = []
		for R_i, w_i in zip(R_list, w_list):
			R_hat_i = w_i
			residual_i = R_i - R_hat_i
			R_hat_list.append(R_hat_i)
			residual_list.append(residual_i)
		R_hat = np.zeros_like(mask, dtype=float)
		residual = np.zeros_like(mask, dtype=float)
		R_hat[mask] = np.concatenate(R_hat_list)
		residual[mask] = np.concatenate(residual_list)
	return R_hat, residual, mask, R

def calculatenewStatistics(w, dl, mask_LME=None):
	R_hat, residual, mask, R = decomposeReturn(w, dl)  
	return R_hat, residual, mask, R