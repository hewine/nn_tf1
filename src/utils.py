import os
import time
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from scipy.stats import f
import csv

sns.set_style("white")


def squeeze_data(data, UNK = -99.99):
	T, N, M = data.shape
	lists_considered = []    
	returns = data[:,:,0]    
	for i in range(N):      
		returns_i = returns[:,i]             
		if np.sum(returns_i!=UNK)>0:        
			lists_considered.append(i)         
	return data[:, lists_considered, :], lists_considered

def deco_print(line, end='\n'):
	print('>==================> ' + line, end=end)
    
def sharpe(r):
	return np.mean(r / r.std())

def construct_decile_portfolios(w, R, mask, value=None, decile=10):
	N_i = np.sum(mask.astype(int), axis=1)
	N_i_cumsum = np.cumsum(N_i)
	w_split = np.split(w, N_i_cumsum)[:-1]
	R_split = np.split(R, N_i_cumsum)[:-1]

	# value weighted
	value_weighted = False
	if value is not None:
		value_weighted = True
		value = value[mask]
		value_split = np.split(value, N_i_cumsum)[:-1]

	portfolio_returns = []

	for j in range(mask.shape[0]):
		R_j = R_split[j]
		w_j = w_split[j]
		if value_weighted:
			value_j = value_split[j]
			R_w_j = [(R_j[k], w_j[k], value_j[k]) for k in range(N_i[j])]
		else:
			R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])]
		R_w_j_sorted = sorted(R_w_j, key=lambda t:t[1])
		n_decile = N_i[j] // decile
		R_decile = []
		for i in range(decile):
			R_decile_i = 0.0
			value_sum_i = 0.0
			for k in range(n_decile):
				R_decile_i += R_w_j_sorted[i*n_decile+k][0] * R_w_j_sorted[i*n_decile+k][2]
				value_sum_i += R_w_j_sorted[i*n_decile+k][2]
			R_decile.append(R_decile_i / value_sum_i)
		portfolio_returns.append(R_decile)
	return np.array(portfolio_returns)

def construct_long_short_portfolio(w, R, mask, value=None, low=0.1, high=0.1, normalize=True):
	# use masked R and value
	N_i = np.sum(mask.astype(int), axis=1)
	N_i_cumsum = np.cumsum(N_i)
	w_split = np.split(w, N_i_cumsum)[:-1]
	R_split = np.split(R, N_i_cumsum)[:-1]

	# value weighted
	value_weighted = False
	if value is not None:
		value_weighted = True
		value_split = np.split(value, N_i_cumsum)[:-1]

	portfolio_returns = []

	for j in range(mask.shape[0]):
		R_j = R_split[j]
		w_j = w_split[j]
		if value_weighted:
			value_j = value_split[j]
			R_w_j = [(R_j[k], w_j[k], value_j[k]) for k in range(N_i[j])]
		else:
			R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])]
		R_w_j_sorted = sorted(R_w_j, key=lambda t:t[1])
		n_low = int(low * N_i[j])
		n_high = int(high * N_i[j])

		if n_high == 0.0:
			portfolio_return_high = 0.0
		else:
			portfolio_return_high = 0.0
			value_sum_high = 0.0
			for k in range(n_high):
				portfolio_return_high += R_w_j_sorted[-k-1][0] * R_w_j_sorted[-k-1][2]
				value_sum_high += R_w_j_sorted[-k-1][2]
			if normalize:
				portfolio_return_high /= value_sum_high

		if n_low == 0:
			portfolio_return_low = 0.0
		else:
			portfolio_return_low = 0.0
			value_sum_low = 0.0
			for k in range(n_low):
				portfolio_return_low += R_w_j_sorted[k][0] * R_w_j_sorted[k][2]
				value_sum_low += R_w_j_sorted[k][2]
			if normalize:
				portfolio_return_low /= value_sum_low
		if np.isnan(portfolio_return_high) or np.isnan(portfolio_return_low) or np.isinf(portfolio_return_high) or np.isinf(portfolio_return_low):
			print (portfolio_return_high)
			print (portfolio_return_low)                 

		portfolio_returns.append(portfolio_return_high - portfolio_return_low)
	return np.array(portfolio_returns)

def plot_variable_importance(var, imp, labelColor, plotPath=None, normalize=True, top=30, figsize=(8,6), color2category=None, name = ''):
	if normalize:
		if min(imp) >= 0:
			imp = imp / sum(imp)
		else:
			deco_print('WARNING: Unable to normalize due to negative importance value! ')
	top = np.minimum(top, len(var))
	var_imp = list(zip(var, imp, labelColor))
	var_imp_sort = sorted(var_imp, key=lambda t:-t[1])
	var_top = [var_imp_sort[i][0] for i in range(top)]
	imp_top = [var_imp_sort[i][1] for i in range(top)]
	color_top = [var_imp_sort[i][2] for i in range(top)]
	y_pos = np.arange(top)
	if not color2category:
		fig, ax = plt.subplots(figsize=figsize)
		ax.barh(y_pos, imp_top, align='center', color='blue')
	else:
		fig, (ax, ax_cb) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[19,1]})
		ax.barh(y_pos, imp_top, align='center', color=color_top)
	ax.set_yticks(y_pos)
	ax.set_yticklabels(var_top, fontweight='bold')
	ticklabels = ax.yaxis.get_ticklabels()
	for i in range(len(ticklabels)):
		ticklabels[i].set_color(color_top[i])
	ax.invert_yaxis()
	if color2category:
		color_list = list(color2category.keys())
		nColors = len(color_list)
		ticks = [color2category[color] for color in color_list]
		cmap = colors.ListedColormap(color_list)
		cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap)
		cb.set_ticks(np.linspace(0,1,nColors+1)[:-1] + 1./nColors/2)
		cb.set_ticklabels(ticks)
		cb.set_label('Category')
	if plotPath:
		plt.savefig(os.path.join(plotPath, name+'.png' ), bbox_inches='tight')
    
def plotIndividualFeatureImportance_cross(dl, logdirs, plotPath=None, top=30, figsize=(8,6), name = ''):
	gradients_list = []
	for logdir in logdirs:
		gradients_list.append(np.load(os.path.join(logdir, 'ave_absolute_gradient_square.npy')))
	gradients = np.array(gradients_list).mean(axis=0)
	gradients = np.sqrt(gradients)

	gradients_sorted = sorted([(idx, gradients[idx], dl.getIndividualFeatureByIdx(idx)) for idx in range(len(gradients))], key=lambda t:-t[1])       
	imp = [item for _, item, _ in gradients_sorted]
	var = [item for _, _, item in gradients_sorted]
	var2color, color2category = dl.getIndividualFeatureColarLabelMap()
	labelColor = [var2color[item] for item in var]
	plot_variable_importance(var, imp, labelColor, plotPath=plotPath, top=top, figsize=figsize, color2category=color2category, name = name)
    
    
def plotIndividualFeatureImportance_cross_group(dl, logdirs, plotPath=None, top=30, figsize=(8,6), name = ''):
	gradients_list = []
	for logdir in logdirs:
		gradients_list.append(np.load(os.path.join(logdir, 'ave_absolute_gradient_square.npy')))
	gradients = np.array(gradients_list).mean(axis=0)
	gradients = np.sqrt(gradients)

	gradients_sorted = sorted([(idx, gradients[idx], dl.getIndividualFeatureByIdx(idx)) for idx in range(len(gradients))], key=lambda t:-t[1])  
	imp = [item for _, item, _ in gradients_sorted]
	var = [item for _, _, item in gradients_sorted]
	var2color, color2category = dl.getIndividualFeatureColarLabelMap()
	labelColor = [var2color[item] for item in var]
	plot_variable_group(var, imp, labelColor, plotPath=plotPath, top=top, figsize=figsize, color2category=color2category, name = name) 
        
def plotconditionalmean_cross(dl, plot_name, name_x, name_y, plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), xlim=[-0.5,0.5], label='Abnormal return prediction', name = '', length = 14, cross_idx_num = 3, legend = True, base_dir = ""):    
	# First load the x and vs. Then take average. 
	v_list = []        
	for cross_idx in range(cross_idx_num):
		v = np.load(base_dir + 'ave_mean'+str(cross_idx)+str(name_x)+str(name_y)+str(length)+'.npy')                
		v_list.append(v)         
	v = np.array(v_list).mean(axis=0)
	x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
	idx_x = dl._var2idx[name_x]
	idx_y = dl._var2idx[name_y]
	xlabel = dl.getFeatureByIdx(idx_x)
	ylabel = dl.getFeatureByIdx(idx_y)   
	plotWeight1D(x, v*100, xlabel, str(plot_name), ylabel= name_y, idx_y=idx_y, plotPath=plotPath, idx=idx_x, figsize=figsize, label=label, name = name, legend = legend)
    
def plot_variable_group(var, imp, labelColor, plotPath=None, normalize=True, top=30, figsize=(8,6), color2category=None, name = ''):
	if normalize:
		if min(imp) >= 0:
			imp = imp / sum(imp)
		else:
			deco_print('WARNING: Unable to normalize due to negative importance value! ')
	var_imp = list(zip(var, imp, labelColor))
    
	var_group = []
	imp_group = []
	color_group = []    
	for color in color2category.keys():       
		var_sub = [i for i in range(len(var_imp)) if var_imp[i][2]==color]       
		if len(var_sub)==0:
			continue
		var_group.append(color2category[color])       
		imp_group.append(np.mean(np.asarray([var_imp[i][1] for i in var_sub])))
		color_group.append(color)  
	imp_group = imp_group/sum(imp_group)        
	var_imp = list(zip(var_group, imp_group, color_group))
	var_imp_sort = sorted(var_imp, key=lambda t:-t[1])
	var_top = [var_imp_sort[i][0] for i in range(len(var_imp_sort))]
	imp_top = [var_imp_sort[i][1] for i in range(len(var_imp_sort))]   
	color_top = [var_imp_sort[i][2] for i in range(len(var_imp_sort))]
	y_pos = np.arange(len(var_imp_sort))
	if not color2category:
		fig, ax = plt.subplots(figsize=figsize)
		ax.barh(y_pos, imp_top, align='center', color='blue')
	else:        
		fig, (ax, ax_cb) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[19,1]})
		ax.barh(y_pos, imp_top, align='center', color=color_top)
	ax.set_yticks(y_pos)
	ax.set_yticklabels(var_top, fontweight='bold')
	ticklabels = ax.yaxis.get_ticklabels()
	for i in range(len(ticklabels)):
		ticklabels[i].set_color(color_top[i])
	ax.invert_yaxis()
	if color2category:
		color_list = list(color2category.keys())
		nColors = len(color_list)
		ticks = [color2category[color] for color in color_list]
		cmap = colors.ListedColormap(color_list)
		cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap)
		cb.set_ticks(np.linspace(0,1,nColors+1)[:-1] + 1./nColors/2)
		cb.set_ticklabels(ticks)
		cb.set_label('Category')
	if plotPath:
		plt.savefig(os.path.join(plotPath, name+'.png'), bbox_inches='tight')

def plotWeight1D(x, v, xlabel, plot_name, 
	ylabel=None, idx_y=None, zlabel=None, idx_z=None, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], 
	plotPath=None, idx=None, figsize=(8,6), label='Abnormal return Prediction', name = '', legend  = True):
	colors = ['b', 'g', 'r', 'c', 'm'] 
	plt.tick_params(labelsize=20)
	fig = plt.figure(figsize=figsize)

	if ylabel is None:
		plt.scatter(x, v)
	else:
		if zlabel is None:
			for i in range(len(quantiles)):
				plt.scatter(x, v[i], c = colors[i], label='%s %d%%' %(ylabel, int(quantiles[i] * 100)))
			if legend:                
				plt.legend()
		else:
			for i in range(len(quantiles)):
				for j in range(len(quantiles)):
					plt.scatter(x, v[i,j], label='%s %d%%,%s %d%%' %(ylabel, int(quantiles[i]*100), zlabel, int(quantiles[j]*100)))
			plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=True)
	dy = 0.1 * (v.max() - v.min())
	plt.ylim(v.min()-dy, v.max()+dy)
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(label, fontsize=18)
	plt.tight_layout()
	if plotPath:
		if idx_y is None:
			plt.savefig(os.path.join(plotPath, 'w_x_%d.pdf' %idx), bbox_inches='tight')
			plt.savefig(os.path.join(plotPath, 'w_x_%d.png' %idx), bbox_inches='tight')
		else:
			if idx_z is None:
				plt.savefig(os.path.join(plotPath, name+'.png' ), bbox_inches='tight')
			else:
				plt.savefig(os.path.join(plotPath, name+'.png' ), bbox_inches='tight')

def plotcontourmean_cross(dl, name_x, name_y, name_z, plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), xlim=[-0.5,0.5], label='Abnormal return prediction', name = '', length = 14, cross_idx_num = 3, legend = True):    
	# First load the x and vs. Then take average. 
	v_list = []        
	for cross_idx in range(cross_idx_num):
		v = np.load('output_RF/tune_try/ave_mean_3d_'+str(cross_idx)+str(name_x)+str(name_y)+str(name_z)+'3.npy')       
		v_list.append(v)         
	v = np.array(v_list).mean(axis=0)
	x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
	y = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1) 
	zs = [-0.35, -0.11,   0.15 ,  0.58 ,  0.922]  
	xlabel = 'flow'
	ylabel = 'F_r12_2'
	zlabel = 'sentiment'      
	plotWeight3D(x, y, v*100, zs, xlabel, ylabel, zlabel, plotPath=plotPath, idx_x=0, idx_y=1, idx_z=2, figsize=figsize, label=label)
    
    
    
def plotWeight3D(x, y, v, zs, xlabel, ylabel, zlabel, plotPath=None, idx_x=None, idx_y=None, idx_z=None, figsize=(8,6), label='weight'):
	parameters = {'axes.labelsize': 15,
          'legend.fontsize': 12,
           'xtick.labelsize': 11,
            'ytick.labelsize':11}
	plt.rcParams.update(parameters)
	fig = plt.figure(figsize=figsize)
	ax = plt.axes(projection='3d')
	levels = np.linspace(np.min(v),np.max(v), 51)
	for k in range(len(zs)):
		z = zs[k]
		im = ax.contourf(x, y, v[:,:,k], offset=z, levels=levels, cmap=cm.magma_r)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	ax.set_zlim(zs[0], zs[-1])
	cbar = plt.colorbar(im, format='%.2f')
	cbar.ax.set_ylabel(label)
	plt.tight_layout()    
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_z_%d.pdf' %(idx_x, idx_y, idx_z)))
		plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_z_%d.png' %(idx_x, idx_y, idx_z)))
	plt.show()