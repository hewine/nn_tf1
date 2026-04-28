import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import csv

import sys
sys.path.append("../")

from src.data import data_layer_cross
from src.model.model_RtnFcst_try import FeedForwardModelWithNA_Return_Ensembled
from src.utils import deco_print

tf.flags.DEFINE_string('config', '', 'Path to the file with configurations')

FLAGS = tf.flags.FLAGS

def main(_):
	with open(FLAGS.config, 'r') as file:
		config = json.load(file)
	deco_print('Read the following in config: ')
	print(json.dumps(config, indent=4)) 


	plot_idx = '4'
	macro_variable = 'sentiment'
	result = [
            ### flow + fund momentum characteristics + sentiment.
             [list(range(56, 60))+[47], 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'],\
            #### fund excluding momentum and flow. 
             [[59]+ [x for x in range(46, 58) if x not in (list(range(54, 58))+[47])], 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'], \
             ### Stock only.
             [range(46), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'],
             #### Fund only.
             [range(46, 59), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'],
            #### Fund + sentiment.
             [range(46, 60), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'],
             #### Stock + fund.
             [range(59), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'],\
             #### F_r12_2+ sentiment
             [[58, 59], 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'],\
            #### Stock + sentiment. 
            [[59]+list(range(0, 46)), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe'],\
            ### Stock + fund + sentiment.
            [range(60), 1, [2**6], 0.95, 6, 0.0,0.001,0.01, 'Factor_sharpe'],\
            ### F_12_2 + flow + sentiment
              [[47, 58, 59], 1, [2**6], 0.95, 6, 0.0,0.001,0.01, 'Factor_sharpe']]
	data_original = np.load('../datasets/CharAll_na_rm_huge_train_variableall4_sentiment_full_new.npz')
	data_original = data_original['data']    
	final_list = np.load('../sampling_folds/random_sampling_folds.npy', allow_pickle = True)
  
	for k, train_model in enumerate(result):     
		[subset, num, hidden, dropout, max_hidden, l1_penalty, l2_penalty, lr, model_selection] = train_model    
		start_chara = subset[0]
		end_chara = subset[-1]              
		directory_all = []  
		## Variables for saving data.         
		residual_all = np.zeros((data_original.shape[0], data_original.shape[1]))
		mask_all = np.zeros((data_original.shape[0], data_original.shape[1]))  
		R_all = np.zeros((data_original.shape[0], data_original.shape[1])) 
		Rhat_all = np.zeros((data_original.shape[0], data_original.shape[1]))         
		for i in range(3):
			[train_idx_list, valid_idx_list, test_idx_list] = final_list[i]   
			x = np.where(np.sum(data_original[test_idx_list, :,0]!=-99.99, axis = 0)!=0)[0]                         
			deco_print('Creating data layer')
			dl_test = data_layer_cross.DataInRamInputLayer(
				config['individual_feature_file_test'],
			test_idx_list, subset)
			deco_print('Data layer created')
			## Read in the saved             
			directories = ['../output_RF/random_sampling/Train_fold_'+str(idx)+'/fullnew'+ str(start_chara)+ str(end_chara)+str(num)+ str(hidden[0]) + str(dropout)+ str(l1_penalty)+ str(l2_penalty)+ str(lr)+ model_selection+'Test'+str(i) for idx in range(1, 9)]  
			tf.reset_default_graph()
			global_step = tf.train.get_or_create_global_step()
			config['num_layers']  = num      
			config['hidden_dim']  =   hidden
			config['dropout']  =   dropout       
			config['reg_l1'] = l1_penalty
			config['reg_l2'] = l2_penalty
			config['learning_rate'] = lr  
			config['individual_feature_dim'] = len(subset)
			model = FeedForwardModelWithNA_Return_Ensembled(directories, config, 'train', global_step=global_step)
			gpu_options = tf.GPUOptions(allow_growth=True)
			sess_config = tf.ConfigProto(gpu_options=gpu_options)
			sess = tf.Session(config=sess_config)        
			### This is evaluating and saving the key data.             
			R_hat, residual, mask, R= model.calculatenewStatistics(sess, dl_test)
			residual_all[np.ix_(test_idx_list, x)] = residual
			mask_all[np.ix_(test_idx_list, x)] = mask.astype(np.float64)
			R_all[np.ix_(test_idx_list, x)] = R
			Rhat_all[np.ix_(test_idx_list, x)] = R_hat             
			directory_all+= directories       
# 			If the gradient of the model is not generated, generate the gradient.
# 			for directory in directories:
# 				if not os.path.exists(directory + "/ave_absolute_gradient_square.npy"):     
# 					model.plotIndividualFeatureImportance(sess, dl_test, plotPath='plots/')     
# 					break
			## Save the interaction terms.                    
# 			basedir = '../result_saved/random_sampling/Interaction/'
# 			plot_list = ['flow', 'F_r12_2', 'F_ST_Rev', 'Family_r12_2']
# 			if 57 in subset and 47 in subset and 53 in subset and 56 in subset and 58 in subset:        
# 				for num_k, k_name in enumerate(plot_list): 
# 					model.plotConditionalReturn(sess, dl_test, str(plot_idx), 4, k_name, macro_variable, plotPath='plots/', figsize=(8,6), name = str(start_chara)+str(end_chara)+str(len(subset)), cross_idx = i, basedir = basedir)
### If you want to have the 3D plot of the parsimonious model, uncomment and run this. 
### But keep in mind that you should input
## the parsimonious model too (flow + fund momentm + sentiment). 
# 			basedir = '../result_saved/random_sampling/parsimonious/'
# 			model.plot2DConditionalReturn(sess, dl_test, 'flow', 'F_r12_2', 'sentiment', cross_idx = i, basedir = basedir) 
		### Save model's output.                   
		np.savez('../result_saved/random_sampling/output_all_verify_'+str(start_chara)+str(end_chara)+str(len(subset))+'.npz', residual_all = residual_all, R_all = R_all, mask_all = mask_all, Rhat_all = Rhat_all)

if __name__ == '__main__':
	tf.app.run()
