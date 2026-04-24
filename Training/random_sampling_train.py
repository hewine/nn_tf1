import os
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import multiprocessing as mp
import matplotlib.pyplot as plt
import csv
import sys

sys.path.append("../")

from src.data import data_layer_cross
from src.model.model_RtnFcst_try import FeedForwardModelWithNA_Return
from src.utils import deco_print

tf.flags.DEFINE_string('config', '', 'Path to the file with configurations')
tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_string('max_num_process', '', 'Max number of processes to run')

tf.flags.DEFINE_boolean('printOnConsole', True, 'Print on console or not')
tf.flags.DEFINE_boolean('saveLog', True, 'Save log or not')
tf.flags.DEFINE_integer('printFreq', 128, 'Frequency to print on console')

FLAGS = tf.flags.FLAGS

def run_code(train_lists):
	for train_model in train_lists:          
		[subset, num_layers, hidden_dim, dropout, max_hidden, l1_penalty, l2_penalty, lr, model_selection, folder_idx] = train_model 
		start_chara = subset[0]
		end_chara = subset[-1]        
		with open(FLAGS.config, 'r') as file:
			config = json.load(file)
		config['num_layers']  = num_layers        
		config['hidden_dim']  =   hidden_dim
		config['dropout']  =   dropout       
		config['reg_l1'] = l1_penalty
		config['reg_l2'] = l2_penalty
		config['learning_rate'] = lr
		config['individual_feature_dim'] = len(subset)
		config['save_name'] = 'beta_sentiment_4'  
		deco_print('Read the following in config: ')
		print(json.dumps(config, indent=4))    
		final_list = np.load('../sampling_folds/random_sampling_folds.npy', allow_pickle = True)
		for i in range(3): 
			[train_idx_list, valid_idx_list, test_idx_list] = final_list[i]  
			deco_print('Creating data layer')
			logdir = FLAGS.logdir+"/Users_Train/random_sampling/Train_fold_" + str(folder_idx)+'/fullnew' + str(start_chara)+ str(end_chara)+ str(num_layers)+ str(hidden_dim[0]) + str(dropout)+ str(l1_penalty)+ str(l2_penalty)+ str(lr)+ model_selection+ 'Test'+str(i) 
			dl = data_layer_cross.DataInRamInputLayer(
				config['individual_feature_file'],train_idx_list , subset)
			dl_valid = data_layer_cross.DataInRamInputLayer(
				config['individual_feature_file_valid'], valid_idx_list, subset)
			dl_test = data_layer_cross.DataInRamInputLayer(
				config['individual_feature_file_test'], test_idx_list, subset)
			if config['weighted_loss']:
				loss_weight = dl.getDateCountList()
				loss_weight_valid = dl_valid.getDateCountList()
				loss_weight_test = dl_test.getDateCountList()
			else:
				loss_weight = None
				loss_weight_valid = None
				loss_weight_test = None
			deco_print('Data layer created')

			tf.reset_default_graph()
			global_step = tf.train.get_or_create_global_step()
			model = FeedForwardModelWithNA_Return(config, 'train', global_step=global_step)
			gpu_options = tf.GPUOptions(allow_growth=True)
			sess_config = tf.ConfigProto(gpu_options=gpu_options)
			sess = tf.Session(config=sess_config)
			model.randomInitialization(sess)

			sharpe_train, sharpe_valid, sharpe_test = model.train(sess, dl, dl_valid, logdir, config['save_name'],  
				loss_weight=loss_weight, loss_weight_valid=loss_weight_valid, 
				dl_test=dl_test, loss_weight_test=loss_weight_test, 
				printOnConsole=FLAGS.printOnConsole, printFreq=FLAGS.printFreq, saveLog=FLAGS.saveLog, model_selection = model_selection)
            
            
def get_tuned_network():
    temp_results  = [
            #### Fund + sentiment.
             [range(46, 60), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'natural'],
    ]
 
    result = []
    for temp_result in temp_results:
        for j in range(1,9):
            result.append(temp_result+[j])
    print (result)    
    return result
    

def main(_):
    lst_t_sample = get_tuned_network()
    if int(FLAGS.max_num_process)>0:
        mp.set_start_method("spawn")
        pool = mp.Pool(processes=int(FLAGS.max_num_process))
    
        
        num_t_each_trunk= 1
    
        num_trunk=int(np.ceil(len(lst_t_sample)/num_t_each_trunk))
        
        for i in range(0,num_trunk):
            lst_ind= np.arange(i*num_t_each_trunk,np.min([(i+1)*num_t_each_trunk,len(lst_t_sample)]))
            lst_t= [lst_t_sample[k] for k in lst_ind]
            pool.apply_async(run_code, args=(lst_t,)) 
        
        pool.close()
        pool.join() 
    else:
        run_code(lst_t_sample)


if __name__ == '__main__':
	tf.app.run()