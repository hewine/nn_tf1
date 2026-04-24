import sys
import numpy as np
import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import pickle
from pathlib import Path
from sklearn import metrics
from scipy.stats import norm
from multiprocessing import Pool, cpu_count

import sys
sys.path.append("../")

from src.data import data_layer_cross
from src.model.model_RtnFcst_try import FeedForwardModelWithNA_Return_Ensembled, FeedForwardModelWithNA_Return
from src.utils import deco_print

mode = 'random_sampling'

def generate_rand_func(Nfuncts, dl_test, config, sess_config):
    variances = np.zeros(Nfuncts)
    absolute_value = np.zeros(Nfuncts)
    for i in range(Nfuncts):
        tf.reset_default_graph()
        global_step = tf.train.get_or_create_global_step()
        model = FeedForwardModelWithNA_Return(config, 'evaluate', global_step=global_step)
        sess = tf.Session(config=sess_config)
        model.randomInitialization(sess)
                
        saver = tf.train.Saver(max_to_keep=100)
        if not os.path.exists('../result_saved/' + mode + '/Simulation/Checkpoints/param_{0}/'):
            os.mkdir('../result_saved/' + mode + '/Simulation/Checkpoints/param_{0}/')
        saver.save(sess, save_path='../result_saved/' + mode + '/Simulation/Checkpoints/param_{0}/param'.format(i)) 
        
#         model.save_weights('./Random_functions_corr/function_{0}'.format(i))
        variances[i] = np.mean(model.getPrediction(sess, dl_test) ** 2)
        absolute_value[i] = np.std(model.getPrediction(sess, dl_test))
        
    with open('../result_saved/' + mode + '/Simulation/variances_{0}'.format(Nfuncts), 'wb') as handle:
        pickle.dump(variances, handle)
    with open('../result_saved/' + mode + '/Simulation/absolute_value_{0}'.format(Nfuncts), 'wb') as handle:
        pickle.dump(absolute_value, handle)        
    
    return variances, absolute_value


def GenerateSamplesRandomFunctions(N, d, nr_normals, Cov, dl_test ):
    # num_run: what is the identification of the     
    # N is the number of simulated gradients. 
    # nr_normals is the dimensional of 
    
    samples_asym_dist = np.zeros((N, d))
    interaction_dist = np.zeros((N, d-1))
    
    for k in range(N):
        normals = np.random.multivariate_normal(mean=np.zeros(nr_normals), cov=Cov)
        max_idx = np.argmax(normals)
        tf.reset_default_graph()
        global_step = tf.train.get_or_create_global_step()  
        model = FeedForwardModelWithNA_Return(config, 'evaluate', global_step=global_step)
        sess = tf.Session(config=sess_config)
        
        model.loadSavedModel(sess, '../result_saved/' + mode + '/Simulation/Checkpoints/param_{0}/'.format(max_idx))
        plot_list  = ['ages','flow','exp_ratio','tna','turnover','Family_TNA','fund_no',\
                      'Family_r12_2','Family_flow','Family_age','F_ST_Rev','F_r2_1','F_r12_2']
        for i, k_name in enumerate(plot_list):                 
            interaction = model.plotConditionalReturn(sess, dl_test, plot_name = '4', idx_plot = 4, name_x = k_name, name_y = 'sentiment', plotPath='plots/', figsize=(8,6), name = '', cross_idx = 0)
            interaction_dist[k, i] = interaction
        model_gradient = model._saveIndividualFeatureImportance(sess, dl_test, logdir = None, delta=1e-6)
        samples_asym_dist[k,:] = model_gradient
        
    with open('../result_saved/' + mode + '/Simulation/samples_gradient_func_test_{0}'.format(N), 'wb') as handle:
        pickle.dump(samples_asym_dist, handle)
    with open('../result_saved/' + mode + '/Simulation/samples_interaction_func_test_{0}'.format(N), 'wb') as handle:
        pickle.dump(interaction_dist, handle)
    return samples_asym_dist, interaction_dist

def simulated_tests(dl_test, config, sess_config, real_magnitude, cross_idx, N_train):
    Nfuncts = 1000
    d = 14
    
    if Path('../result_saved/' + mode + '/Simulation/variances_{0}'.format(Nfuncts)).is_file():
        # print('Read random functions variances')
        with open('../result_saved/' + mode + '/Simulation/variances_{0}'.format(Nfuncts), 'rb') as f:
            variances = pickle.load(f)
        with open('../result_saved/' + mode + '/Simulation/absolute_value_{0}'.format(Nfuncts), 'rb') as f:
            absolute_value = pickle.load(f)
    else:
#         raise ValueError
        print('Generate random functions and corresponding variances')
        variances, absolute_value = generate_rand_func(Nfuncts, dl_test, config, sess_config)

    Cov = np.diag(variances) 

    simulate_magnitude = np.mean(absolute_value)
    N = 1000       
        
    if Path('../result_saved/' + mode + '/Simulation/samples_gradient_func_test_{0}'.format(N)).is_file():
        # print('Read samples from random functions')
        with open('../result_saved/' + mode + '/Simulation/samples_gradient_func_test_{0}'.format(N), 'rb') as f:
            samples = pickle.load(f)
        with open('../result_saved/' + mode + '/Simulation/samples_interaction_func_test_{0}'.format(N), 'rb') as f:
            interaction_samples = pickle.load(f)                  
    else:
#         raise ValueError
        print('Generate samples from random functions')
        samples, interaction_samples = GenerateSamplesRandomFunctions(N, d, Nfuncts, Cov, dl_test)

    # Compute the test statistics. How large is it?
    
    estimation_rate = (N_train / np.log(N_train)) ** ((d + 1) / (2 * (2 * d + 1)))
    
    ### First, calculate the 
    logdirs = ['../output_RF/' + mode+'/Train_fold_'+str(k)+'/fullnew46591640.950.00.0010.01Factor_sharpeTest'+str(cross_idx) for k in range(1,9)]
    gradients_list = []
    for logdir in logdirs:
        gradients_list.append(np.load(os.path.join(logdir, 'ave_absolute_gradient_square.npy')))        
    gradients = np.array(gradients_list).mean(axis=0)
    
    #### Next, calculate the gradient generated by the model.
    plot_list  = ['ages','flow','exp_ratio','tna','turnover','Family_TNA','fund_no',\
                      'Family_r12_2','Family_flow','Family_age','F_ST_Rev','F_r2_1','F_r12_2']
    interaction_real = np.zeros((len(plot_list)))
    v_list = []        
    for i, name_x in enumerate(plot_list):  
        v_new = np.load('../result_saved/' + mode + '/Interaction/ave_mean'+str(cross_idx)+ str(name_x)+'sentiment14.npy') 
        interaction_real[i] = (v_new[4,-1] - v_new[4,0]) - (v_new[0,-1] - v_new[0,0])  
    
    test_stats = samples* real_magnitude**2/(simulate_magnitude**2 * (estimation_rate**2))
    interaction_stats =  interaction_samples * real_magnitude/ (simulate_magnitude* estimation_rate)
    
    return gradients, interaction_real, test_stats, interaction_stats, plot_list


def average_over_fold(samples, interaction_samples, gradients, interaction_real):
    
    q_lower = np.percentile(samples, 90, axis = 0)
    q = np.percentile(samples, 95, axis = 0)
    q_higher = np.percentile(samples, 99, axis = 0)
    
    
    q_interaction_lower = np.percentile(interaction_samples, 90, axis = 0)
    q_interaction = np.percentile(interaction_samples, 95, axis = 0)
    q_higher_interaction = np.percentile(interaction_samples, 99, axis = 0)

    return (gradients> q_lower).astype(int), (gradients>q).astype(int), (gradients>q_higher).astype(int), (interaction_real>q_interaction).astype(int), (interaction_real>q_interaction).astype(int), (interaction_real> q_higher_interaction).astype(int)


   

def print_significance(sig_90, sig_95, sig_99, sig_inter_90, sig_inter_95, sig_inter_99, gradients, interactions, plot_list, print_order):
    N = len(sig_90)
    
    for idx in range(N):
        i = print_order[idx]
        string = '\_'.join(plot_list[i].split('_'))
        if sig_99[i]:
            string+='&%.2f***'%(np.sqrt(gradients[i])*100)
        elif sig_95[i]:
            string+='&%.2f**'%(np.sqrt(gradients[i])*100)
        elif sig_90[i]:
            string+= '&%.2f*'%(np.sqrt(gradients[i])*100)
        else:
            string+= '&%.2f'%(np.sqrt(gradients[i])*100)
        
        if i<N-1:
            if sig_inter_99[i]:
                string+= '&%.2f***'%(interactions[i]*100)
            elif sig_inter_95[i]:
                string+= '&%.2f**'%(interactions[i]*100)
            elif sig_inter_90[i]:
                string+= '&%.2f*'%(interactions[i]*100)
            else:
                string+= '&%.2f'%(interactions[i]*100)
        
        string+='\\\\'
            
        print (string)
    
    
if __name__ == "__main__":
    with open('../Training/config.json', 'r') as file:
        config = json.load(file)

    subset = range(46, 60)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
        
    final_list = np.load('../sampling_folds/random_sampling_folds.npy', allow_pickle = True)    
    real_magnitude_lists = []
    
    gradients_list, interaction_list, simu_grad_list, simu_interaction_list = [], [], [], []
    
    data_original = np.load('../datasets/CharAll_na_rm_huge_train_variableall4_sentiment_full_new.npz')
    data_original = data_original['data']  
    
    for cross_idx in range(3):
        [train_idx_list, valid_idx_list, test_idx_list] = final_list[cross_idx]
        dl_test = data_layer_cross.DataInRamInputLayer(config['individual_feature_file_test'], test_idx_list, subset)
    
        directories = ['../output_RF/' + mode + '/Train_fold_'+str(idx)+'/fullnew46591640.950.00.0010.01Factor_sharpeTest'+str(cross_idx) for idx in range(1, 9)]
        tf.reset_default_graph()
        global_step = tf.train.get_or_create_global_step()
        model = FeedForwardModelWithNA_Return_Ensembled(directories, config, 'train', global_step=global_step)
        sess = tf.Session(config=sess_config)
    
        real_magnitude_list = []
    
        for logdir in directories:
            model._model.loadSavedModel(sess, logdir)
            model_prediction = model._model.getPrediction(sess, dl_test)
        
            real_magnitude_list.append(np.std(model_prediction))
        
        real_magnitude = np.mean(real_magnitude_list)
        
        
            
        
        alpha_part  = data_original[test_idx_list, :,0]
        N_train = np.sum((alpha_part!=-99.99))
        
        if not os.path.exists('../result_saved/' + mode + '/Simulation/real_magnitude_' + str(cross_idx) + '.npz'):
            np.savez('../result_saved/' + mode + '/Simulation/real_magnitude_' + str(cross_idx) + '.npz', N_train = N_train, real_magnitude = real_magnitude)
    
        gradients, interaction_real, simu_grad, simu_interaction, plot_list = simulated_tests(dl_test, config, sess_config, real_magnitude, cross_idx, N_train)
        
        gradients_list.append(gradients), interaction_list.append(interaction_real), simu_grad_list.append(simu_grad), simu_interaction_list.append(simu_interaction) 
        
    plot_list+=['sentiment']
    
    avg_gradient, avg_interaction, avg_grad_stats, avg_interaction_stats = np.mean(gradients_list, axis = 0), np.mean(interaction_list, axis = 0), np.mean(simu_grad_list, axis = 0), np.mean(simu_interaction_list, axis = 0)
    
    sig_gradient_90, sig_gradient_95, sig_gradient_99, sig_interaction_90, sig_interaction_95, sig_interaction_99 = average_over_fold(np.abs(avg_grad_stats), np.abs(avg_interaction_stats), np.abs(avg_gradient), np.abs(avg_interaction))

    print_order = np.argsort(avg_gradient)[::-1]    

    print_significance(sig_gradient_90, sig_gradient_95, sig_gradient_99, sig_interaction_90, sig_interaction_95, sig_interaction_99, avg_gradient, avg_interaction, plot_list, print_order)
    
    





