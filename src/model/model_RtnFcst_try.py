import copy
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.layers.core import Dense
from tensorflow.core.framework import summary_pb2

from .model_base import ModelBase
from .model_utils import calculatenewStatistics
from src.utils import deco_print
from src.utils import sharpe
from src.utils import plotWeight1D
from src.utils import plotWeight3D
from src.utils import construct_long_short_portfolio
from src.utils import construct_decile_portfolios
from src.utils import plot_variable_importance
from src.utils import plot_variable_group

class FeedForwardModelWithNA_Return_Ensembled:
	def __init__(self, logdirs, model_params, mode, force_var_reuse=False, global_step=None):
		self._logdirs = logdirs
		self._model = FeedForwardModelWithNA_Return(model_params, mode, force_var_reuse=force_var_reuse, global_step=global_step)
        
	def getPrediction(self, sess, dl):
		pred = []
		for logdir in self._logdirs:
			self._model.loadSavedModel(sess, logdir)
			pred.append(self._model.getPrediction(sess, dl))
		return np.array(pred).mean(axis=0)
    
	def calculatenewStatistics(self, sess, dl):
		w = self.getPrediction(sess, dl)
		return calculatenewStatistics(w, dl)

	def plotIndividualFeatureImportance(self, sess, dl, plotPath=None, top=30, figsize=(8,6), name = '', square = True):
		gradients_list = []
		for logdir in self._logdirs:
			filename = 'ave_absolute_gradient'       
			if square:
				filename += '_square'
			filename += '.npy'   
# 			if not os.path.exists(os.path.join(logdir, filename)):
			self._model.loadSavedModel(sess, logdir)
			self._model._saveIndividualFeatureImportance(sess, dl, logdir)

	def plot2DConditionalReturn(self, sess, dl, name_x, name_y, name_z, xlim=[-0.5,0.5], plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), label='Residual prediction', name = '', cross_idx = 0, logdir = None, basedir = 'output_RF/tune_try/'):

		idx_x = 0
		idx_y = 1
		idx_z = 2       
        
		xlabel = 'flow'
		ylabel = 'F_r12_2'
		zlabel = 'sentiment'

		x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
		y = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
		zs = [-0.35, -0.11,   0.15 ,  0.58 ,  0.922]
		x, y = np.meshgrid(x, y)

		v_list = []
		for logdir in self._logdirs:
			self._model.loadSavedModel(sess, logdir)
			v = np.zeros((len(y), len(x), len(zs)))
			for k in range(len(zs)):
				z = zs[k]
				rFun = self._model.construct2DNonlinearFunction(sess, idx_x, idx_y, idx_z=idx_z, v_z=z)
				for i in range(len(y)):
					for j in range(len(x)):
						v[i,j,k] = rFun(x[i,j], y[i,j])
			v_list.append(v)
		v = np.array(v_list).mean(axis=0)
		if logdir:
			np.save(basedir + 'ave_mean_3d_'+str(cross_idx)+str(name_x)+\
                        str(name_y)+str(name_z)+ '3.npy', v)
            
            
	def plotConditionalReturn(self, sess, dl, plot_name, idx_plot, name_x, name_y, plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), xlim=[-0.5,0.5], label='Residual prediction', name = '', cross_idx = 0, logdir = None, basedir = 'output_RF/tune_try/'):
		meanMacroFeature, stdMacroFeature = dl.getMacroFeatureMeanStd()
 		### Origianlly we also have other interaction measures, but are not included in the paper.        
		if idx_plot == 4:
# 			quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
			if name_x== 'sentiment' or name_y =='sentiment':
# 			These are the 0.1, 0.25, 0.5, 0.75, and 0.9 percentile of sentiment in our sample.        
				quantiles = [-0.35, -0.11,   0.15 ,  0.58 ,  0.922]
			elif name_y=='RecCFNAI' or  name_x=='RecCFNAI' :
				quantiles = [-0.694, -0.29 ,  0.02,   0.31 ,  0.57]              
			idx_x = dl._var2idx[name_x]
			idx_y = dl._var2idx[name_y]
			xlabel = dl.getFeatureByIdx(idx_x)
			ylabel = dl.getFeatureByIdx(idx_y)
			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			v = np.zeros((len(quantiles), len(x)))
			for j in range(len(quantiles)):
				v_list = []
				for logdir in self._logdirs:
					self._model.loadSavedModel(sess, logdir)
					rFun = self._model.construct2DNonlinearFunction(sess, idx_x, idx_y, 
						meanMacroFeature=meanMacroFeature, 
						stdMacroFeature=stdMacroFeature)
					v_j = np.zeros_like(x)
					for i in range(len(x)):
						v_j[i] = rFun(x[i], quantiles[j])
					v_list.append(v_j)
				v[j] = np.array(v_list).mean(axis=0)
			if logdir:
				np.save(basedir + 'ave_mean'+str(cross_idx)+str(name_x)+\
                        str(name_y)+str(self._model._individual_feature_dim)+'.npy', v)

class FeedForwardModelWithNA_Return(ModelBase):
	def __init__(self, model_params, mode, force_var_reuse=False, global_step=None):
		super(FeedForwardModelWithNA_Return, self).__init__(model_params, mode, global_step)
		self._force_var_reuse = force_var_reuse
		self._macro_feature_dim = self.model_params['macro_feature_dim']
		self._individual_feature_dim = self.model_params['individual_feature_dim']
		self._reg_l1 = self.model_params['reg_l1']
		self._reg_l2 = self.model_params['reg_l2']      


		self._I_macro_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self._macro_feature_dim], name='macroFeaturePlaceholder')
		self._I_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, self._individual_feature_dim], name='individualFeaturePlaceholder')
		self._R_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None], name='returnPlaceholder')
		self._mask_placeholder = tf.placeholder(dtype=tf.bool, shape=[None, None], name='maskPlaceholder')
		self._dropout_placeholder = tf.placeholder_with_default(1.0, shape=[], name='Dropout')

		if self.model_params['weighted_loss']:
			self._loss_weight = tf.placeholder(dtype=tf.float32, shape=[None, None], name='weightPlaceholder')

		with tf.variable_scope(name_or_scope='Model_Layer', reuse=self._force_var_reuse):
			self._build_forward_pass_graph()
            
		deco_print('Trainable variables (scope=%s)' %'Model_Layer')        
		total_params = 0        
		trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model_Layer')
		for var in trainable_variables:
			var_params = 1   
			for dim in var.get_shape():
				var_params *= dim.value
			total_params += var_params
			print('Name: {} and shape: {}'.format(var.name, var.get_shape()))
		deco_print('Number of parameters: %d' %total_params)
		if self._mode == 'train':
			self._train_model_op = self._build_train_op(self._loss, scope='Model_Layer', reg_l1 = self._reg_l1, reg_l2 = self._reg_l2)

	def _build_forward_pass_graph(self):
		with tf.variable_scope('NN_Layer'):
			NSize = tf.shape(self._R_placeholder)[1]
			I_macro_tile = tf.tile(tf.expand_dims(self._I_macro_placeholder, axis=1), [1,NSize,1])
			I_macro_masked = tf.boolean_mask(I_macro_tile, mask=self._mask_placeholder)
			I_masked = tf.boolean_mask(self._I_placeholder, mask=self._mask_placeholder)
			I_concat = tf.concat([I_masked, I_macro_masked], axis=1)
			R_masked = tf.boolean_mask(self._R_placeholder, mask=self._mask_placeholder)

			h_l = I_concat
			for l in range(self.model_params['num_layers']):
				with tf.variable_scope('dense_layer_%d' %l):
					layer_l = Dense(units=self.model_params['hidden_dim'][l], activation=tf.nn.relu)
					h_l = layer_l(h_l)
					h_l = tf.nn.dropout(h_l, self._dropout_placeholder)

			with tf.variable_scope('last_dense_layer'):
				layer = Dense(units=1)
				R_pred = layer(h_l)
				self._R_pred = tf.reshape(R_pred, shape=[-1])

		if self.model_params['weighted_loss']:
			loss_weight_masked = tf.boolean_mask(self._loss_weight, mask=self._mask_placeholder)
			loss_weight_masked /= tf.reduce_sum(loss_weight_masked) # normalize weight
			self._loss = tf.reduce_sum(tf.square(R_masked - self._R_pred) * loss_weight_masked)
		else:
			self._loss = tf.reduce_mean(tf.square(R_masked - self._R_pred))

	def train(self, sess, dl, dl_valid, logdir, save_name, loss_weight=None, loss_weight_valid=None, 
			dl_test=None, loss_weight_test=None, 
			printOnConsole=True, printFreq=128, saveLog=True, model_selection = 'Factor_sharpe'):
		saver = tf.train.Saver(max_to_keep=100)
		if saveLog:
			sw = tf.summary.FileWriter(logdir, sess.graph)

		best_valid_sharpe = -1*float('inf')
		best_valid_top = -1*float('inf') 
		best_valid_loss = float('inf') 
		best_valid_mean = -1*float('inf')         
		sharpe_train = []
		sharpe_valid = []        
		### evaluate test data
		evaluate_test_data = False
		if dl_test is not None:
			evaluate_test_data = True
			sharpe_test = []

		time_start = time.time()
		for epoch in range(self.model_params['num_epochs']):
			for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=self.model_params['sub_epoch'])):
				fetches = [self._train_model_op]
				feed_dict = {self._I_macro_placeholder:I_macro,
							self._I_placeholder:I,
							self._R_placeholder:R,
							self._mask_placeholder:mask,
							self._dropout_placeholder:self.model_params['dropout']}
				if self.model_params['weighted_loss']:
					feed_dict[self._loss_weight] = loss_weight
				sess.run(fetches=fetches, feed_dict=feed_dict)

			### evaluate train loss / sharpe
			train_epoch_loss = self.evaluate_loss(sess, dl, loss_weight)
			train_epoch_sharpe = self.evaluate_sharpe(sess, dl)
			sharpe_train.append(train_epoch_sharpe)             

			### evaluate valid loss / sharpe
			valid_epoch_loss = self.evaluate_loss(sess, dl_valid, loss_weight_valid)
			valid_epoch_sharpe = self.evaluate_sharpe(sess, dl_valid)
			valid_epoch_top = self.evaluate_top(sess, dl_valid)
			valid_epoch_mean = self.evaluate_mean(sess, dl_valid)            
			sharpe_valid.append(valid_epoch_sharpe)
            
			### evaluate test loss / sharpe
			if evaluate_test_data:
				test_epoch_loss = self.evaluate_loss(sess, dl_test, loss_weight_test)
				test_epoch_sharpe = self.evaluate_sharpe(sess, dl_test)
				sharpe_test.append(test_epoch_sharpe)

			### print loss / sharpe
			if printOnConsole and epoch % printFreq == 0:
				print('\n\n')
				deco_print('Doing epoch %d' %epoch)
				if evaluate_test_data:
					deco_print('Epoch %d train/valid/test loss: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss, test_epoch_loss))
					deco_print('Epoch %d train/valid/test sharpe: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe, test_epoch_sharpe))
				else:
					deco_print('Epoch %d train/valid loss: %0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss))
					deco_print('Epoch %d train/valid sharpe: %0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe))
			if saveLog:
				value_loss_train = summary_pb2.Summary.Value(tag='Train_epoch_loss', simple_value=train_epoch_loss)
				value_loss_valid = summary_pb2.Summary.Value(tag='Valid_epoch_loss', simple_value=valid_epoch_loss)
				value_sharpe_train = summary_pb2.Summary.Value(tag='Train_epoch_sharpe', simple_value=train_epoch_sharpe)
				value_sharpe_valid = summary_pb2.Summary.Value(tag='Valid_epoch_sharpe', simple_value=valid_epoch_sharpe)
				if evaluate_test_data:
					value_loss_test = summary_pb2.Summary.Value(tag='Test_epoch_loss', simple_value=test_epoch_loss)
					value_sharpe_test = summary_pb2.Summary.Value(tag='Test_epoch_sharpe', simple_value=test_epoch_sharpe)
					summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_loss_test, value_sharpe_train, value_sharpe_valid, value_sharpe_test])
				else:
					summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_sharpe_train, value_sharpe_valid])
				sw.add_summary(summary, global_step=epoch)
				sw.flush()
				

			### save epoch
			if model_selection=='natural':           
				if printOnConsole and epoch % printFreq == 0:
					deco_print('Saving current best checkpoint')
				saver.save(sess, save_path=os.path.join(logdir, 'model-best'))     
			elif model_selection=='Factor_sharpe':    
				if printOnConsole and epoch % printFreq == 0:
					deco_print('Saving current best checkpoint')               
				if valid_epoch_sharpe > best_valid_sharpe:
					best_valid_sharpe = valid_epoch_sharpe
					if printOnConsole and epoch % printFreq == 0:
						deco_print('Saving current best checkpoint')
					saver.save(sess, save_path=os.path.join(logdir, 'model-best'))           
			### time
			if printOnConsole and epoch % printFreq == 0:
				time_elapse = time.time() - time_start
				time_est = time_elapse / (epoch+1) * self.model_params['num_epochs']
				deco_print('Epoch %d Elapse/Estimate: %0.2fs/%0.2fs' %(epoch, time_elapse, time_est))   
		if evaluate_test_data:
			return sharpe_train, sharpe_valid, sharpe_test
		else:
			return sharpe_train, sharpe_valid

	def evaluate_loss(self, sess, dl, loss_weight=None):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			if self.model_params['weighted_loss']:
				feed_dict[self._loss_weight] = loss_weight
			loss, = sess.run([self._loss], feed_dict=feed_dict)
		return loss

	def evaluate_sharpe(self, sess, dl):
		R_pred = self.getPrediction(sess, dl)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			portfolio = construct_long_short_portfolio(R_pred, R[mask], mask, low=0.2, high=0.2) # equally weighted
		return sharpe(portfolio)

	def evaluate_top(self, sess, dl):
		R_pred = self.getPrediction(sess, dl)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			portfolio = construct_decile_portfolios(R_pred, R[mask], mask, decile = 10) # equally weighted        
		return np.mean(portfolio[:,-1])*100


	def evaluate_mean(self, sess, dl):
		R_pred = self.getPrediction(sess, dl)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			portfolio = construct_long_short_portfolio(R_pred, R[mask], mask, low=0.2, high=0.2) # equally weighted
		return np.mean(portfolio) 
    
	def evaluate_factor(self, sess, dl):
		R_pred = self.getPrediction(sess, dl)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			portfolio = construct_long_short_portfolio(R_pred, R[mask], mask, low=0.2, high=0.2)
		return portfolio
    
	def getPrediction(self, sess, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			R_pred, = sess.run(fetches=[self._R_pred], feed_dict=feed_dict)
		return R_pred

	def construct2DNonlinearFunction(self, sess, idx_x, idx_y, idx_z=None, v_z=None, meanMacroFeature=None, stdMacroFeature=None):
		I_macro = np.zeros(shape=(1, self._macro_feature_dim))
		I = np.zeros(shape=(1, 1, self._individual_feature_dim))
		R = np.array([[0.0]], dtype=float)
		mask = np.array([[True]], dtype=bool)
		if idx_z and v_z:
			if idx_z < self._individual_feature_dim:
				I[0,0,idx_z] = v_z
			else:
				idx_macro = idx_z - self._individual_feature_dim
				I_macro[0,idx_macro] = (v_z - meanMacroFeature[idx_macro]) / stdMacroFeature[idx_macro]

		def f(x, y):
			if idx_x < self._individual_feature_dim:
				I[0,0,idx_x] = x
			else:
				idx_macro = idx_x - self._individual_feature_dim
				I_macro[0,idx_macro] = (x - meanMacroFeature[idx_macro]) / stdMacroFeature[idx_macro]
			if idx_y < self._individual_feature_dim:
				I[0,0,idx_y] = y
			else:
				idx_macro = idx_y - self._individual_feature_dim
				I_macro[0,idx_macro] = (y - meanMacroFeature[idx_macro]) / stdMacroFeature[idx_macro]

			feed_dict = {self._I_macro_placeholder:I_macro, 
						self._I_placeholder:I, 
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			R_pred, = sess.run(fetches=[self._R_pred], feed_dict=feed_dict)
			return R_pred[0]
		return f


	def _saveIndividualFeatureImportance(self, sess, dl, logdir = None, delta=1e-6, square = True):
		R_pred = self.getPrediction(sess, dl)
		gradients = np.zeros(shape=(self._individual_feature_dim))

		time_start = time.time()
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			for idx in range(self._individual_feature_dim):
				I_copy = copy.deepcopy(I)
				I_copy[mask, idx] += delta

				feed_dict = {self._I_macro_placeholder:I_macro,
							self._I_placeholder:I_copy,
							self._R_placeholder:R,
							self._mask_placeholder:mask,
							self._dropout_placeholder:1.0}
				R_pred_idx, = sess.run(fetches=[self._R_pred], feed_dict=feed_dict)
				if square:
					gradients[idx] = np.mean((R_pred_idx - R_pred)**2)                    
				else:                    
					gradients[idx] = np.mean(np.absolute(R_pred_idx - R_pred))
				time_last = time.time() - time_start
				time_est = time_last / (idx+1) * self._individual_feature_dim
				deco_print('Calculating VI for %s\tElapse / Estimate: %.2fs / %.2fs' %(dl.getIndividualFeatureByIdx(idx), time_last, time_est))
		if square:
			gradients /= (delta**2)   
		else:
			gradients /= delta
            
		if logdir:
			deco_print('Saving output in %s' %logdir)
			if square:    
				np.save(os.path.join(logdir, 'ave_absolute_gradient_square.npy'), gradients)  
			else:
				np.save(os.path.join(logdir, 'ave_absolute_gradient.npy'), gradients)                  
		return gradients 
    
	def plotIndividualFeatureImportance(self, sess, dl, logdir, plotPath=None, top=30, figsize=(8,6)):
		if not os.path.exists(os.path.join(logdir, 'ave_absolute_gradient.npy')):
			self._saveIndividualFeatureImportance(sess, dl, logdir)