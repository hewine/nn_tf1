import numpy as np

from src.utils import deco_print, squeeze_data

class FirmChar:
	def __init__(self):
		self._category = ['Fund mom','Fund char', 'Fund Family', 'Sentiment']
		self._category2variables = {
			'Fund mom': ['F_ST_Rev', 'F_r2_1', 'F_r12_2'],
			'Fund char': ['ages', 'flow', 'exp_ratio', 'tna', 'turnover'],
			'Fund Family': ['Family_TNA', 'fund_no', 'Family_r12_2', 'Family_flow', 'Family_age'], 
			'Sentiment': ['sentiment', 'RecCFNAI', 'sentiment_lsq', 'sentiment_lad', 'CFNAI_orth', 'leading'], 
		}
		self._variable2category = {}
		for category in self._category:
			for var in self._category2variables[category]:
				self._variable2category[var] = category
		self._category2color = {
			'Fund mom': 'blue',
			'Fund char': 'plum',
			'Fund Family':'lime',
			'Sentiment':'darkgreen'
		}
		self._color2category = {value:key for key, value in self._category2color.items()}

	def getColorLabelMap(self):       
		return {var: self._category2color[self._variable2category[var]] for var in self._variable2category}

class DataInRamInputLayer:
	def __init__(self, 
				pathIndividualFeature, 
				idx_list,
				subset,                 
				pathMacroFeature=None,
				macroIdx=None, 
				meanMacroFeature=None, 
				stdMacroFeature=None, 
				normalizeMacroFeature=True):
		self.idx_list = idx_list 
		self.subset = subset        
		self._UNK = -99.99
		self._load_individual_feature(pathIndividualFeature)
		self._load_macro_feature(pathMacroFeature, macroIdx, meanMacroFeature, stdMacroFeature, normalizeMacroFeature)
		self._firm_char = FirmChar()

	def _create_var_idx_associations(self, varList):
		idx2var = {idx:var for idx, var in enumerate(varList)}
		var2idx = {var:idx for idx, var in enumerate(varList)}
		return idx2var, var2idx

	def _load_individual_feature(self, pathIndividualFeature):
		tmp = np.load(pathIndividualFeature)
		data = tmp['data']
		column_considered = [0]+[x+1 for x in self.subset]        
		data = data[:,:,column_considered]        
		data, list_considered = squeeze_data(data[self.idx_list])
		### Data Stored Here
		self._return = data[:,:,0]
		self._individualFeature = data[:,:,1:]     
       
		self._mask = (self._return != self._UNK)

		### Dictionary
		self._idx2date, self._date2idx = self._create_var_idx_associations(tmp['date'])
		self._idx2permno, self._permno2idx = self._create_var_idx_associations(tmp['wficn'][list_considered])
		self._idx2var, self._var2idx = self._create_var_idx_associations(tmp['variable'][self.subset])        
		self._dateCount, self._permnoCount, self._varCount = data.shape
		self._varCount -= 1

	def _load_macro_feature(self, pathMacroFeature, macroIdx=None, meanMacroFeature=None, stdMacroFeature=None, normalizeMacroFeature=True):
		self._macroFeature = np.empty(shape=[self._dateCount, 0])
		self._meanMacroFeature = None
		self._stdMacroFeature = None

	def getIndividualFeatureByIdx(self, idx):
		return self._idx2var[idx]

	def getFeatureByIdx(self, idx):
		if idx < self._varCount:
			return self.getIndividualFeatureByIdx(idx)
		else:
			return self.getMacroFeatureByIdx(idx - self._varCount)

	def getMacroFeatureMeanStd(self):
		return self._meanMacroFeature, self._stdMacroFeature

	def getIndividualFeatureColarLabelMap(self):
		return self._firm_char.getColorLabelMap(), self._firm_char._color2category

	def iterateOneEpoch(self, subEpoch=False):
		if subEpoch:
			for _ in range(subEpoch):
				yield self._macroFeature, self._individualFeature, self._return, self._mask
		else:
			yield self._macroFeature, self._individualFeature, self._return, self._mask