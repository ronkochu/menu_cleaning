
import numpy as np
import _pickle as pickle
import gym, os, datetime, time, copy, re, sys, csv
from sklearn.gaussian_process import GaussianProcess
from pprint import pprint
from datetime import datetime
import tensorflow as tf
import pandas as pd
from random import shuffle
import itertools
import gensim, json
from multiprocessing import Pool
from menu2vec import m2v


#slightly difference from v1, unique labels is no longer in params, need a seperate set of unique labels for each sub-model
#pool not necessary actually, pool was used in v1 to speed up order check, 

def pool_output_predictions(stim_block, pred_block, t_pos, params, model_name, b_idx):
	out_arr = [];
	top_n = 1;
	rev = 'rev' in model_name;
	m2v.ts_print('starting pool block - %s' % b_idx)
	for idx, stim in enumerate(stim_block):
		#out_row = [stim,labels1[idx],labels2[idx],labels3[idx],labels4[idx]];
		out_row = [stim];
		pred_row = [];
		conf_row = [];
		for idx2, pred in enumerate(pred_block):
			pred_pos = pred[idx].argsort()[::-1][:top_n];
			confidence = pred[idx,pred_pos].tolist()[0];
			pred_label =  np.array(params['unique_labels'][idx2])[pred_pos].tolist()[0];
			pred_row += [pred_label];
			conf_row += [confidence];
		if rev:
			pred_row = pred_row[::-1];
			conf_row = conf_row[::-1];
				
		out_row += pred_row;
		out_row += conf_row;
		out_row += [m2v.check_label_chain(t_pos, pred_row)];
		out_arr.append(out_row);
	
	m2v.ts_print('pool block done - %s' % b_idx)
	return out_arr;

def extract_pred_block(all_preds, start, stop):
	out_arr = [];
	for preds in all_preds:
		out_arr.append(preds[start:stop])
	return out_arr;

#sloppy AF
def kw_lookup_check(all_stim, pred_cols, label_idx, prev_tier_label, kw_lookup):
	#print('\n\n looking up %s \n\n' % prev_tier_label);
	lookup_subset = kw_lookup[kw_lookup['prev_tier_label'] == prev_tier_label];
	in_stim = all_stim[label_idx];
	for s_idx, stim in enumerate(in_stim):
		stim_arr = m2v.clean_split_doc(stim, pp_func = 'bad_chars_no_digits')
		assigned = False;
		for lr_idx, lookup_row in lookup_subset.iterrows():
			if assigned: break;
			exclusion_check = False;
			
			#check exclusions first
			ex_arrs = lookup_row['exclusions'];
			if ex_arrs == ex_arrs: #exclude nans
				for ex_arr in ex_arrs:
					if set(ex_arr) & set(stim_arr) == set(ex_arr):
						exclusion_check = True;
				if exclusion_check: continue;
			
			#now check keywords
			kw_arrs = lookup_row['key_words'];
			for kw_arr in kw_arrs:
				if set(kw_arr) & set(stim_arr) == set(kw_arr):
					new_assignment = lookup_row['assignment']
					#
					ass_val = 69;
					old_assignment = pred_cols[label_idx[s_idx], 0];
					if new_assignment != old_assignment:
						ass_val = 6969
						#print('\n', stim, old_assignment, new_assignment, '\n')
					#
					pred_cols[label_idx[s_idx], 0] = new_assignment;
					pred_cols[label_idx[s_idx], 1] = ass_val;
					#print('\n\n %s assigned to %s from %s \n\n' % (stim, new_assignment, prev_tier_label))
					assigned = True;
					break 
	return pred_cols
	
	
def extract_full_tier(out_df, params, tier, fdir, model_names):
	
	kw_fname = '/home/ron/code/mcln/mcln_m2v_category_keywords.json'
	kw_lookup = json.load(open(kw_fname));
	kw_lookup = pd.DataFrame(kw_lookup);
	
	#pred_cols = np.zeros((out_df.index.shape[0],3)).astype(str);
	#pred_cols[:,:] = '';
	pred_cols = np.empty((out_df.index.shape[0],3), object)
	all_stim = out_df['menu_item'].values;
	for label in out_df['prediction_%s' % (tier-1)].unique():
		label_idx = np.where((out_df['prediction_%s' % (tier-1)] == label).values)[0];
		prev_tier_labels = np.unique(out_df['prev_tier_label'].values[label_idx]);
		
		for pvl in prev_tier_labels:
			
			label_idx = np.where(((out_df['prediction_%s' % (tier-1)] == label) & (out_df['prev_tier_label'] == pvl)).values)[0]
			prev_tier_label = '%s%s' % (pvl, label);
			
			#tf_input = params['tf_input'];
			
			#tf_input = get_tf_input(params['tf_inputs'], prev_tier_label);
			#if len(tf_input) > 0:
			tf_input, preds = bag_model_predictions(all_stim[label_idx], fdir, model_names, prev_tier_label);
			if len(preds) > 0:
				max_idx = preds.argmax(axis = 1);
				max_vals = np.amax(preds, axis = 1);
				
				pred_cols[label_idx,0] = tf_input['categories'][max_idx];
				pred_cols[label_idx,1] = max_vals;
				pred_cols[label_idx,2] = '%s|'  % prev_tier_label;
				#print(prev_tier_label)
				
				if params['do_kw_lookup']:
					#do kw_lookup here for now, maybe find better place for this
					if prev_tier_label in kw_lookup['prev_tier_label'].values:
						pred_cols = kw_lookup_check(all_stim, pred_cols, label_idx, prev_tier_label, kw_lookup) 
		
	out_df['prediction_%s' % tier] = pred_cols[:,0];
	out_df['confidence_%s' % tier] = pred_cols[:,1];
	out_df['prev_tier_label'] = pred_cols[:,2];
	
	return out_df

#bagging across models, takes average probability over predicted classes
def bag_model_predictions(all_stim, fdir, model_names, prev_tier_label):
	model_preds = [];
	
	for model_name in model_names:
		savename = '%s%s' % (fdir,model_name);
		#params = pickle.load(open('%s_tier_1_params.p'%savename, 'rb'));
	
		prev_tier_label_fname = '-'.join('_'.join(prev_tier_label.split('|')).split('/'));
		in_model = '%s_%s' % (model_name, prev_tier_label_fname);
		savename = '%s%s' % (fdir,in_model);
		try:
			params =  pickle.load(open('%s_params.p'%savename, 'rb'));
		except:
			return None, []
			
		tf_input = params['tf_input'];
		tf_input['input_size'] = tf_input['input_shape'];
		#tf_input = get_tf_input(params['tf_inputs'], prev_tier_label);
		
		preds = extract_tier_subset(all_stim, params, tf_input, savename);
		model_preds.append(preds);
	
	preds = np.mean(model_preds, axis = 0);
	return tf_input, preds

#extract from models here
def extract_tier_subset(all_stim, params, tf_input, savename):
	tf.reset_default_graph()
	
	keep_prob = tf.placeholder(tf.float32);
	
	###SOOOOOOOOOOO HACKY
	if tf_input['input_shape'][-1] == 1:
		tf_input['input_shape'] = tf_input['input_shape'][:-1]
	input_x = tf.placeholder(tf.float32, [None] + tf_input['input_shape']);
	
	nn_layers, nn_biases, nn_weights, nn_out_vecs = m2v.deepnn(input_x, tf_input, keep_prob);
	y_conv  = nn_layers[-1];
	prediction = y_conv*1;
	all_predictions = [
		tf.nn.softmax(y_conv),
	];
	
	config = tf.ConfigProto(
		inter_op_parallelism_threads=6,
		intra_op_parallelism_threads=6,
	);
	
	sess = tf.Session(config=config);
	saver = tf.train.Saver();
	saver.restore(sess, savename);
		
	tf_vars = {
		'input_x': input_x,
		'keep_prob': keep_prob,
		'prediction': all_predictions
	};
	
	params['tf_input']['dic'] = np.array(params['tf_input']['dic'])
	if 'token' in savename:
		all_preds = m2v.get_tf_prediction_pvals(all_stim.tolist(), sess, params['tf_input'], tf_vars, 'tokens');
	else:
		all_preds = m2v.get_tf_prediction_pvals(all_stim.tolist(), sess, params['tf_input'], tf_vars, 'chars');
	
	return all_preds[0];

	
	
def get_tf_input(tf_inputs, prev_tier_label):
	for tf_input in tf_inputs:
		if tf_input['prev_tier_label'] == prev_tier_label:
			return tf_input
	
	#print('%s not found!' % prev_tier_label)
	return [];
	
#same as extract block above, exept column names are changed because of course they are... 
def extraction_block(df, fdir, model_names, params):
	
	kw_fname = '/home/ron/code/mcln/mcln_m2v_category_keywords.json'
	kw_lookup = json.load(open(kw_fname));
	kw_lookup = pd.DataFrame(kw_lookup);
	
	all_stim = df['menu_item'].str.lower().values.astype(str);
	
	prev_tier_label = 'tier_1';
	tf_input, preds = bag_model_predictions(all_stim, fdir, model_names, prev_tier_label);
	
	max_idx = preds.argmax(axis = 1);
	max_vals = np.amax(preds, axis = 1)
	
	out_df = pd.DataFrame(all_stim, columns = ['menu_item'])
	out_df['prediction_1'] = tf_input['categories'][max_idx];
	out_df['confidence_1'] = max_vals;
	out_df['prev_tier_label'] = '';
	
	#do kw lookup here, scrappy implementation for now
	pred_cols = out_df[['prediction_1', 'confidence_1', 'prev_tier_label']].values;
	#print(pred_cols)
	prev_tier_label = '';
	label_idx = np.where(np.ones(pred_cols.shape[0], dtype = bool))[0];
	if params['do_kw_lookup']:
		pred_cols = kw_lookup_check(all_stim, pred_cols, label_idx, prev_tier_label, kw_lookup) 
	
	
	out_df['prediction_1'] = pred_cols[:,0];
	out_df['confidence_1'] = pred_cols[:,1];
	out_df['prev_tier_label'] = pred_cols[:,2];
	
	for tier in range(2,5):
		out_df = extract_full_tier(out_df, params, tier, fdir, model_names)
	
	df['prediction_1'] = out_df['prediction_1'].values
	df['prediction_2'] = out_df['prediction_2'].values
	df['prediction_3'] = out_df['prediction_3'].values
	df['prediction_4'] = out_df['prediction_4'].values
	
	df['confidence_1'] = out_df['confidence_1'].values
	df['confidence_2'] = out_df['confidence_2'].values
	df['confidence_3'] = out_df['confidence_3'].values
	df['confidence_4'] = out_df['confidence_4'].values
	
	return df
	
def generate_relabel_df(df):
	out_idx =  np.ones(df.index.shape[0], dtype = bool);
	
	for tier in range(3,5):
		out_idx &= df['confidence_%s' % tier].astype(float) < 69
	
	out_idx &= df['bad_items'] == 1
	out_df = df[out_idx];
	
	out_df['new_prediction_1'] = ''
	out_df['new_prediction_2'] = ''
	out_df['new_prediction_3'] = ''
	out_df['new_prediction_4'] = ''
	
	out_df = out_df[[
		'new_prediction_1',
		'new_prediction_2',	
		'new_prediction_3',	
		'new_prediction_4',	
		'menu_item',
		'prediction_1',
		'prediction_2',
		'prediction_3',
		'prediction_4',
		'confidence_1',
		'confidence_2',	
		'confidence_3',
		'confidence_4',
	]];
	
	return out_df

def prediction_merge(df):
	idx = df['CLEANSED_TIER_1'].isnull();
	
	for tier in range(1,5):
		df.loc[idx,'CLEANSED_TIER_%s' % tier] = df.loc[idx,'prediction_%s' % tier];
	
	return df;


in_path = sys.argv[1]
in_col = sys.argv[2]
if __name__ == "__main__":
	#########load parameters and set file names
	fdir = '/DataDrive/ron/menu_cleaning/';
	model_fdir = '/DataDrive/ron/menu_cleaning/models/';
	
	model_names = [
		'menu2vec_token_subset_20180912',
		'menu2vec_char_subset_20180912',
	];
	
	
	savename = '%s%s' % (model_fdir,model_names[0]);
	params = pickle.load(open('%s_tier_1_params.p'%savename, 'rb'));
	
	params['do_kw_lookup'] = False;
	
	t_pos = pd.read_csv('%slabel_tier_start_stop.csv' % fdir);
	
	out_name = '%s_labelled.csv' % in_path.split('.csv')[0]
	##########
	
	df = pd.read_csv(in_path, encoding = 'latin-1')
	in_df = df.drop_duplicates(subset = [in_col]);
	in_df['menu_item'] = in_df[in_col];
	
	in_df = df.drop_duplicates(subset = [in_col]);
	in_df['menu_item'] = in_df[in_col];
	in_df['menu_item'] = in_df['menu_item'].str.lower().values.astype(str);
	in_df['menu_item'] = in_df['menu_item'].apply(lambda x: m2v.drop_prefixes(x));
	in_df['menu_item'] = in_df['menu_item'].apply(lambda x: m2v.string_preproc(x, params['tf_input']['pp_func']));
	in_df = in_df[['menu_item', in_col]];

	block_size = 10000;
	nrows = in_df.index.shape[0];
	out_df = pd.DataFrame([]);
	
	out_fdir = '/DataDrive/ron/menu_cleaning/';	
	for bi in range(0, nrows, block_size):
		m2v.ts_print('starting extraction block %s/%s' % (bi,in_df.index.shape[0]))
		block_df = in_df.iloc[bi:bi+block_size,]
		block_df = extraction_block(block_df, model_fdir, model_names, params)
		out_df = pd.concat((out_df,block_df))
		
	out_df = out_df.drop(columns = ['menu_item'])
	out_df = pd.merge(df, out_df, how = 'left')
	out_df.to_csv(out_name, index = False, encoding='utf-8')	
