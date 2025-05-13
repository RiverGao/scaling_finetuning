import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV
import warnings
import time
pd.options.mode.chained_assignment = None
from joblib import Parallel, delayed

DIR = '/scratch/ResearchGroups/lt_jixingli/readbrain/'
os.chdir(DIR)

group = sys.argv[1]
name = sys.argv[2]
size = sys.argv[3]
layer = sys.argv[4]

model_size_heads = {'gpt2_base':12,'gpt2_medium':16,'gpt2_large':20,'gpt2_xlarge':25,
'llama_7B':32,'llama_13B':40,'llama_30B':52,'llama_65B':64,
'alpaca_7B':32,'alpaca_13B':40,'vicuna_7B':32,'vicuna_13B':40,
'mistral_7B':32,'gemma_7B':16}
n_head = model_size_heads['%s_%s' %(name,size)]

arti_num,snt_max,word_max,n_subj = 5,31,16,51
split_idx = int(148*0.9)
words = pd.read_csv('Analysis/words.csv')
tril_i,tril_j = np.tril_indices(word_max,k=-1)
words_list = pickle.load(open('Analysis/words_list.p','rb'))
snts_list = [31,31,28,28,30]
cums = np.cumsum(snts_list)

def load_subject_events(group,subj,arti_num):
	events_all = []
	for article in range(1,arti_num+1):
		for run in range(1,6):
			fname = 'Data/%s/%s/func/%s_task-read_run-%s_events.tsv' %(group,subj,subj,run)
			if not os.path.exists(fname):
				continue
			df = pd.read_csv(fname,delimiter='\t').dropna().reset_index(drop=True)
			if int(df.SentenceID[0].split('.')[1])==article:
				df['article'] = article
				events_all.append(df)
				break
	return pd.concat(events_all, ignore_index=True)

def process_subject(subj_id,X_train,X_test,rm_id):
	subj = 'sub-0%s' %subj_id if subj_id<10 else 'sub-%s' %subj_id
	events = load_subject_events(group,subj,arti_num)
	y_num_train, y_num_test,y_dur_train,y_dur_test = [],[],[],[]
	
	for article in range(1,arti_num+1):
		n_snts = int(words[words.SentenceID.str.match(f't.0{article}')].SentenceID.iloc[-1].split('.')[-1])
		for i in range(1,n_snts+1):
			snt_id = f't.0{article}.0{i}' if i<10 else f't.0{article}.{i}'
			event = events[events.SentenceID.str.match(snt_id)].copy()
			sid = i-1 if article == 1 else cums[article-2]+i-1
			sent_len = words_list[sid]
			sac_num = np.zeros((sent_len,sent_len))
			sac_dur = np.zeros((sent_len,sent_len))
			tril_idx = np.tril_indices(sent_len,k=-1)
			if len(event) >= 2:
				event['duplicate'] = event.CURRENT_FIX_INTEREST_AREA_ID.eq(event.CURRENT_FIX_INTEREST_AREA_ID.shift())
				dup_idx = event[event['duplicate']].index
				for ind in dup_idx:
					if ind > 0:
						event.at[ind-1,'duration'] += event.at[ind,'duration']
					event = event.drop(dup_idx).reset_index(drop=True)
					for ind in range(len(event)-1):
						row = int(event.iloc[ind].CURRENT_FIX_INTEREST_AREA_ID)-1
						col = int(event.iloc[ind+1].CURRENT_FIX_INTEREST_AREA_ID)-1
						if 0 <= row < sent_len and 0 <= col < sent_len:
							sac_num[row,col]+=1
							sac_dur[row,col]+=event.iloc[ind].duration + event.iloc[ind+1].duration
			if sid<split_idx:
				y_num_train.extend(sac_num[tril_idx])
				y_dur_train.extend(sac_dur[tril_idx])
			else:
				y_num_test.extend(sac_num[tril_idx])
				y_dur_test.extend(sac_dur[tril_idx])
 
	y_num_full = np.array(y_num_train+y_num_test)
	y_num_full = np.nan_to_num(zscore(y_num_full,nan_policy='omit'))
	y_num_train = y_num_full[:len(y_num_train)]
	y_num_test  = y_num_full[len(y_num_train):]

	y_dur_full = np.array(y_dur_train+y_dur_test)
	y_dur_full = np.nan_to_num(zscore(y_dur_full,nan_policy='omit'))
	y_dur_train = y_dur_full[:len(y_dur_train)]
	y_dur_test  = y_dur_full[len(y_dur_train):]

 	model_num = RidgeCV(alphas=np.logspace(1,3,20)).fit(X_train.T, y_num_train)
 	pred_num = model_num.predict(X_test.T)
 	corr_num, _ = pearsonr(pred_num,y_num_test)
 	model_dur = RidgeCV(alphas=np.logspace(1,3,20)).fit(X_train.T, y_dur_train)
 	pred_dur = model_dur.predict(X_test.T)
 	corr_dur, _ = pearsonr(pred_dur,y_dur_test)
 	return corr_num, corr_dur

layer_attn = np.load('Analysis/attns/%s/%s/p1/rb_p1_layer%d.npy' %(name,size,layer))
attn = layer_attn.swapaxes(2,0)
X = np.array([np.concatenate(i,axis=0) for i in attn])
X = np.delete(X,rm_id,axis=1)

start = time.time()
X_train,X_test = [],[]
for i in range(X.shape[1]):
	tril_idx = np.tril_indices(words_list[i],k=-1)
	if i<split_idx:
		X_snt_train = X[:,i,:words_list[i],:words_list[i]]
		X_train.append(X_snt_train[:,tril_idx[0],tril_idx[1]])
	else:
		X_snt_test = X[:,i,:words_list[i],:words_list[i]]
		X_test.append(X_snt_test[:,tril_idx[0],tril_idx[1]])
		
X_train = np.concatenate(X_train,axis=1)
X_test = np.concatenate(X_test,axis=1)
results = Parallel(n_jobs=8)(delayed(process_subject)(sid,X_train,X_test,rm_id) for sid in range(1,n_subj+1))
layer_num = [r[0] for r in results if r is not None]
layer_dur = [r[1] for r in results if r is not None]
end = time.time()
print(f'Total time: {end-start:.2f} seconds')

layer_num = np.array(layer_num)
layer_dur = np.array(layer_dur)
np.save(f'Results/ridge/sac_cv/%s%s_layer%s_num.npy' %(name,size,layer), layer_num)
np.save(f'Results/ridge/sac_cv/%s%s_layer%s_dur.npy' %(name,size,layer), layer_dur)

