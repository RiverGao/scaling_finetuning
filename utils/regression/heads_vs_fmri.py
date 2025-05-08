import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV
import warnings
import time
from joblib import Parallel, delayed
pd.options.mode.chained_assignment = None
from scipy.stats import pearsonr, ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)

DIR = '/scratch/ResearchGroups/lt_jixingli/readbrain/'
os.chdir(DIR)

subj_id = int(sys.argv[1])
group = sys.argv[2]
name = sys.argv[3]
size = sys.argv[4]
hem = sys.argv[5]
layer = sys.argv[6]

subj = 'sub-0%d' % subj_id if subj_id<10 else 'sub-%d' % subj_id
model_size_heads = {'gpt2_base':12,'gpt2_medium':16,'gpt2_large':20,'gpt2_xlarge':25,
'llama_7B':32,'llama_13B':40,'llama_30B':52,'llama_65B':64,
'alpaca_7B':32,'alpaca_13B':40,'vicuna_7B':32,'vicuna_7B':40,
'mistral_7B':32,'gemma_7B':16}
n_head = model_size_heads['%s_%s' %(name,size)]

rm_id = [142,143,147,148,152,153,154]
LMask_id = np.load('Analysis/%sMask_fs5_id.npy' %hem)
words = pd.read_csv('Analysis/words.csv')
words_list = pickle.load(open('Analysis/words_list.p', 'rb'))
snts_list = [31,31,28,28,30]
cums = np.cumsum(snts_list)

layer_attn = np.load('Analysis/attns/%s/%s/p1/rb_p1_layer%d.npy' %(name,size,layer))
attn = layer_attn.swapaxes(2,0)
X = np.array([np.concatenate(i,axis=0) for i in attn])
X = np.delete(X,rm_id,axis=1)

X = np.nan_to_num(zscore(X.flatten(),nan_policy='omit')).reshape(X.shape)
X_train,X_test = [],[]
for i in range(X.shape[1]):
	tril_idx = np.tril_indices(words_list[i],k=-1)
	if i <133:
	 X_snt_train = X[:,i,:words_list[i],:words_list[i]]
	 X_train.append(X_snt_train[:,tril_idx[0],tril_idx[1]])
	else:
	 X_snt_test = X[:,i,:words_list[i],:words_list[i]]
	 X_test.append(X_snt_test[:,tril_idx[0],tril_idx[1]])
X_train = np.concatenate(X_train,axis=1)
X_test = np.concatenate(X_test,axis=1)

surf=[]
events_lst=[]
for article in range(1,6):
 run = 1
 events = pd.read_csv('Data/%s/%s/func/%s_task-read_run-%d_events.tsv' %(group,subj,subj,run),delimiter='\t')
 events = events.dropna().reset_index(drop=True)
 run_article = int(events.SentenceID[0].split('.')[1])
 while run_article != article and run < 5:
  run += 1
  events = pd.read_csv('Data/%s/%s/func/%s_task-read_run-%d_events.tsv' %(group,subj,subj,run),delimiter='\t')
  events = events.dropna().reset_index(drop=True)
  run_article = int(events.SentenceID[0].split('.')[1])
 events['article'] = [article]*len(events)
 events_lst.append(events)
 fmri_gii = nib.load('Data/fsaverage5/%s_task-read_run-%d_hemi-%s_space-fsaverage5_bold.func.gii' %(subj,run,hem))
 fmri_data = np.column_stack([arr.data for arr in fmri_gii.darrays])
 fmri_data = np.nan_to_num(zscore(fmri_data,axis=0,nan_policy='omit'))
 surf.append(fmri_data)
events = pd.concat(events_lst).reset_index(drop=True)
corr = np.zeros(len(LMask_id))

def process_vertex(v,X_train,X_test,events,surf,words):
 y_train, y_test = [],[]
 for article in range(1,6):
  n_snts = int(words[words.SentenceID.str.match('t.0%d' %article)].SentenceID.iloc[-1].split('.')[-1])
  for i in range(1,n_snts+1):
   snt_id = 't.0%d.0%d' % (article,i) if i < 10 else 't.0%d.%d' %(article,i)
   event = events[events.SentenceID.str.match(snt_id)]
   sid = i-1 if article==1 else cums[article-2]+i-1   
   fmri_snt = np.zeros((words_list[sid],words_list[sid]))
   tril_idx = np.tril_indices(words_list[sid],k=-1)
   event = event.reset_index(drop=True) 
   for ind, e in event.iterrows():
    if ind < len(event)-1:
     row = int(e.CURRENT_FIX_INTEREST_AREA_ID)-1
     col = int(event.iloc[ind+1].CURRENT_FIX_INTEREST_AREA_ID)-1
     scan = int(np.ceil((event.iloc[ind+1].onset/1000+5)/0.4))
     while scan >= surf[article-1].shape[1]:
      scan -= 1
     fmri_snt[row,col] = surf[article-1][v,scan]
   if sid <133:
	   y_train.extend(fmri_snt[tril_idx[0],tril_idx[1]])
   else:
	   y_test.extend(fmri_snt[tril_idx[0],tril_idx[1]])
 y_train = np.array(y_train)
 y_test = np.array(y_test)
 model_train = RidgeCV(alphas=[0.1,1.0,10.0]).fit(X_train.T,y_train)
 y_predict = X_test.T @ model_train.coef_
 corr[v],_ = pearsonr(y_predict,y_test)
 corr[v] = np.nan_to_num(corr[v])
 return corr[v]

start = time.time()
results = Parallel(n_jobs=-1)(
    delayed(process_vertex)(v,X_train,X_test,events,surf,words) for v in range(len(LMask_id))
)
corr = np.array(results)
end = time.time()
elapsed = end - start
print(f'Time taken: {elapsed:.6f} seconds')
np.save('Results/ridge/cv/%s/%s/%s_subj%d_corr%s.npy' %(name,size,group,subj_id,hem),corr)
