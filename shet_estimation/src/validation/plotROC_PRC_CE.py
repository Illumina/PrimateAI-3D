
# Copyright 2024 Illumina Inc, San Diego, CA                                                                                                                           
                                                                                                                                                                         
#                                                                                                                                                                        
# This program is licensed under the terms of the Polyform strict license                                                                                                
#                                                                                                                                                                        
# ***As far as the law allows, the software comes as is, without                                                                                                         
# any warranty or condition, and the licensor will not be liable                                                                                                         
# to you for any damages arising out of these terms or the use                                                                                                           
# or nature of the software, under any kind of legal claim.***                                                                                                           
#                                                                                                                                                                        
# You should have received a copy of the PolyForm Strict License 1.0.0                                                                                                   
# along with this program.  If not, see <https://polyformproject.org/licenses/strict/1.0.0>.                                                                             
#                                                                                                                                                                        
#                                                                                                                                                                        
#                                                                                                                                                                        

'''                                                                                                                                                                      
USAGE                                                                                                                                                                    
The command to run the script for plotting ROC and PRC curves for validation data set CE (cell essential) is :                                                                           
python  /path/to/source/validation/plotROC_PRC_CE.py \                                                                                                             
     [datafile]                                                                                                                                                          
                                                                                                                                                                         
datafile contains gene contraint scores merged from multiple methods, including \                                                                                        
'selbin10', 'selPTV', 'selMax' from our methods,                                                                                                                         
'pLI','oe_lof_upper' from 'gnomad.v2.1.1.lof_metrics.by_gene.txt',                                                                                                       
'shet_weghorn' from 'Supplementary_Table1.txt' in Weghorn et al. 2019,                                                                                                   
'GeneBayes' from 'GeneBayes_supptable2.tsv' in Zeng et al. Biorxiv,                                                                                                      
'shet_molly' from 'elife-83172-supp2-v2.txt' in Agarwal et al. 2023,                                                                                                     
'shet_regeneron' from 'SuppTable2.txt' in Sun et al. Biorxiv.                                                                                                            
'''


import numpy as np
import glob
import pandas as pd
import os
import sys
import scipy
import scipy.stats
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import glob
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42

datafile=sys.argv[1]

d1=pd.read_csv(datafile,header=0,sep='\t')

ce=pd.read_csv('CEgene.list.txt',header=None)
nce=pd.read_csv('nonCEgene.list.txt',header=None)
ce.columns=['genename']
nce.columns=['genename']
ce['label']=1
nce['label']=0
dce=pd.concat([ce,nce],ignore_index=True)

df1=dce.merge(d1,how='left',on='genename')
df1['LOEUF']=-df1['oe_lof_upper']


metrics = ['selbin10', 'selPTV', 'selMax', 'pLI','LOEUF', 'shet_weghorn', 'GeneBayes', 'shet_molly',
       'shet_regeneron']
labels=['s$_{het}$ [ missense top 10% ]', 's$_{het}$ [ LOF ]', 's$_{het}$ [ max ]','pLI',
        'LOEUF$^{*}$','s$_{het}$ (Weghorn 2019)', 's$_{het}$ (GeneBayes)', 's$_{het}$ (Agarwal 2023)',
      's$_{het}$ (Sun Biorxiv)']
our_metrics = ['selbin10', 'selPTV', 'selMax']
colors=['palevioletred','magenta','crimson','olive','orange','lightseagreen','green','dodgerblue','blue']



roc_auc_scores = {metric: {} for metric in metrics}
true_labels = df1['label']
for metric_index,metric in enumerate(metrics):
    print(metric)
    mask_no_nan = ~np.isnan(df1[metric])
    true_labels_no_nan = true_labels[mask_no_nan]
    metric_values_no_nan = df1[metric][mask_no_nan]
    precision, recall, _ = precision_recall_curve(true_labels_no_nan, metric_values_no_nan)
    pr_auc = auc(recall, precision)
    metric_name=labels[metric_index]
    print(pr_auc)
    roc_auc_scores[metric]['n_genes']=len(true_labels_no_nan)
    roc_auc_scores[metric]['auROC'] = roc_auc_score(true_labels_no_nan, metric_values_no_nan)
    roc_auc_scores[metric]['auPRC'] = pr_auc


roc_auc_df = pd.DataFrame.from_dict(roc_auc_scores,orient='index').reset_index()
roc_auc_df['label']=labels
roc_auc_df.to_csv(dir2+'auROC_PRC.csv',index=False)



###plot ROC Curves 
roc_auc_df=pd.read_csv(dir2+'auROC_PRC.csv',header=0)
roc_auc_df['color']=colors
roc_auc_df=roc_auc_df.sort_values(by='auROC',ascending=False)
roc_auc_df['mode']=np.where(roc_auc_df['index'].isin(our_metrics), 'Our methods','Other methods')

roc_auc_df1=roc_auc_df[roc_auc_df['mode']=='Other methods']
metrics1 = list(roc_auc_df1['index'])
labels1=list(roc_auc_df1['label'])
color1=list(roc_auc_df1['color'])

roc_auc_df2=roc_auc_df[roc_auc_df['mode']=='Our methods']
metrics2 = list(roc_auc_df2['index'])
labels2=list(roc_auc_df2['label'])
color2=list(roc_auc_df2['color'])

roc_auc_df1=roc_auc_df.sort_values(by='auROC',ascending=False)
metrics = list(roc_auc_df1['index'])
labels=list(roc_auc_df1['label'])

fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)
true_labels = df1['label']
list1=[]
for metric_index,metric in enumerate(metrics1):
    print(metric)
    mask_no_nan = ~np.isnan(df1[metric])
    true_labels_no_nan = true_labels[mask_no_nan]
    metric_values_no_nan = df1[metric][mask_no_nan]
    fpr, tpr, _ = roc_curve(true_labels_no_nan, metric_values_no_nan)
    roc_auc = auc(fpr, tpr)
    metric_name=labels1[metric_index]
    line1, =ax.plot(fpr, tpr, color=color1[metric_index], label=f'{metric_name}  (auROC = {roc_auc:.3f})')
    list1.append(line1)
    print(roc_auc)

list2=[]
for metric_index,metric in enumerate(metrics2):
    print(metric)
    mask_no_nan = ~np.isnan(df1[metric])
    true_labels_no_nan = true_labels[mask_no_nan]
    metric_values_no_nan = df1[metric][mask_no_nan]
    fpr, tpr, _ = roc_curve(true_labels_no_nan, metric_values_no_nan)
    roc_auc = auc(fpr, tpr)
    metric_name=labels2[metric_index]
    line2, =ax.plot(fpr, tpr, color=color2[metric_index], label=f'{metric_name}  (auROC = {roc_auc:.3f})')
    list2.append( line2 )


x = np.linspace(0, 1, 100)
y = x 
ax.plot(x, y, color='gray', linestyle='--')
l1=ax.legend(handles=list1, loc=(0.44,0.03), labelcolor=color1, title='Constraint metrics (other):',alignment='left')
ax.add_artist(l1)
l2=ax.legend(handles=list2, loc=(0.44,0.3), labelcolor=color2,title='Constraint metrics (this work):',alignment='left')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.set_title(f"Classifying genes essential for cell survival $in$ $vitro$")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
fig.savefig('Figure_ROCurve.pdf',format='pdf')


###plot Precision-Recall Curves
roc_auc_df=pd.read_csv(dir2+'auROC_PRC.csv',header=0)
roc_auc_df['color']=colors
roc_auc_df=roc_auc_df.sort_values(by='auPRC',ascending=False)
roc_auc_df['mode']=np.where(roc_auc_df['index'].isin(our_metrics), 'Our methods','Other methods')

roc_auc_df1=roc_auc_df[roc_auc_df['mode']=='Other methods']
metrics1 = list(roc_auc_df1['index'])
labels1=list(roc_auc_df1['label'])
color1=list(roc_auc_df1['color'])

roc_auc_df2=roc_auc_df[roc_auc_df['mode']=='Our methods']
metrics2 = list(roc_auc_df2['index'])
labels2=list(roc_auc_df2['label'])
color2=list(roc_auc_df2['color'])


fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)
true_labels = df1['label']
list1=[]
for metric_index,metric in enumerate(metrics1):
    print(metric)
    mask_no_nan = ~np.isnan(df1[metric])
    true_labels_no_nan = true_labels[mask_no_nan]
    metric_values_no_nan = df1[metric][mask_no_nan]
    precision, recall, _ = precision_recall_curve(true_labels_no_nan, metric_values_no_nan)
    pr_auc = auc(recall, precision)
    metric_name=labels1[metric_index]
    line1, =ax.plot(recall, precision, color=color1[metric_index], label=f'{metric_name}  (auPRC = {pr_auc:.3f})')
    list1.append(line1)
    print(pr_auc)

list2=[]
for metric_index,metric in enumerate(metrics2):
    print(metric)
    mask_no_nan = ~np.isnan(df1[metric])
    true_labels_no_nan = true_labels[mask_no_nan]
    metric_values_no_nan = df1[metric][mask_no_nan]
    precision, recall, _ = precision_recall_curve(true_labels_no_nan, metric_values_no_nan)
    pr_auc = auc(recall, precision)
    metric_name=labels2[metric_index]
    line2, =ax.plot(recall, precision, color=color2[metric_index], label=f'{metric_name}  (auPRC = {pr_auc:.3f})')
    list2.append(line2)

l1=ax.legend(handles=list1, loc=(0.4,0.5), labelcolor=color1, title='Constraint metrics (other):',alignment='left')
ax.add_artist(l1)
l2=ax.legend(handles=list2, loc=(0.4,0.8), labelcolor=color2,title='Constraint metrics (this work):',alignment='left')

plt.xlabel('Recall')
plt.ylabel('Precision')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
fig.savefig('Figure_PRcurve.pdf',format='pdf')
