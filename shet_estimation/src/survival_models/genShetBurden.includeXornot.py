#
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
The command to run the script for generating individual shet burden is :

python  /path/to/source/survival_models/genShetBurden.includeXornot.py \
     [shet_cutoff] [AF_cutoff] [include_chrX] [input_file_list] \
     [output_file]                                                                                                                                                          
                                                                                                                                                                         
The input paramters are explained below.
shet_cutoff : the cutoff for selection coefficient. Ususally set shet_cutoff at 0.001.
AF_cutoff : the cutoff for allele frequency. Ususally set AF_cutoff at 0.0001.
include_chrX : indicating whether variants on chrX will be included in the shet burden calculation. The values can be 'Yes' or 'No'.
input_file_list : input file list for all the chrs
output_file : the output file for individual shet burden

'''



import numpy as np
import pandas as pd
import sys
import os
import glob


shet_cutoff=sys.argv[1]
AF_cutoff=sys.argv[2]
include_chrX=sys.argv[3]   ##can be 'Yes' or 'No'
input_file_list=sys.argv[4]
output_file=sys.argv[5]  

##read in missense/PTV variant files of UKBB samples for each chr
list_of_files = glob.glob(input_file_list)
if includeX=='No':
    list_of_files = [filename for filename in list_of_files if 'chrX' not in filename]


n_files = len(list_of_files)
list_df = []
for k in range(n_files):
    try:
        filename = list_of_files[k]
        print(filename)
        df1 = pd.read_csv(filename, header = 0, sep="\t")
        list_df.append(df1)
    except Exception as e:
        print(f"Error reading {list_of_files[k]}: {e}")
        continue

d2=pd.concat(list_df,ignore_index=True)
d2.columns=['chr','pos','ref','alt','sampleid','ENST','genename','selcoeff']
d2=d2.drop_duplicates()
d2['fitness']=1-d2['selcoeff']

m1=d2[ (d2.selcoeff > shet_cutoff)].groupby(['sampleid','ENST','genename']).agg({'pos':'count', 'fitness':'prod'}).reset_index()
m1.columns=['sampleid','ENST','genename','count_sel','fitness_sel']
m1['shet_cutoff']='s > 0.001'
m1['burden']=1-m1['fitness_sel']

m1.to_csv(output_file,index=False,sep="\t",compression='gzip')


