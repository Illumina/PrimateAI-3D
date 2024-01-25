# Copyright 2024 Illumina Inc, San Diego, CA                                                                                                                                  
#                                                                                                                                                                             
#    This program is free software: you can redistribute it and/or modify                                                                                                     
#    it under the terms of the GNU General Public License as published by                                                                                                     
#    the Free Software Foundation, either version 3 of the License, or                                                                                                        
#    (at your option) any later version.                                                                                                                                      
#                                                                                                                                                                             
#    This program is distributed in the hope that it will be useful,                                                                                                          
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                                                                                                           
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                                                                                            
#    GNU General Public License for more details.                                                                                                                             
#                                                                                                                                                                             
#    You should have received a copy of the GNU General Public License                                                                                                        
#    along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.txt                                                                                           


'''
The forward time population simulation incorporates the effects of selection pressure, using the optimal paramaters generated from simulation under neutral evolution.                                                                
The goal is to simulation allele frequency spectrum for rare variants under different mutation rates.

Consider 192 mutation rates for the 192 tri-nucleotide contexts. For those CpGTi contexts, two typese of mutation rates are included, the one with high methylation rates and the other with low methylation rates.

36 selection coefficients are used in the simulation, including (0.0001, 0.0002, …,0.0009), (0.001,0.002,…,0.009), (0.01,0.02,…,0.09), (0.1,0.2,…,0.9).

Assume the census population sizes listed below, 30 years per generation, de novo mutation rates,                                                             
 and the ratio between effective population size and the census population size.                                                                      
You may use the effective population for your simulation, not the census population size.                                                                     
r=Nc/Ne, the ratio between effective population size and the census population size.                                                                          
nu represents growth rate.                                                                                                                                    
                                                                                                                                                              
Simulate population expansion history as follows:                                                                                                             
??000 BC: 10,000 initial effective population size, ??000 census population size, Nc0=r*Ne0. Ne0=10,000                                                       
??000 BC: ?? million census population size, Nc1=r*Ne1, Ne1=2Ne0, nu1=Ne1/Ne0 = Nefactor                                                                      
1400 AD:  360 million census population size, Nc2=360m, Ne2=Nc2/r,nu2=Ne2/Ne1                                                                                 
1700 AD:  620 million census population size, Nc3=620m, Ne3=Nc3/r,nu3=62/36                                                                                   
2000 AD:  6.2 billion census population size, Nc4=6.2b, Ne4=Nc4/r,nu4=10                                                                                      
Now plug in the optimal paramaters generated from simulation under neutral evolution.


USAGE                                                                                                                                                         
The command to run the script for forward time population simulation under selection is :                                                                                                                                                              
python  /path/to/source/simulation/simulation_SFS_with_selection.py \
     [selection] [snptype] [mutrate] [nreps] [samplesize] [outdir]

The input paramters are explained below.
selection: selection coefficients used in the simulation
snptype: one of the 192 tri-nucleotide contexts
mutrate: mutation rates for one of the 192 tri-nucleotide contexts
nreps: Number of output samples sampled from the simulated final population at present day. Note these samples have a sample size resembling gnomAD exome sample size, 123K. 
samplesize: Number of individuals in each sample sampled from the simulated final population at present day, e.g., 500K corresponds to samplesize=1000000 
outdir: output directory  

'''

import numpy as np
import sys
import os
from collections import Counter



selection=float(sys.argv[1])  ##selection coefficients used in the simulation
snptype=sys.argv[2]      ###one of the 192 tri-nucleotide contexts
mutrate=float(sys.argv[3])  ##mutation rates for one of the 192 tri-nucleotide contexts
nreps=int(sys.argv[4])   ##Number of samples sampled from the simulated final population at present day, e.g., nreps=1000
samplesize=int(sys.argv[5])  ##Number of individuals in each sample sampled from the simulated final population at present day, e.g., 500K corresponds to samplesize=1000000
outdir=sys.argv[6]  ##output directory
if not os.path.isdir(outdir):
    os.mkdir(outdir)

###Simulate 100,000 independent loci 
seqlen=100000

popratio=30 ## r=Nc/Ne, its optimal value is 30
Nefactor=2.0  ##Nefactor = Ne1/Ne0, its optimal value is 2.0
N_ref =10000*2  ###inital effective popultion size
T1 = 3500   ##Number of generations during the burn-in period, its optimal value is 3500
nu1 = Nefactor ** (1. / T1)   
s1=popratio*1.0   
T2= 530    ##Number of generations during the first expension period after burn-in, its optimal value is 530 
nu2 = (36000./Nefactor/s1) ** (1./T2)
T3= 10 # int((1700-1400)/gen)
nu3 = (62./36.) ** (1./T3)
T4= 10  #int((2000-1700)/gen)
nu4= 10. **(1./T4)
T5 = T1+T2+T3+T4

print( str(T1)+" "+str(T2)+" "+str(T3)+" "+str(T4) )
print( str(nu1)+" "+str(nu2)+" "+str(nu3)+" "+str(nu4) )


if __name__ == '__main__':
    #initial
    ndenovo = int(N_ref*seqlen*mutrate)
    pos0=np.random.randint((N_ref*seqlen), size=ndenovo)  ##(N_ref, seqlen), p=[1-mutrate, mutrate])
    #prob0 = np.random.random(size = ndenovo)
    pos_0=pos0[ np.random.random(size = ndenovo) > selection ]
    snphash_prev = {key0: value0 for key0, value0 in zip(*np.divmod(pos_0, seqlen))}
    #snphash_prev={}
    #for i in range(len(pos_0)):
    #    key0,value0=divmod(pos_0[i],seqlen)
    popsize_prev=N_ref
    generations = {value0: 0 for value0 in snphash_prev.values()}
    
    for i in range(T5):
        if i <T1:
            nu=nu1
        elif i>=T1 and i< (T2+T1):
            nu=nu2
        elif i>=(T2+T1) and i< (T3+T2+T1):
            nu=nu3
        elif i>=(T3+T2+T1):
            nu=nu4
        popsize_curr = int(popsize_prev*nu)
        ###sampling current genome
        snphash_curr={}
        allsnps=set()
        idx = np.random.randint(popsize_prev, size=popsize_curr)
        for j in range(len(idx)):
            if idx[j] in snphash_prev:
                snppos = list(set(str(snphash_prev[ idx[j] ]).split(',')))
                probs=np.random.random(size = len(snppos))
                ###each individual has multiple variants. For each variant, sample selection prob to determine keeping the variant or not
                snppos1 = [snppos[i] for i in range(len(snppos)) if probs[i] > selection]
                if len(snppos1) > 0:
                    allsnps.update(snppos1)
                    snphash_curr[j] = ','.join(snppos1)
        generations = {snp: generation for snp, generation in generations.items() if str(snp) in allsnps}
        ###generate de novo mutations
        ndenovo = int(popsize_curr*seqlen*mutrate)
        #print str(i)+" "+str(popsize_prev) + " "+str(popsize_curr)+" "+str(ndenovo)
        pos1=np.random.randint((popsize_curr*seqlen), size=ndenovo)
        #prob1 = np.random.random(size = ndenovo)  ##record whether each denovo mutation is selected against or not
        pos_1=pos1[ np.random.random(size = ndenovo) > selection ]
        dnms = set()  ##recorde all the de novo mutations
        for k in range(len(pos_1)):
                key1,value1=divmod(pos_1[k],seqlen)
                dnms.update([value1])
                if key1 in snphash_curr:
                    snphash_curr[key1]=str(snphash_curr[key1])+","+str(value1)
                else:
                    snphash_curr[key1]=str(value1)
        generations.update({snp: i+1 for snp in dnms if snp not in generations})  ##add in SNPs newly generated
        popsize_prev = popsize_curr
        snphash_prev=snphash_curr
        '''
        if (i+1) % 100 == 0:
            for keys in snphash_prev:
                print str(keys)+":"+str(snphash_prev[keys])
        '''
        
    f = open(outdir+'afs_simul_'+snptype+'_select'+str(selection)+'.generation.txt', "w")
    for n in range(nreps):
        idx1=np.random.randint(popsize_prev, size=samplesize)
        samples = {j: snphash_curr[idx] for j, idx in enumerate(idx1) if idx in snphash_curr}
        snpstr=','.join(samples.values())
        snps = str(snpstr).split(',')
        snp_counts = Counter(snps)
        snp_count_dict = dict(snp_counts)
        for snp in snp_count_dict.keys():
            if snp :
                f.write(str(selection)+" "+snptype+" "+str(n)+" "+str(snp)+" "+str(snp_count_dict[snp]) + ' '+str(generations[int(snp)])+'\n')
    f.close()
