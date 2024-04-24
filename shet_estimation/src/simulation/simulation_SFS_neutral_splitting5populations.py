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
'''                                                                                                                                                                           
The forward time population simulation assume neutral evolution.                                                                                                              
The goal is to find the best-fit historical demographic parameters.                                                                                                           
Assume the census population sizes listed below, 30 years per generation, de novo mutation rates,                                                                             
 and the unknown ratio between effective population size and the census population size.                                                                                      
You may use the effective population for your simulation, not the census population size.                                                                                     
r=Nc/Ne, the ratio between effective population size and the census population size.                                                                                          
nu represents growth rate.                                                                                                                                                    
                                                                                                                                                                              
Simulate population expansion history as follows:                                                                                                                             
??000 BC: 10,000 initial effective population size, ??000 census population size, Nc0=r*Ne0. Ne0=10,000                                                                       
??000 BC: ?? million census population size, Nc1=r*Ne1, Ne1=2Ne0, nu1=Ne1/Ne0 = Nefactor                                                                                      
1400 AD:  360 million census population size, Nc2=360m, Ne2=Nc2/r,nu2=Ne2/Ne1                                                                                                 
1700 AD:  620 million census population size, Nc3=620m, Ne3=Nc3/r,nu3=62/36                                                                                                   
2000 AD:  6.2 billion census population size, Nc4=6.2b, Ne4=Nc4/r,nu4=10                                                                                                      
                                                                                                                                                                              
USAGE                                                                                                                                                                         
The command to run the script for forward time population simulation under neutral evolution is :                                                                             
                                                                                                                                                                              
python  /path/to/source/simulation/simulation_SFS_neutral_splitting5populations.py \
     [popratio] [snptype] [Nefactor] [T1] [T2] [nreps] [outdir]               
                                                                                                                                                                              
The input paramters are explained below.
popratio: ratio between effective population size and the census population size, its optimal value is 30                                                                     
snptype: SNP type, take in values 'nonCpGTi', 'CpGTi', 'Tv', 'CpGTi_MethyHigh',and 'CpGTi_MethyLow'
Nefactor: the ratio Ne1/Ne0, its optimal value is 2.0. Ne0 is 10,000, initial effective population size. Ne1 is effective population size after the burn-in period.           
T1: Number of generations during the burn-in period, its optimal value is 3500                                                                                                
T2: Number of generations during the first expension period after burn-in, its optimal value is 530                                                                           
nreps: Number of output samples sampled from the simulated final population at present day. Note these samples have a sample size resembling gnomAD exome sample size, 123K.  
outdir: output directory                                                                                                                                                      
                                                                                                                                                                              
'''



import numpy as np
import sys

popratio=int(sys.argv[1]) ## r=Nc/Ne                                                                                                                          
snptype=sys.argv[2]   ##snptype include nonCpGTi, CpGTi, and Tv                                                                                               
Nefactor=float(sys.argv[3])  ##Nefactor = Ne1/Ne0

#gen=20  ###each generation is 30 years
T1 = int(sys.argv[4]) ##Number of generations during the burn-in period, its optimal value is 3500                                                            
T2 = int(sys.argv[5]) ##Number of generations during the first expension period after burn-in, its optimal value is 530                                       
nreps=int(sys.argv[6]) ##Number of samples sampled from the simulated final population at present day, e.g., nreps=1000                                       
outdir=sys.argv[7]  ##output directory                                                                                                                     

###Simulate 100,000 independent loci
seqlen=100000

###de novo mutation rates derived from trio studies 
nonCpGTirate=5.552238e-09
CpGTirate=9.57758845127557e-8
Tvrate=2.034197e-09
CpGTirate_Methy1=1.011751e-07
CpGTirate_Methy2=2.264356e-08

if snptype == 'nonCpGTi':
    mutrate=nonCpGTirate
elif snptype == 'CpGTi':
    mutrate=CpGTirate
elif snptype == 'Tv':
    mutrate=Tvrate
elif snptype == 'CpGTi_MethyHigh':
    mutrate=CpGTirate_Methy1
elif snptype == 'CpGTi_MethyLow':
    mutrate=CpGTirate_Methy2
else:
    print('wrong SNP type')


N_ref =10000*2  ###inital effective popultion size
nu1 = Nefactor ** (1. / T1)   
s1=popratio*1.0
nu2 = (36000./Nefactor/s1) ** (1./T2)
T3= 10 # int((1700-1400)/gen)
nu3 = (62./36.) ** (1./T3)
T4= 10  #int((2000-1700)/gen)
nu4= 10. **(1./T4)
T5 = T1+T2+T3+T4

#splitting into 5 sub populations 
npop=5

if __name__ == '__main__':
    #initial
    ndenovo = int(N_ref*seqlen*mutrate)
    pos0=np.random.randint((N_ref*seqlen), size=ndenovo)  
    snphash_prev={}
    for i in range(len(pos0)):
        key,value=divmod(pos0[i],seqlen)
        snphash_prev[key]=value
    popsize_prev=N_ref
    for i in range(T1):
        nu=nu1
        popsize_curr = int(popsize_prev*nu)
        snphash_curr={}
        idx = np.random.randint(popsize_prev, size=popsize_curr)
        for j in range(len(idx)):
            if idx[j] in snphash_prev:
                snphash_curr[j] = snphash_prev[ idx[j] ]
        ndenovo = int(popsize_curr*seqlen*mutrate)
        pos0=np.random.randint((popsize_curr*seqlen), size=ndenovo)
        for k in range(len(pos0)):
            key1,value1=divmod(pos0[k],seqlen)
            if key1 in snphash_curr:
                snphash_curr[key1]=str(snphash_curr[key1])+","+str(value1)
            else:
                snphash_curr[key1]=value1
        popsize_prev = popsize_curr
        snphash_prev=snphash_curr
    #5 sub populations
    for i in range(T1,T5):
        if i>=T1 and i< (T2+T1):
            nu=nu2
        elif i>=(T2+T1) and i< (T3+T2+T1):
            nu=nu3
        elif i>=(T3+T2+T1):
            nu=nu4
        popsize_curr = int(popsize_prev*nu)
        #if (i+1) % 100 == 0:
        #print str(i)+" "+str(popsize_prev) + " "+str(popsize_curr)
        ###sampling current genome
        snphash_curr={}
        popsize_step=int(popsize_prev/npop)
        popsize_curr_step=int(popsize_curr/npop)
        range_starts = list(range(0, popsize_prev, popsize_step))
        range_ends = list(range(popsize_step, popsize_prev + 1, popsize_step))
        idx = []
        for start, end in zip(range_starts, range_ends):
            sampled_interval = np.random.randint(low=start, high=end, size=popsize_curr_step)
            idx.extend(sampled_interval)
        n_remain=popsize_curr -npop * popsize_curr_step
        idx.extend( np.random.randint(low=range_starts[npop-1], high=range_ends[npop-1], size=n_remain) )
        for j in range(len(idx)):
            if idx[j] in snphash_prev:
                snphash_curr[j] = snphash_prev[ idx[j] ]
        #
        ###generate de novo mutations
        ndenovo = int(popsize_curr*seqlen*mutrate)
        #print str(i)+" "+str(popsize_prev) + " "+str(popsize_curr)+" "+str(ndenovo)
        pos0=np.random.randint((popsize_curr*seqlen), size=ndenovo)
        for k in range(len(pos0)):
            key1,value1=divmod(pos0[k],seqlen)
            if key1 in snphash_curr:
                snphash_curr[key1]=str(snphash_curr[key1])+","+str(value1)
            else:
                snphash_curr[key1]=value1
        popsize_prev = popsize_curr
        snphash_prev=snphash_curr
        '''
        if (i+1) % 100 == 0:
            for keys in snphash_prev:
                print str(keys)+":"+str(snphash_prev[keys])
        '''

    ####this sample size mimics the gnomAD exome sample size 123K
    samplesize=246000
    f = open(outdir+'afs_simul_5pop_synony_'+snptype+'_'+str(T1)+'_'+str(Nefactor)+'_'+str(popratio)+'.txt', "w")
    for n in range(nreps):
        idx1=np.random.randint(popsize_prev, size=samplesize)
        samples={}
        for j in range(len(idx1)):
            if idx1[j] in snphash_curr:
                samples[j] = snphash_curr[ idx1[j] ]
        d1={}
        for k1 in samples:
            snps = list(set(str(samples[k1]).split(',')))
            for i in range(len(snps)):
                if snps[i] in d1:
                    d1[snps[i]] += 1
                else:
                    d1[snps[i]] = 1
        for m1 in d1:
            f.write(str(T1)+' '+str(Nefactor)+' '+str(popratio)+' '+snptype+" "+str(n)+" "+str(m1)+" "+str(d1[m1]) + '\n')
    f.close()
    
