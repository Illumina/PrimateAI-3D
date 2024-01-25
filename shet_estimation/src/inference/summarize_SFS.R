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
# This script counts the number of variants with different allele count cutoff in the simulated output files
# Usage:
#	Rscript  /path/to/source/inference/summarize_SFS.R \
#		 [selcoeff] [trinuc] [indir] [outdir]
#
#The input paramters are explained below.
#selection: selection coefficients used in the simulation
#snptype: one of the 192 tri-nucleotide contexts
#indir: input directory
#outdir: output directory


ACcutoff=2^(0:6)
nAC=length(ACcutoff)
totalsites=100000
nreps=1000

processData <- function(x){

    ##check if a file is empty or not
    info = file.info(x)
    if(info$size > 0){
        x1=gsub(".txt","",basename(x))
	tnuc=strsplit(x1, split="_")[[1]][3]
#	print(tnuc)
        d1=read.delim(x,header=F,sep=" ",stringsAsFactors=F)
    	colnames(d1)= c("count","selcoeff","snptype","rep","AC")

	df1=NULL
	for(k in 1:nreps){    
	     d2=d1[d1$rep== (k-1),]
	     cnt1=rep(0, (nAC+1))
	     for(i in 1:nAC){
        	 if(i==1){
		     cnt1[i]=sum(d2[d2$AC == ACcutoff[i], "count"])
         	 }else{
             	     cnt1[i]=sum(d2[d2$AC <= ACcutoff[i] & d2$AC > ACcutoff[i-1], "count"])
                 }
    	     }
    	     cnt1[nAC+1] = sum(d2[d2$AC > ACcutoff[nAC], "count"])
             dft=data.frame(trinuc=tnuc, rep=k, AC=c(0,ACcutoff,"rest"), count=c((totalsites-sum(cnt1)), cnt1))
	     colnames(dft)=c("trinuc","rep","AC","countSites")
	     df1=rbind(df1,dft)
        }
        return(df1)
    }else{ print(x) }
}


args = commandArgs(trailingOnly=TRUE)
selcoeff = args[1]
trinuc = args[2]
indir = args[3]
outdir=args[4]
dir.create(outdir)

lf1=list.files(indir, pattern=paste("afs_simul_",trinuc,"_select",selcoeff,"_summ.txt",sep=""),full.names=T, recursive=T)

ldf1=lapply(lf1,processData )
m1 = do.call(rbind,ldf1)
m1=as.data.frame(m1)
m1$selection = selcoeff

write.csv(m1, file=paste(outdir, "afs_simul_",trinuc,"_select",selcoeff,".csv",sep=""),quote=F,row.names=F)






