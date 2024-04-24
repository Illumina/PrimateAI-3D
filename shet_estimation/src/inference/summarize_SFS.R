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






