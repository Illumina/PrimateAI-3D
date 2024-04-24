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

# This script infers selection coefficients given a vector of depletion metrics by applying linear interpolation to the depletion-selection curve
# Usage:
#       Rscript  /path/to/source/inference/inferSelectionCoeff.R \
#  		 [input_file] [depletion_file] [output_file]
#
#The input paramters are explained below.
#input_file: input file containing relationship between depletion metrics and selection coefficients 
#depletion_file: input file containing depletion metrics that need to be converted to selection coefficients
#output_file: output directory

args = commandArgs(trailingOnly=TRUE)
input_file = args[1]
depletion_file=args[2]
output_file=args[3]

depdf=read.csv(depletion_file, stringsAsFactors=F,header=T)

d1=read.csv(input_file, stringsAsFactors=F,header=F)
colnames(d1)=c("AC","Counts","selcoeff","frac")

d1$varfrac=1-d1$frac
d1$depletion=1-d1$varfrac/d1$varfrac[d1$selcoeff==0]
d2=d1[d1$selcoeff > 0, ]
d2$selection = log10(d2$selcoeff)

s1=10^( approx(x = d2$depletion,y = d2$selection,xout=depdf$depletion, rule=2, method="linear")$y )
s1[s1 < 0] = 0
s1[s1 > 1] = 1

depdf$selcoeff =s1
write.table(depdf, file=output_file,quote=F,row.names=F,sep="\t")





