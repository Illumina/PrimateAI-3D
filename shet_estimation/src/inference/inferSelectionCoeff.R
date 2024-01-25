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





