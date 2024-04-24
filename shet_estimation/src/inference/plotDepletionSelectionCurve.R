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

# This script plots the depletion-selection curve
# Usage:
#       Rscript  /path/to/source/inference/plotDepletionSelectionCurve.R \
#  		 [input_file] [outdir]
#
#The input paramters are explained below.
#input_file: input file
#outdir: output directory

args = commandArgs(trailingOnly=TRUE)
input_file = args[1]
outdir=args[2]
dir.create(outdir)


d1=read.csv(input_file, stringsAsFactors=F,header=F)
colnames(d1)=c("AC","Counts","selcoeff","frac")

d1$varfrac=1-d1$frac
d1$depletion=1-d1$varfrac/d1$varfrac[d1$selcoeff==0]
d2=d1[d1$selcoeff > 0, ]

png(file=paste(outdir,"selection_depletion_curve.png",sep=""),width=600,height=550)
plot(log10(d2$selcoeff), d2$depletion, type="l", lwd=2, col="blue", xlab="Selection coefficients (log10)",
	ylab="Fraction of variants filtered by selection")
points(log10(d2$selcoeff), d2$depletion, pch=3,cex=1.5, col="red")
grid()
dev.off()

png(file=paste(outdir,"depletion_selection_curve.png",sep=""),width=600,height=550)
plot(d2$depletion, log10(d2$selcoeff),  type="l", lwd=2,col="blue", ylab="Selection coefficients (log10)",
			 xlab="Fraction of variants filtered by selection")
points(d2$depletion, log10(d2$selcoeff),  pch=3,cex=1.5, col="red")
grid()
dev.off()



