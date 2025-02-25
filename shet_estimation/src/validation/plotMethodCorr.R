
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

#########################
#USAGE
#The command to run the script for plotting Spearman correlation among scores of gene constraintmethods is :
#
#Rscript  /path/to/source/validation/plotMethodCorr.R \
#     [datafile] 
#
#The input paramters are explained below.
#datafile contains gene contraint scores merged from multiple methods, including \
#'selbin10', 'selPTV', 'selMax' from our methods,
#'pLI','oe_lof_upper' from 'gnomad.v2.1.1.lof_metrics.by_gene.txt', 
#'shet_weghorn' from 'Supplementary_Table1.txt' in Weghorn et al. 2019,
#'GeneBayes' from 'GeneBayes_supptable2.tsv' in Zeng et al. Biorxiv,
#'shet_molly' from 'elife-83172-supp2-v2.txt' in Agarwal et al. 2023,
#'shet_regeneron' from 'SuppTable2.txt' in Sun et al. Biorxiv.
#########################



library(ggplot2)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)
datafile=args[1]

m1=read.csv(datafile,stringsAsFactors=F,header=T,sep="\t")


cor1=cor(m1$shet_weghorn,m1[,c('selbin1','selbin10','selPTV')], method='spearman',use="pairwise.complete.obs")
cor2=cor(m1$pLI,m1[,c('selbin1','selbin10','selPTV')], method='spearman',use="pairwise.complete.obs")
cor3=cor(m1$oe_lof_upper,m1[,c('selbin1','selbin10','selPTV')], method='spearman',use="pairwise.complete.obs")
cor4=cor(m1$shet_molly,m1[,c('selbin1','selbin10','selPTV')], method='spearman',use="pairwise.complete.obs")
cor5=cor(m1$shet_regeneron,m1[,c('selbin1','selbin10','selPTV')], method='spearman',use="pairwise.complete.obs")
cor6=cor(m1$GeneBayes,m1[,c('selbin1','selbin10','selPTV')], method='spearman',use="pairwise.complete.obs")

method1=c('s_het (Weghorn 2019)','pLI','| LOEUF |','GeneBayes','s_het (Agarwal 2023)','s_het (Sun Biorxiv)')
dfPTV=data.frame(method=method1, corr=c(cor1[3],cor2[3],cor3[3],cor6[3],cor4[3],cor5[3]))
dfPTV$mode='PTV'

dfmis=data.frame(method=method1, corr=c(cor1[2],cor2[2],cor3[2],cor6[2],cor4[2],cor5[2]))
dfmis$mode='PrimateAI-3D top 10%\nmissense'

dfmis1=data.frame(method=method1, corr=c(cor1[1],cor2[1],cor3[1],cor6[1],cor4[1],cor5[1]))
dfmis1$mode='PrimateAI-3D bottom 10%\nmissense'

df1=rbind(dfmis,dfPTV,dfmis1)
df1$abs_corr=abs(df1$corr)

df1$mode=factor(df1$mode,levels=c('PrimateAI-3D bottom 10%\nmissense','PrimateAI-3D top 10%\nmissense','PTV'))
df1$method=factor(df1$method,levels=c('GeneBayes','s_het (Sun Biorxiv)','s_het (Agarwal 2023)','s_het (Weghorn 2019)','pLI','| LOEUF |'))
label1=c(expression(~italic("s")[het]*" [ missense bottom 10% ]"),expression(~italic("s")[het]*" [ missense top 10% ]"),
   expression(~italic("s")[het]*" [ LOF ]"))


pdf("Figure_Spearman_correlation.pdf",width=6.5, height=6.5)
ggplot(df1, aes(x = mode, y = abs_corr, fill = method)) +
  geom_bar(stat = "identity", position = "dodge",alpha=0.6) +
  xlab('')+ylab('Spearman correlation')+theme_bw()+
  ylim(0,1)+scale_fill_manual(values=c("darkorange","green", "purple",'gray','red','blue'),labels=c(expression(~italic("s")[het]*" (GeneBayes)"),
    expression(~italic("s")[het]*" (Sun Biorxiv)"), expression(~italic("s")[het]*" (Agarwal 2023)"),
    expression(~italic("s")[het]*" (Weghorn 2019)"),expression('pLI'),expression('| LOEUF |')))+
  scale_x_discrete(labels = label1)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+  #xlim(0,1.0)+
  theme(panel.border = element_blank(),axis.line = element_line())+ theme(legend.title = element_blank())+ 
  theme(legend.position = c(0.18,0.85))+
  theme(legend.text.align = 0)
dev.off()
