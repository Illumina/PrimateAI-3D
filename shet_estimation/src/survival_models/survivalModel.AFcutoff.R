# Copyright 2024 Illumina Inc, San Diego, CA                                                                                                                            \

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
#The command to run the script for performing survival analysis using shet burden on UKBB subjects and their parents is
#
#Rscript  /path/to/source/survival_models/survivalModel.AFcutoff.R   \
#     [shet_cutoff] [AF_cutoff] [include_chrX] [pheno_file] [burden_file] \
#     [output_file] 
#
#The input paramters are explained below.
#shet_cutoff : the cutoff for selection coefficient. Ususally set shet_cutoff at 0.001.                                                                                                
#AF_cutoff : the cutoff for allele frequency. Ususally set AF_cutoff at 0.0001.                                                                                                          
#include_chrX : indicating whether variants on chrX will be included in the shet burden calculation. The values can be 'Yes' or 'No'.                                            
#pheno_file : the input file for individual clinical information
#burden_file : the input file for individual shet burden   
#output_file : the output file for survival analysis results
#########################


library(dplyr)
library(MASS)
library(survival)
library(data.table)


args = commandArgs(trailingOnly=TRUE)
shet_cutoff=args[1]
AF_cutoff=args[2]
include_chrX=args[3]
pheno_file=args[4]
burden_file=args[5]
output_file=args[6]


d1=fread(pheno_file,header=T,sep="\t")
d1$age2=d1$age_at_recruitment_f21022_0_0 * d1$age_at_recruitment_f21022_0_0
d1=d1[(d1$year_of_birth_f34_0_0>=1934) &(d1$year_of_birth_f34_0_0<=1970)
 &(d1$genetic_ethnic_grouping_f22006_0_0=='Caucasian'),]
df_parent=d1[(!is.na(d1$father_curr_age)) & ( d1$father_curr_age >=15 )
              &(!is.na(d1$mother_curr_age)) &  (d1$mother_curr_age >=15)& (d1$adopted_as_a_child_f1767_0_0 != 'Yes'),]
df_subject=d1

sel=fread(burden_file,header=T,sep="\t")

d3=merge(sel, df_subject,by.x='sampleid',by.y='eid', all.x=F,all.y=T)
d_3=merge(sel, df_parent,by.x='sampleid',by.y='eid', all.x=F,all.y=T)
d3$burden[is.na(d3$burden)]=0
d_3$burden[is.na(d_3$burden)]=0


covariate_vec=paste0("genetic_principal_components_f22009_0_",1:40)
covariate_string = paste(covariate_vec, collapse = " + ")

self_cox1<-function(x){
    self.cox <- coxph(as.formula( paste0('Surv(self_surv_age_2023, self_surv_status_2023) ~ burden+birth_year_cohort+
          age_at_recruitment_f21022_0_0 + age2 + gender+',covariate_string)),
          data = x)
    HR1=as.numeric(coef(summary(self.cox))[, "exp(coef)"][1])
    sd1=as.numeric(coef(summary(self.cox))[, "se(coef)"][1])
    zval=as.numeric(coef(summary(self.cox))[, "z"][1])
    pval=as.numeric(coef(summary(self.cox))[, "Pr(>|z|)"][1])
    return(c(HR1,sd1,zval,pval))
}



father_cox1<-function(x){
        self.cox <- coxph(as.formula( paste0("Surv(father_curr_age, father_curr_status) ~ burden+birth_year_cohort+
                age_at_recruitment_f21022_0_0 + age2 + gender+",covariate_string)),
                                    data = x)
    HR1=as.numeric(coef(summary(self.cox))[, "exp(coef)"][1])
    sd1=as.numeric(coef(summary(self.cox))[, "se(coef)"][1])
    zval=as.numeric(coef(summary(self.cox))[, "z"][1])
    pval=as.numeric(coef(summary(self.cox))[, "Pr(>|z|)"][1])
    return(c(HR1,sd1,zval,pval))
}

mother_cox1<-function(x){
        self.cox <- coxph(as.formula( paste0("Surv(mother_curr_age, mother_curr_status) ~ burden+birth_year_cohort+
                age_at_recruitment_f21022_0_0 + age2 + gender+",covariate_string)),
                                    data = x)
    HR1=as.numeric(coef(summary(self.cox))[, "exp(coef)"][1])
    sd1=as.numeric(coef(summary(self.cox))[, "se(coef)"][1])
    zval=as.numeric(coef(summary(self.cox))[, "z"][1])
    pval=as.numeric(coef(summary(self.cox))[, "Pr(>|z|)"][1])
    return(c(HR1,sd1,zval,pval))
}


res1=rbind(self_cox1(d3),self_cox1(d3[d3$gender==1,]),self_cox1(d3[d3$gender==0,]),
  father_cox1(d_3),  mother_cox1(d_3))
res1=as.data.frame(res1)
colnames(res1)=c("HR","beta_sd","t_value","p_value")

res1$survival_type=c("self","male","female","father","mother")
res1$shet_cutoff = shet_cutoff
res1$n_sample=c( nrow(d3), nrow(d3[d3$gender==1,]),nrow(d3[d3$gender==0,]),
   nrow(d_3),nrow(d_3)) 
res1$n_death=c( sum(d3$self_surv_status_2023==1),sum(d3[d3$gender==1,]$self_surv_status_2023==1),
  sum(d3[d3$gender==0,]$self_surv_status_2023==1),sum(d_3$father_curr_status==1),
  sum(d_3$mother_curr_status==1) )

res1$dead_fraction=res1$n_death / res1$n_sample


res2=res1[,c("survival_type","selcoeff", "n_sample", "n_death",
 "dead_fraction", "HR","beta_sd","t_value","pvalue")]
res2$AF_cutoff=AF_cutoff
res2$include_chrX=include_chrX
write.table(res2,file=output_file,quote=F,row.names=F,sep="\t")



