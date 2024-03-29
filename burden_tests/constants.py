
## Data folders

ROOT_PATH = '' # set correct path
ROOT_PATH_SD = '' # set correct path
ROOT_DNANEXUS = '' # set correct path

CLUSTER_ROOT = ''

DATA_PATH = '../../data/'
GNOMAD_EXOMES_PATH = DATA_PATH + 'gnomad.exomes.r2.1'
GNOMAD_GENOMES_PATH = DATA_PATH + 'gnomad.genomes.r2.1'

GENCODE_PATH = DATA_PATH + 'gencode'

PRIMATEAI_PATH = DATA_PATH + 'primateai'

TOPMED_PATH = '../../data/topmed'

# VCF fields
VCF_VEP = 'vep'
VCF_CSQT = 'CSQT'
VCF_CSQR = 'CSQR'
VCF_AF = 'AF'
VCF_AC = 'AC'
VCF_AN = 'AN'
VCF_PASS = 'PASS'
VCF_MISSENSE_VARIANT = 'missense_variant'
VCF_SYNONYMOUS_VARIANT = 'synonymous_variant'
VCF_FRAMESHIFT_VARIANT = 'frameshift_variant'
VCF_STOP_GAINED = 'stop_gained'
VCF_FILTER = 'FILTER'
VCF_QUAL = 'QUAL'

VCF_STOP_LOST = 'stop_lost'
VCF_NMD_TRANSCRIPT_VARIANT = 'NMD_transcript_variant'
VCF_INFRAME_DELETION = 'inframe_deletion'
VCF_SPLICE_REGION_VARIANT = 'splice_region_variant'
VCF_SPLICE_DONOR_VARIANT = 'splice_donor_variant'
VCF_SPLICE_ACCEPTOR_VARIANT = 'splice_acceptor_variant'
VCF_START_LOST = 'start_lost'
VCF_CONSEQUENCE = 'consequence'

EQTL = 'eQTL'

CODING = 'coding'
NON_CODING = 'non_coding'
SPLICING = 'splicing'

DELETERIOUS_VARIANT = 'DELETERIOUS_VARIANT'
DELETERIOUS_EXCEPT_PTV = 'DELETERIOUS_EXCEPT_PTV'
DELETERIOUS_MISSENSE = 'DELETERIOUS_MISSENSE'
MISSENSE_AND_PTVS = 'MISSENSE_AND_PTVS'

VCF_TRANSCRIPT_ABLATION = 'transcript_ablation'
VCF_TRANSCRIPT_TRUNCATION = 'transcript_truncation'

VCF_EXON_LOSS_VARIANT = 'exon_loss_variant'
VCF_GENE_FUSION_VARIANT = 'gene_fusion'
VCF_BIDIRECTIONAL_GENE_FUSION = 'bidirectional_gene_fusion'


ALL_PTV = [VCF_STOP_GAINED,
           VCF_STOP_LOST,
           VCF_FRAMESHIFT_VARIANT,
           VCF_SPLICE_DONOR_VARIANT,
           VCF_SPLICE_ACCEPTOR_VARIANT,
           VCF_START_LOST,
           VCF_TRANSCRIPT_ABLATION,
           VCF_TRANSCRIPT_TRUNCATION,
           VCF_EXON_LOSS_VARIANT,
           VCF_GENE_FUSION_VARIANT,
           VCF_BIDIRECTIONAL_GENE_FUSION
           ]

ALL_PTV_EXCEPT_SPLICE_VARIANTS = [VCF_STOP_GAINED,
           VCF_STOP_LOST,
           VCF_FRAMESHIFT_VARIANT,
           VCF_START_LOST,
           VCF_TRANSCRIPT_ABLATION,
           VCF_TRANSCRIPT_TRUNCATION,
           VCF_EXON_LOSS_VARIANT,
           VCF_GENE_FUSION_VARIANT,
           VCF_BIDIRECTIONAL_GENE_FUSION
           ]

SPLICE_VARIANTS = [VCF_SPLICE_DONOR_VARIANT,
                   VCF_SPLICE_ACCEPTOR_VARIANT]


VCF_MISSING_FRACTION = 'MISSING_FRACTION'
VCF_N_MISSING_SAMPLES = 'N_MISSING_SAMPLES'
VCF_N_TOTAL_SAMPLES = 'N_TOTAL_SAMPLES'

VCF_CHROM = 'CHROM'
VCF_POS = 'pos'
VCF_ID = 'ID'
VCF_REF = 'REF'
VCF_ALT = 'ALT'
VCF_QUAL = 'QUAL'
VCF_RSID = 'RSID'
VARID_REF_ALT = 'varid_ref_alt'
VARID = 'varid'
QC_AF = 'QC_AF'

START = 'start'
END = 'end'

ANCESTRAL_ALLELE = 'ancestral_allele'

REF_AA = 'REF_AA'
ALT_AA = 'ALT_AA'

REFSEQ_ID = 'REFSEQ_ID'


MEAN_COVERAGE = 'MEAN_COVERAGE'
MEDIAN_COVERAGE = 'MEDIAN_COVERAGE'
COVERAGE_HISTOGRAM = 'COVERAGE_HISTOGRAM'

SAMPLE_ID = 'sample_id'
SAMPLE_STATUS = 'status'
VARIANT_IDX = 'index'
GENE_NAME = 'gene_name'
TRANSCRIPT_ID = 'transcript_id'
CONSEQUENCE = 'consequence'
BIOTYPE = 'biotype'
IS_EUROPEAN = 'is_european'

SHUFFLED_IDX = 'shuffled_idx'
IS_REAL_DATA = 'is_real_data'

ALL_SAMPLES = 'all_samples'
HETEROZYGOTES = 'heterozygotes'
HOMOZYGOTES = 'homozygotes'
MISSING = 'missing'

MALE = 'male'
FEMALE = 'female'
BOTH = 'both'

# integer to encode missing genotypes in sparse matrices
MISSING_GENOTYPE = 3

VAR_INFO = 'VAR_INFO'
VAR_ANNO = 'VAR_ANNO'
GENOTYPE_SDF = 'GENOTYPE_SDF'

FILE_NAME = 'file_name'

BLOOD_BIOMARKER = 'BLOOD_BIOMARKER'
URINE_BIOMARKER = 'URINE_BIOMARKER'
BLOOD_CELLTYPE_BIOMARKER = 'BLOOD_CELLTYPE_BIOMARKER'
NIGHTINGALE_HEALTH_METABOLOME = 'NIGHTINGALE_HEALTH_METABOLOME'
OTHER = 'OTHER'
BINARY = 'BINARY'
ICD10 = 'ICD10'
ICD10_PHENOTYPE_CODE = 10 ** 10
RANDOM = 'random'
GET_ALL_COVARIATES = 'GET_ALL_COVARIATES'

AGE_1st_visit = '21003-0.0'
AGE_2nd_visit = '21003-1.0'

REVCOMP = {'A': 'T',
           'T': 'A',
           'C': 'G',
           'G': 'C'}



UKB_STATINS = {'1141146234': 'atorvastatin',
               '1141192414': 'crestor 10mg tablet',
               '1140910632': 'eptastatin',
               '1140888594': 'fluvastatin',
               '1140864592': 'lescol 20mg capsule',
               '1141146138': 'lipitor 10mg tablet',
               '1140861970': 'lipostat 10mg tablet',
               '1140888648': 'pravastatin',
               '1141192410': 'rosuvastatin',
               '1141188146': 'simvador 10mg tablet',
               '1140861958': 'simvastatin',
               '1140881748': 'zocor 10mg tablet',
               '1141200040': 'zocor heart pro 10mg tablet'}


UKB_SAMPLES_TO_EXCLUDE = [] # add correct samples

PRIMATEAI_SCORE = 'primateAI score'
PRIMATEAI_UCSC = 'UCSC gene'
MIN_DELETERIOUS_SPLICEAI_SCORE = 0.2
MIN_DELETERIOUS_PRIMATEAI_SCORE_QUANTILE = 0.5


SPLICEAI_MAX_SCORE = 'max spliceAI score'
SPLICEAI_DS_AG = 'spliceAI DS_AG'
SPLICEAI_DS_AL = 'spliceAI DS_AL'
SPLICEAI_DS_DG = 'spliceAI DS_DG'
SPLICEAI_DS_DL = 'spliceAI DS_DL'

GNOMAD_MEDIAN_COVERAGE = 'median_coverage/gnomAD'
GNOMAD_AF = 'AF/gnomAD'
GNOMAD_AC = 'AC/gnomAD'
GNOMAD_AN = 'AN/gnomAD'
TOPMED_AF = 'AF/topmed'

GNOMAD_POPMAX = 'gnomAD/popmax'
GNOMAD_POPMAX_AF = 'AF/gnomAD/popmax'
GNOMAD_POPMAX_AC = 'AC/gnomAD/popmax'
GNOMAD_POPMAX_AN = 'AN/gnomAD/popmax'

IS_CANONICAL = 'CANONICAL'

SPLIT_SAMPLE_IDS_BY_UNDERSCORE = '__SPLIT_SAMPLE_IDS_BY_UNDERSCORE__'

RANKSUM_TEST = 'ranksum_test'
T_TEST = 't_test'
LOGIT = 'logit'
REGRESSION = 'regression'
CHI2_TEST = 'chi2_test'
CHI2_FOLLOWED_BY_LOGIT = 'chi2_test_followed_by_logit'
LOGIT_ADJUSTED_BY_CHI2 = 'logit_adjusted_by_chi2'
CHI2_FOLLOWED_BY_SIDAK_ADJUSTED_LOGIT = 'chi2_test_followed_by_Sidak_adjusted_logit'
ACAT_TEST = 'ACAT_test'

AUTOSOMES = 'authosomes'

SPHERICAL_EQUIVALENT_CODE = 46123786
SPHERICAL_EQUIVALENT_CODE_LEFT = 46123787
SPHERICAL_EQUIVALENT_CODE_RIGHT = 46123788

AGE = 'age'

TCGA_GERMLINE_PATH = DATA_PATH + 'tcga_germline'
TCGA_METAINFO_PATH = '../../data/tcga_metadata'

TUMOR_SUPPRESSORS_LI_DING = ['APC', 'ATM', 'ATR', 'BLM', 'BMPR1A', 'BAP1', 'BRIP1', 'BRCA1', 'BRCA2', 'BUB1B', 'CDH1', 'CDC73', 'CHEK2', 'CDKN2A', 'CYLD', 'DDB2', 'DICER1', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'ERCC1', 'EXT1', 'EXT2', 'FANCI', 'FANCL', 'FANCM', 'FANCA', 'FANCC', 'FANCD2', 'FANCE', 'FANCF', 'FANCG', 'FH', 'HNF1A', 'MEN1', 'MLH1', 'MSH2', 'MSH6', 'MUTYH', 'MAX', 'NF1', 'NF2', 'NBN', 'PHOX2B', 'PALB2', 'PTCH1', 'PRF1', 'PTEN', 'PMS1', 'PMS2', 'POLD1', 'POLE', 'POLH', 'RAD51C', 'RAD51D', 'RECQL4', 'RB1', 'STK11', 'SBDS', 'SMAD4', 'SDHAF2', 'SDHB', 'SDHC', 'SDHD', 'SUFU', 'SMARCA4', 'SMARCB1', 'TSC1', 'TSC2', 'TP53', 'VHL', 'WRN', 'WT1', 'XPA', 'XPC']
ONCOGENES_LI_DING = ['ALK', 'CBL', 'CEBPA', 'CDK4', 'EGFR', 'ETV6', 'GATA2', 'ITK', 'LMO1', 'MET', 'MITF', 'MAP2K1', 'MAP2K2', 'MPL', 'NRAS', 'PAX5', 'PDGFRA', 'PTPN11', 'RET', 'RUNX1', 'SETBP1', 'STAT3', 'TERT', 'TSHR', 'HRAS', 'KRAS', 'KIT', 'BRAF', 'RAF1']

FAMILIAL_CANCER_GENES = {'APC', 'ATM', 'BARD1', 'BMPR1A', 'BRCA1', 'BRCA2', 'BRIP1', 'CDH1', 'CDK4', 'CDKN2A', 'CHEK2', 'DICER1', 'EPCAM', 'GREM1', 'HOXB13', 'MLH1', 'MRE11A',
                         'MSH2', 'MSH6', 'MUTYH', 'NBN', 'NF1', 'PALB2', 'PMS2', 'POLD1', 'POLE', 'PTEN', 'RAD50', 'RAD51C', 'RAD51D', 'SMAD4', 'SMARCA4', 'STK11', 'TP53', 'VHL'}

CVD_GWAS_GENES = ["ABCG8", "ADAMTS7", "AGT", "ANGPTL4", "APOB", "APOE", "ARHGEF26", "ARNTL", "ATP2B1", "C1S", "CARF", "CCDC158", "CCM2", "CDH13", "CENPN", "CENPW", "CYP46A1", "DAB2IP", "DAGLB", "DHX58", "DNAJC13", "EDNRA", "EHBP1L1", "FES", "FN1", "GIP", "GUCY1A3", "GUCY1C3", "HAPLN3", "HECTD4", "HGFAC", "HHIPL1", "HNF1A", "IL6R", "KCNK5", "KIAA1462", "LAMB2", "LDLR", "LIPA", "LIPG", "LMOD1", "LPA", "LPL", "MAP1S", "MLH3", "MRAS", "NOS1", "NOS3", "PARP12", "PCSK9", "PRDM16", "PROCR", "SEMA5A", "SERPINA1", "SERPINH1", "SH2B3", "SIPA1", "SMAD3", "SMG6", "SWAP70", "TCF21", "TNS1", "TRIM5", "TRIP4", "ZC3HC1"]
