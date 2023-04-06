
################################################################################
# define some data paths
################################################################################
datafolder=./
trait_name=Test_trait.IRNT
ancestry_paths="$datafolder/samples.ancestry1.txt.gz
                $datafolder/samples.ancestry2.txt.gz
                $datafolder/samples.ancestry3.txt.gz
                $datafolder/samples.ancestry4.txt.gz"

phenotype_path=$datafolder/simulated.phenotypes.db
bgen_path=$datafolder/simulated.array_genotypes.bgen
exome_path=$datafolder/simulated.exome.db
gencode_path=$datafolder/simulated.gencode.gtf.gz

# sample subsets
training_path=$datafolder/samples.training.txt.gz
testing_path=$datafolder/samples.testing.txt.gz
training_ancestry1=$datafolder/samples.train_ancestry1.txt.gz
testing_ancestry1=$datafolder/samples.test_ancestry1.txt.gz

# result paths
burden_results=$datafolder/results.burden_test.txt.gz
prs_results=$datafolder/results.prs_values.txt.gz
prs_model=$datafolder/results.prs_model.json

################################################################################
# generate synthetic datasets
################################################################################
python $datafolder/make_synthetic_datasets.py \
    --phenotypes $phenotype_path \
    --pheno-name $trait_name \
    --array-genotypes $bgen_path \
    --exome-genotypes $exome_path \
    --training $training_path \
    --testing $testing_path \
    --ancestry-samples $ancestry_paths \
    --gencode $gencode_path

# intersect training/testing groups with the primary ancestry group
zcat $training_path $datafolder/samples.ancestry1.txt.gz | sort -h | uniq -d | gzip > $training_ancestry1
zcat $testing_path $datafolder/samples.ancestry1.txt.gz | sort -h | uniq -d | gzip > $testing_ancestry1

################################################################################
# run burden testing, not included in this folder
################################################################################
ukb_rare_burden \
    --geno-db $exome_path \
    --pheno-db $phenotype_path \
    --subset $training_ancestry1 \
    --phenotype $trait_name \
    --score-type primateai \
    --optimize \
    --max-af 0.001 \
    --output $burden_results

################################################################################
# construct rare variant PRS
################################################################################

# swap to ancestry samples that do not overlap the training group
ancestry_paths="$raining_ancestry1
                $datafolder/samples.ancestry2.txt.gz
                $datafolder/samples.ancestry3.txt.gz
                $datafolder/samples.ancestry4.txt.gz"

# run rare variant PRS
rvPRS \
  --rare-results $burden_results \
  --score-type primateai \
  --exome-db $exome_path \
  --pheno-db $phenotype_path \
  --train-samples $training_ancestry1 \
  --test-samples $testing_ancestry1 \
  --ancestry-samples $ancestry_paths \
  --trait $trait_name \
  --gencode $gencode_path \
  --output $prs_results \
  --output-model $prs_model
