
import argparse
import gzip
import sqlite3

import numpy

from bgen import BgenWriter

from rvPRS.rare.id_decoder import to_bytes
from rvPRS.rare.consequence import PTV, MISSENSE, SYNONYMOUS

CONSEQUENCES = {'ptv': PTV, 'missense': MISSENSE, 'synonymous': SYNONYMOUS}

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--array-genotypes', required=True, 
                        help='path to bgen file to write synthetic common variants to')
    parser.add_argument('--exome-genotypes', required=True,
                        help='path to sqlite3 db file to write synthetic exome variants to')
    parser.add_argument('--phenotypes', required=True,
                        help='path to file to write phenotypes to')
    parser.add_argument('--training', required=True,
                        help='path to write training samples to')
    parser.add_argument('--testing', required=True,
                        help='path to write testing samples to')
    parser.add_argument('--ancestry-samples', nargs='*',
                        help='paths to write samples to according to ancestries to')
    parser.add_argument('--gencode', required=True,
                        help='path to write gencode GTF to')
    parser.add_argument('--n-exome-variants', type=int, default=1000)
    parser.add_argument('--n-array-variants', type=int, default=1000)
    parser.add_argument('--cohort-size', type=int, default=10000)
    parser.add_argument('--pheno-name', default='Test_trait.IRNT')
    return parser.parse_args()

def simulate_sample_ids(n: int):
    ''' create array of random sample IDs
    '''
    max_val = 8000000
    return numpy.array(list(map(str, numpy.random.choice(max_val, n, replace=False))))

def simulate_phenotype(n: int):
    return numpy.random.random(n).astype(numpy.float32)

def make_correlated(arr: numpy.array, r2: float):
    ''' given a numpy array, make another array with some correlation
    '''
    y = numpy.random.random(len(arr))
    y -= r2 * (y - arr)  # shrink values in 2nd array towards the first
    return y.astype(numpy.float32)

def binarize(arr: numpy.array, quantile: float=0.5):
    ''' binarize values in a numpy array to 0 or 1, based on their quantile
    '''
    cutpoint = numpy.quantile(arr, quantile)
    return (arr > cutpoint).astype(numpy.float32)

def simulate_genotypes(pheno: numpy.array, r2: float, af: float=0.05):
    ''' simulate genotype dosages correlated with a phenotype at an AF
    '''
    # ensure genotypes are sampled in HWE
    hom_ref_n = round(len(pheno) * (1 - af) ** 2)
    het_n = round(len(pheno) * 2 * af * (1 - af))
    # get correlated array, and figure out cutpoints to assign genotypes
    dose = make_correlated(pheno, r2)
    hom_ref_cut = numpy.quantile(dose, hom_ref_n / len(pheno))
    het_cut = numpy.quantile(dose, (hom_ref_n + het_n) / len(pheno))
    # set genotype dosages based on their quantile
    dose[dose < hom_ref_cut] = 0
    dose[dose > het_cut] = 2
    dose[(dose > hom_ref_cut) & (dose <= het_cut)] = 1
    return dose

def dosage_to_probs(arr: numpy.array):
    ''' convert genotype dosage array to genotype probabilities
    '''
    probs = numpy.zeros((len(arr), 3))
    probs[arr == 0, 0] = 1  # set homozygous refs
    probs[arr == 1, 1] = 1  # set heterozygotes
    probs[arr == 2, 2] = 1  # set homozygous alts
    return probs

def create_phenotype_file(path: str, sample_ids: numpy.array, phenotype: numpy.array, pheno_name: str):
    with gzip.open(path, 'wt') as output:
        output.write(f'sample_id\t{pheno_name}\n')
        for sample_id, pheno in zip(sample_ids, phenotype):
            output.write(f'{sample_id}\t{pheno}\n')

def simulate_array_genotypes(bgen_path: str, sample_ids: numpy.array, n_variants: int, pheno: numpy.array):
    ''' construct a bgen file with simulated genotypes
    '''
    alleles = list('ACTG')
    idx = 0
    with BgenWriter(bgen_path, len(sample_ids), samples=sample_ids) as bfile:
        while idx < n_variants:
            # simulate genotypes, but under a model where at least some of them
            # are expected to be significant
            r2 = numpy.random.beta(0.01, 10)
            af = numpy.random.beta(0.1, 10)
            dose = simulate_genotypes(pheno, r2, af)
            # skip rare variants for array variants
            if af < 0.005 or (dose.sum() / (2 * len(dose))) < 0.005:
                continue
            idx += 1
            probs = dosage_to_probs(dose)
            # simulate alleles and write variant to disk
            _alleles = numpy.random.choice(alleles, 2, replace=False)
            bfile.add_variant(f'var{idx+1}', f'rs{str(idx+1).zfill(4)}',
                              f'chr{1 + idx // (n_variants / 22)}', idx + 1,
                              _alleles, probs)

def create_variants_table(conn: sqlite3.Connection):
    ''' create a variants table
    '''
    conn.execute('''CREATE TABLE IF NOT EXISTS variants (varid text, rsid text,
        chrom text, pos int, ref text, alt text, af real, ac int, an int, 
        missing blob, homs blob, hets blob,
        UNIQUE(chrom, pos, ref, alt))''')

def create_annotations_table(conn: sqlite3.Connection):
    ''' create table with all the per gene annotations
    '''
    conn.execute('''CREATE TABLE IF NOT EXISTS annotations (varid int, symbol text, 
        consequence text, primateai real, spliceai real, gnomad_af real, topmed_af real,
        UNIQUE(varid, symbol, consequence),
        FOREIGN KEY(varid) REFERENCES variants(id))''')

def create_samples_table(conn: sqlite3.Connection, sample_ids: numpy.array):
    ''' create table with all sample IDs
    '''
    conn.execute('''CREATE TABLE IF NOT EXISTS samples (sample_id text)''')
    if conn.execute('''SELECT COUNT(*) FROM samples''').fetchone()[0] > 0:
        return
    for x in sample_ids:
        conn.execute('''INSERT INTO samples VALUES (?)''', (x, ))
    conn.commit()

def annotate_variant(conn: sqlite3.Connection, rowid: str, symbol: str, 
                     consequence: str, primateai: float, spliceai: float, 
                     gnomad_af: float, topmed_af: float):
    ''' insert annotations for a variant for a specific gene 
    ''' 
    query = f'''INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?)'''
    params = (rowid, symbol, consequence, primateai, spliceai, gnomad_af, topmed_af)
    conn.execute(query, params)

def encode_samples(samples):
    ''' encode samples as byte array (each sample as 24-bit signed int)
    
    Converts sample IDs to integer values, then byte-encode for lower storage/IO.
    '''
    if len(samples) == 0:
        return None
    return to_bytes(map(int, sorted(set(samples))))

def add_variant(conn: sqlite3.Connection, varid: str, rsid: str, chrom: str, 
                pos: int, ref: str, alt: str, af: float, ac: int, an: int, 
                missing: numpy.array, homs: numpy.array, hets: numpy.array):
    ''' insert variant data into db table
    '''
    missing = encode_samples(missing)
    homozygotes = encode_samples(homs)
    heterozygotes = encode_samples(hets)
    query = '''INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''' 
    params = (varid, rsid, chrom, pos, ref, alt, af, ac, an, missing, homozygotes, heterozygotes)
    conn.execute(query, params)
    return conn.lastrowid

def simulate_correlation(is_correlated: bool, cq_type: str):
    ''' pick a correlation for simulating genotypes
    
    The correlation depends on whether we've selected the gene to be correlated,
    and the variant consequence type (synonymous variants aren't correlated).
    '''
    if is_correlated and cq_type != 'synonymous':
        return numpy.random.beta(0.5, 0.5)
    else:
        return numpy.random.beta(0.01, 10)

def simulate_population_af(af: float):
    ''' simulate an external AF source, given some noise around the cohort AF
    '''
    noise = numpy.random.beta(0.01, 50)
    if numpy.random.random() > 0.5:
        noise *= -1
    return max(1e-6, af + noise)

def simulate_exome_genotypes(db_path: str, sample_ids: numpy.array, 
                             n_variants: int, pheno: numpy.array):
    ''' create a db containing simulated exome genotypes
    '''
    alleles = list('ACGT')
    variants_per_gene = 100
    signif_fraction = 0.25
    symbol_count = 1
    correlated_gene = False
    CQ_TYPES = ['ptv', 'synonymous', 'missense']

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        create_variants_table(conn)
        create_annotations_table(conn)
        create_samples_table(conn, sample_ids)
        idx = 0
        while idx < n_variants:
            cq_type = numpy.random.choice(CQ_TYPES, 1)[0]
            r2 = simulate_correlation(correlated_gene, cq_type)
            af = numpy.random.beta(0.01, 10)
            
            dose = simulate_genotypes(pheno, r2, af)
            if dose.sum() < 1:
                continue
            idx += 1

            homs = sample_ids[dose == 2]
            hets = sample_ids[dose == 1]
            missed_frac = numpy.random.beta(0.01, 10)
            missed = numpy.random.choice(sample_ids, round(len(sample_ids) * missed_frac))
            
            varid = f'var{idx+1}'
            rsid = f'rs{str(idx+1).zfill(4)}'
            chrom = f'chr{1 + idx // (n_variants / 22)}'
            pos = idx + 1
            ref, alt = numpy.random.choice(alleles, 2, replace=False)
            ac = int(dose.sum())
            an = 2 * (len(sample_ids) - len(missed))
            af = ac / an

            rowid = add_variant(cursor, varid, rsid, chrom, pos, ref, alt, 
                                    af, ac, an, missed, homs, hets)

            cq = numpy.random.choice(list(CONSEQUENCES[cq_type]), 1)[0]
            primateai = None
            if cq_type == 'missense':
                primateai = float(numpy.random.beta(0.5, 0.5))
            spliceai = float(numpy.random.beta(0.01, 1))
            if spliceai < 0.001:
                spliceai = None
            
            gnomad_af = simulate_population_af(af)
            topmed_af = simulate_population_af(af)
            annotate_variant(conn, rowid, f'GENE{symbol_count}', cq, primateai, spliceai, gnomad_af, topmed_af)
            if numpy.random.random() < (1 / variants_per_gene):
                symbol_count += 1
                # assess if the new gene should be correlated
                correlated_gene = numpy.random.random() < signif_fraction
    
    return symbol_count

def write_sample_ids(path, sample_ids):
    '''write list of sample IDs to disk
    '''
    opener = gzip.open if str(path).endswith('gz') else open
    with opener(path, 'wt') as output:
        output.write('sample_id\n')
        for sample_id in sorted(map(int, sample_ids)):
            output.write(f'{sample_id}\n')

def prepare_training_splits(sample_ids, train_path, test_path, train_frac=0.8):
    ''' make training and testing splits and write to disk
    '''
    train = numpy.random.choice(sample_ids, round(len(sample_ids) * train_frac), replace=False)
    train = set(train)
    test = set(sample_ids) - train
    write_sample_ids(train_path, train)
    write_sample_ids(test_path, test)

def ancestry_freqs(n: int):
    ''' simulate ancestry frequencies within the cohort
    
    This models the cohort as having one large subset of one ancestry, and 
    increasingly smaller groups.

    Args:
        n: number of ancestry groups to model
    '''
    freqs = [1 / (i + 1) ** 2 for i in range(n)]
    total = sum(freqs)
    return [x / total for x in freqs]

def make_ancestry_groups(sample_ids: numpy.array, ancestry_paths: list[str]):
    ''' simulate sample lists for ancestry groups
    '''
    freqs = ancestry_freqs(len(ancestry_paths))
    freqs = [round(x * len(sample_ids)) for x in freqs]
    
    # account for rounding discrepancy, assign missing sample to largest group
    if sum(freqs) == (len(sample_ids) - 1):
        freqs[0] += 1
    
    assert sum(freqs) == len(sample_ids)
    ancestries = {}
    remainder = list(sample_ids)
    for freq, path in zip(freqs, ancestry_paths):
        samples = numpy.random.choice(remainder, freq, replace=False)
        for prev in ancestries.values():
            # make sure there's no overlap with other groups
            assert len(set(samples) & set(prev)) == 0
        remainder = list(set(remainder) - set(samples))
        write_sample_ids(path, samples)
        ancestries[path] = samples

def write_gencode(path: str, n_genes: int):
    ''' make a mock gencode gene annotations file
    '''
    opener = gzip.open if str(path).endswith('gz') else open
    with opener(path, 'wt') as output:
        output.write('##description: evidence-based annotation of the human genome (GRCh38)\n')
        output.write('##provider: GENCODE\n')
        output.write('##format: gtf\n')
        for x in range(n_genes):
            chrom = f'chr{x + 1}'
            symbol = f'"GENE{x + 1}"'
            enst = f'"ENST0{x+1}"'
            info = f'transcript_id {enst}; gene_name {symbol}; transcript_type "protein_coding"; tag "appris_principal_1";'

            output.write(f'{chrom}\tHAVANA\tgene\t1\t100\t.\t+\t.\tgene_name {symbol};\n')
            output.write(f'{chrom}\tHAVANA\ttranscript\t1\t100\t.\t+\t.\t{info}\n')
            output.write(f'{chrom}\tHAVANA\texon\t1\t100\t.\t+\t.\t{info}\n')
            output.write(f'{chrom}\tHAVANA\tCDS\t1\t100\t.\t+\t0\t{info}\n')
            output.write(f'{chrom}\tHAVANA\tstart_codon\t1\t3\t.\t+\t0\t{info}\n')
            output.write(f'{chrom}\tHAVANA\tstop_codon\t98\t100\t.\t+\t0\t{info}\n')

def main():
    numpy.random.seed(4696)
    args = get_options()

    sample_ids = simulate_sample_ids(args.cohort_size)
    pheno_vals = simulate_phenotype(args.cohort_size)

    create_phenotype_file(args.phenotypes, sample_ids, pheno_vals, args.pheno_name)
    simulate_array_genotypes(args.array_genotypes, sample_ids, args.n_array_variants, pheno_vals)
    n_genes = simulate_exome_genotypes(args.exome_genotypes, sample_ids, args.n_exome_variants, pheno_vals)
    prepare_training_splits(sample_ids, args.training, args.testing)
    make_ancestry_groups(sample_ids, args.ancestry_samples)

    write_gencode(args.gencode, n_genes)

if __name__ == '__main__':
    main()
