''' mimics GWAS results, to create plausible data for removing common variant signal
'''

import argparse
import sqlite3

from bgen import BgenReader
import numpy
from scipy.stats import linregress

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bgen')
    parser.add_argument('--pheno-db')
    parser.add_argument('--trait-name')
    parser.add_argument('--output')
    return parser.parse_args()

def read_phenotype(db_path, trait):
    query = f'''SELECT s.value, p.value
               FROM "sample_id" s
               INNER JOIN "{trait}" p ON s.rowid = p.rowid'''
    with sqlite3.connect(db_path) as conn:
        return dict(conn.execute(query))

def create_database(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS variants (varid text, rsid text,
                chrom text, pos int, minor text, major text, beta real, se real,
                r2 real, p_value real,
                UNIQUE(chrom, pos, minor, major))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS metadata (key text, value text)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS filepaths (date text, bgen_path text, sample_path text text)''')
    conn.execute('''CREATE INDEX IF NOT EXISTS varid_index ON variants(varid)''')
    conn.execute('''CREATE INDEX IF NOT EXISTS chrom_index ON variants(chrom)''')
    conn.execute('''CREATE INDEX IF NOT EXISTS pos_index ON variants(pos)''')
    conn.execute('''CREATE INDEX IF NOT EXISTS pval_index ON variants(p_value)''')

def add_metadata(conn, pheno_path, trait_name, bgen_path):
    query = '''INSERT INTO metadata VALUES (?, ?)'''
    conn.execute(query, ('pheno_path', pheno_path))
    conn.execute(query, ('phenotype', trait_name))
    query = '''INSERT INTO filepaths VALUES (?, ?, ?)'''
    conn.execute(query, ('2023-01-01', bgen_path, None))

def add_variant(conn, varid, rsid, chrom, pos, minor, major, beta, se, r2, p_value):
    query = '''INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    params = (varid, rsid, chrom, pos, minor, major, beta, se, r2, p_value)
    conn.execute(query, params)

def main():
    args = get_options()
    
    pheno = read_phenotype(args.pheno_db, args.trait_name)
    with BgenReader(args.bgen) as bfile, sqlite3.connect(args.output) as conn:
        create_database(conn)
        add_metadata(conn, args.pheno_db, args.trait_name, args.bgen)
        samples = bfile.samples
        samples = [x for x in samples if x in pheno]
        # make phenotype array aligned with genotypes
        pheno = numpy.array([pheno[x] for x in samples]).astype(numpy.float32)
        
        for var in bfile:
            dose = var.minor_allele_dosage
            minor = var.minor_allele
            major = next(iter(set(var.alleles) - set([minor])))
            fit = linregress(dose, pheno)
            add_variant(conn, var.varid, var.rsid, var.chrom, var.pos, minor, major, 
                        fit.slope, fit.stderr, fit.rvalue ** 2, fit.pvalue)

if __name__ == '__main__':
    main()

