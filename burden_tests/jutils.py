import io
import math
import random
import re
import sqlite3
import sys
import datetime
import pickle
import traceback
import warnings
import os
import subprocess

import gc
import pandas as pd
import numpy as np
import tempfile

import gzip
import scipy.stats
import statsmodels.stats.multitest
# from matplotlib import pyplot as plt

import scipy.sparse

from constants import *


def echo(*message, logfile_only=False, sep=' '):

    message_string = "[%d: %s] %s" % (os.getpid(), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sep.join(map(str, message)))

    if not logfile_only:
        print(message_string)

    if open_log.log_file is not None:
        open_log.log_file.write(message_string + '\n')
        open_log.log_file.flush()


def warn(*message, logfile_only=False, sep=' '):
    echo(*(['[WARNING]'] + list(message)), logfile_only=logfile_only, sep=sep)


def open_log(fname, mode='wt'):

    open_log.log_file = open(fname, mode)
    open_log.log_fname = fname
    open_log.time = datetime.datetime.now()

    echo("NEW LOG STARTED:", fname)


open_log.log_file = None
open_log.log_fname = None
open_log.time = None

RUNNING_ON = None

USFC = 'USFC'
USSD = 'USSD'
DNANEXUS = 'DNANEXUS'
UNKNOWN_ENVIRONMENT = 'UNKNOWN_ENVIRONMENT'

script_location = os.path.split(os.path.realpath(__file__))[0]

if script_location.startswith(ROOT_PATH):
    UKB_DATA_PATH = ROOT_PATH + '/ukbiobank/data/'
    UKB_RARE_VARIANTS_PATH = ROOT_PATH + '/pfiziev/rare_variants/data/ukbiobank/'
    ROOT_PFIZIEV_PATH = ROOT_PATH + '/pfiziev/'
    BGEN_DIR = UKB_DATA_PATH + '/array_genotypes/'
    RUNNING_ON = USFC

elif script_location.startswith(ROOT_PATH_SD):
    UKB_DATA_PATH = ROOT_PATH_SD + '/pfiziev/ukbiobank/data/'
    UKB_RARE_VARIANTS_PATH = ROOT_PATH_SD + '/pfiziev/rare_variants/data/ukbiobank/'
    ROOT_PFIZIEV_PATH = ROOT_PATH_SD + '/pfiziev/'
    BGEN_DIR = UKB_DATA_PATH + '/array_genotypes/'
    RUNNING_ON = USSD

elif os.path.exists(script_location + '/running_on') and ''.join(open(script_location + '/running_on', 'rt').readlines()).strip() == 'dnanexus':
    UKB_DATA_PATH = ROOT_DNANEXUS + '/illumina/'
    UKB_RARE_VARIANTS_PATH = None
    ROOT_PFIZIEV_PATH = ROOT_DNANEXUS
    BGEN_DIR = None
    RUNNING_ON = DNANEXUS

else:
    UKB_DATA_PATH = ROOT_PATH + '/ukbiobank/data/'
    UKB_RARE_VARIANTS_PATH = ROOT_PATH + '/pfiziev/rare_variants/data/ukbiobank/'
    ROOT_PFIZIEV_PATH = ROOT_PATH + '/pfiziev/'

    echo('ERROR: Weird script path:', script_location, ', assuming script runs on Foster City cluster')
    BGEN_DIR = UKB_DATA_PATH + '/array_genotypes/'
    RUNNING_ON = UNKNOWN_ENVIRONMENT

def log_max_memory_usage():

    pid = os.getpid()
    status_fname = f'/proc/{pid}/status'

    if os.path.exists(status_fname):
        with open(status_fname) as in_f:
            for l in in_f:
                if 'VmPeak' in l:
                    echo('PEAK MEMORY:', l.strip())


def close_log():

    log_max_memory_usage()

    echo("LOG ENDED:", open_log.log_fname)
    echo("ELAPSED TIME:", datetime.datetime.now() - open_log.time)

    open_log.log_file.close()

    with open(open_log.log_fname, 'rb') as f_in, gzip.open(open_log.log_fname + '.gz', 'wb') as f_out:
        f_out.writelines(f_in)

    os.unlink(open_log.log_fname)

    open_log.log_file = None
    open_log.log_fname = None


def echo_debug(*args, **kwargs):
    if echo_debug.on:
        echo(*args, **kwargs)


echo_debug.on = False


class _GZipFileWriter:
    def __init__(self, fileName):
        # echo('Constructing _GZipFileWriter:', fileName)

        self.f = open(fileName, 'wb')
        self.p = subprocess.Popen(['bgzip'], stdin=subprocess.PIPE, stdout=self.f, universal_newlines=True)
        # echo('Starting gzip child process:', self.p.pid)
        self.write = self.p.stdin.write

    def __enter__(self):
        return self.p.stdin

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.p.stdin.close()
        # echo('Waiting on', self.p.pid)
        self.p.wait()
        self.f.close()


class _GZipFileReader:
    def __init__(self, fileName):
        # echo('Constructing _GZipFileWriter:', fileName)

        # self.f = open(fileName, 'rb')
        # self.p = subprocess.Popen(['bgzip', '-c', '-d', fileName], stdout=subprocess.PIPE, universal_newlines=True)
        self.p = subprocess.Popen(['zcat', fileName], stdout=subprocess.PIPE, universal_newlines=True)
        # echo('Starting gzip child process:', self.p.pid)
        # self.read = self.p.stdin.read

    def __enter__(self):
        return self.p.stdout

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.p.stdout.close()
        # echo('Waiting on', self.p.pid)
        self.p.wait()
        # self.f.close()

USE_PYTHON_GZIP = False
def open_file(fname, mode='r', verbose=False):

    if verbose:
        echo(('Reading:' if 'r' in mode else 'Writing:'), fname)

    if fname.endswith('.gz') or fname.endswith('.bgz'):

        if USE_PYTHON_GZIP:
            return gzip.open(fname, mode=mode)

        if mode in ['w', 'wt'] :
            return _GZipFileWriter(fname)
        elif mode in ['r', 'rt']:
            # return io.BufferedReader(gzip.open(fname, 'rt'))
            return _GZipFileReader(fname)

            # return gzip.open(fname, 'rt')
            # in_f = open(fname, 'rb')
            # p = subprocess.Popen(['gunzip'], stdout=subprocess.PIPE, stdin=in_f)
            # return p.stdout
        else:
            raise Exception('Only w and r are supported modes for gzipped files. fname=' + fname + ', mode=' + mode)

    else:
        return open(fname, mode)


def remove_special_chars(s):
    return re.sub(r'[^0-9a-zA-Z]+', '_', s).strip('_')


def ranksums_ranked(ranked_x, ranked_y):

    """
    Compute the Wilcoxon rank-sum statistic for two samples.

    Adapted from scipy.stats

    """

    n1 = len(ranked_x)
    n2 = len(ranked_y)

    s = np.sum(ranked_x, axis=0)
    expected = n1 * (n1+n2+1) / 2.0

    z = (s - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)

    prob = 2 * scipy.stats.distributions.norm.sf(abs(z))

    return (z, prob)


def ranksums_with_bgr_z(x, y, bgr_z=None):
    """
    Compute the Wilcoxon rank-sum statistic for two samples.

    """
    x, y = map(np.asarray, (x, y))

    n1 = len(x)
    n2 = len(y)

    alldata = np.concatenate((x, y))
    ranked = scipy.stats.rankdata(alldata)

    x = ranked[:n1]
    s = np.sum(x, axis=0)

    expected = n1 * (n1 + n2 + 1) / 2.0

    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    if bgr_z is not None:
        z = bgr_z - z
        prob = 2 * scipy.stats.distributions.norm.sf(abs(z), scale=2)
    else:
        prob = 2 * scipy.stats.distributions.norm.sf(abs(z))

    return (z, prob)


def load_primateai(fname):
    """Load primateAI scores as a dataframe """

    echo('Reading primateAI scores from:', fname)
    primateai = {VCF_CHROM: [],
                 VCF_POS: [],
                 VCF_REF: [],
                 VCF_ALT: [],
                 PRIMATEAI_SCORE: [],
                 PRIMATEAI_UCSC: []}

    with open_file(fname, 'rt') as in_f:
        for l_idx, l in enumerate(in_f):

            if l_idx > 0 and l_idx % 1000000 == 0:
                echo('Lines processed:', l_idx)

            if l.startswith('#'):
                continue

            buf = l.strip().split()

            chrom = buf[0].replace('chr', '')

            pos = int(buf[1])

            ref = buf[2]
            alt = buf[3]

            strand = buf[6]

            if strand == '0':
                alt = REVCOMP[alt]
                ref = REVCOMP[ref]

            score = float(buf[10])

            primateai[VCF_CHROM].append(chrom)
            primateai[VCF_POS].append(pos)
            primateai[VCF_REF].append(ref)
            primateai[VCF_ALT].append(alt)

            primateai[PRIMATEAI_SCORE].append(score)
            primateai[PRIMATEAI_UCSC].append(buf[8])

    echo('Converting to primateAI scores to dataframe')

    res = pd.DataFrame(primateai)

    res[VCF_CHROM] = res[VCF_CHROM].astype(str)

    return res


def spliceai_vcf_to_dataframe(vcf_fname, min_spliceai_score_threshold):
    from pysam import VariantFile

    echo('Reading:', vcf_fname)
    echo('SpliceAI threshold=', min_spliceai_score_threshold)

    spliceai_info = {VCF_CHROM: [],
                     VCF_POS: [],
                     GENE_NAME: [],
                     VCF_REF: [],
                     VCF_ALT: [],
                     SPLICEAI_MAX_SCORE: []}

    with VariantFile(vcf_fname) as vcf_in:

        for rec_idx, rec in enumerate(vcf_in):

            if rec_idx % 1000000 == 0:
                echo(rec_idx, 'variants processed')

            max_spliceai_score = max(rec.info['DS_AG'],
                                     rec.info['DS_AL'],
                                     rec.info['DS_DG'],
                                     rec.info['DS_DL'])

            if max_spliceai_score >= min_spliceai_score_threshold:
                spliceai_info[VCF_CHROM].append(rec.chrom)
                spliceai_info[VCF_POS].append(rec.pos)

                spliceai_info[VCF_REF].append(rec.ref)
                spliceai_info[VCF_ALT].append(rec.alts[0])

                spliceai_info[GENE_NAME].append(rec.info['SYMBOL'])
                spliceai_info[SPLICEAI_MAX_SCORE].append(max_spliceai_score)
    #                 break
    #                 print(spliceai_info)
    #                 print(rec)

    #                 return None

    res = pd.DataFrame(spliceai_info)
    res[VCF_CHROM] = res[VCF_CHROM].astype(str)

    out_fname = vcf_fname.replace('.gz', '').replace('.vcf', '') + '.min_score_' + str(
        min_spliceai_score_threshold) + '.pandas.pickle'

    echo('Storing dataframe in:', out_fname)

    with open(out_fname, 'wb') as outf:
        pickle.dump(res, outf, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def read_gnomad_exomes_spliceai(spliceai_fname=DATA_PATH+'spliceai/gnomad.exomes.r2.1.spliceai.tsv.gz'):

    spliceai_info = {VCF_CHROM: [],
                     VCF_POS: [],
                     VCF_REF: [],
                     VCF_ALT: [],
                     GENE_NAME: [],
                     SPLICEAI_MAX_SCORE: [],
                     SPLICEAI_DS_AG: [],
                     SPLICEAI_DS_AL: [],
                     SPLICEAI_DS_DG: [],
                     SPLICEAI_DS_DL: []
                     }

    echo('Reading:', spliceai_fname)
    with open_file(spliceai_fname, 'rt') as in_f:
        for l in in_f:
            buf = l.strip().split('\t')

            chrom = buf[0]
            pos = int(buf[1])

            ref = buf[2]
            alt = buf[3]

            spliceai_annotations = [anno.split('|') for anno in buf[4].split('=')[1].split(',')]

            for (ALLELE, SYMBOL, DS_AG, DS_AL, DS_DG, DS_DL, DP_AG, DP_AL, DP_DG, DP_DL) in spliceai_annotations:

                if SYMBOL == '.':
                    continue

                DS_AG, DS_AL, DS_DG, DS_DL = map(float, [DS_AG, DS_AL, DS_DG, DS_DL])

                spliceai_info[VCF_CHROM].append(chrom)
                spliceai_info[VCF_POS].append(pos)
                spliceai_info[VCF_REF].append(ref)
                spliceai_info[VCF_ALT].append(alt)

                spliceai_info[GENE_NAME].append(SYMBOL)

                spliceai_info[SPLICEAI_MAX_SCORE].append(max([DS_AG, DS_AL, DS_DG, DS_DL]))

                spliceai_info[SPLICEAI_DS_AG].append(DS_AG)
                spliceai_info[SPLICEAI_DS_AL].append(DS_AL)
                spliceai_info[SPLICEAI_DS_DG].append(DS_DG)
                spliceai_info[SPLICEAI_DS_DL].append(DS_DL)
    #             break

    res = pd.DataFrame(data=spliceai_info)

    numeric_columns = [VCF_POS, SPLICEAI_MAX_SCORE, SPLICEAI_DS_AG, SPLICEAI_DS_AL, SPLICEAI_DS_DG, SPLICEAI_DS_DL]
    for col in spliceai_info:
        if col in numeric_columns:
            res[col] = pd.to_numeric(res[col])
        else:
            res[col] = res[col].astype(str)

    return res


def get_TCGA_samples_with_variant(var_chrom, var_pos, var_ref, var_alt, sparse_data, return_AC=False):
    """ Return all TCGA samples with a given variant """

    var_info = sparse_data[VAR_INFO]
    var_gt = sparse_data[GENOTYPE_SDF]

    tcga_sample_ids = list(var_gt.columns.values)

    var_index = var_info[(var_info[VCF_CHROM] == var_chrom) &
                         (var_info[VCF_POS] == var_pos) &
                         (var_info[VCF_REF] == var_ref) &
                         (var_info[VCF_ALT] == var_alt)][VARIANT_IDX].values

    if len(var_index) == 0:
        return []
    elif len(var_index) > 1:
        echo('ERROR: more than one variant found for:', var_chrom, var_pos, var_ref, var_alt)
        return None
    else:
        var_index = var_index[0]
        #     print (var_index, type(var_index))
        all_allele_counts = list(var_gt.iloc[var_index])
        sample_ids = [tcga_sample_ids[i]
                      for i in range(len(all_allele_counts)) if not np.isnan(all_allele_counts[i])]

        if return_AC:
            allele_counts = [all_allele_counts[i]
                             for i in range(len(all_allele_counts)) if not np.isnan(all_allele_counts[i])]
            return sample_ids, allele_counts
        else:
            return sample_ids


def get_samples_with_variant(vcfdata,
                             sparse_gt=None,
                             **kwargs):


    echo('Filtering variants')
    filtered_variants = filter_variants(vcfdata, **kwargs)

    all_sample_ids = list(vcfdata.sparse_data)

    var_indexes = filtered_variants.info[VARIANT_IDX].values

    echo('n_variants=', len(var_indexes))

    if sparse_gt is None:
        echo('Converting sparse dataframe to sparse lil matrix')
        sparse_gt = vcfdata.sparse_data.to_coo().tolil()

    new_info = filtered_variants.info.copy()

    var_all_samples = []
    var_homozygous = []
    var_heterozygous = []

    echo('Iterating over variants')
    for var_index in var_indexes:
        if var_index % 10000 == 0:
            echo('Variants processed:', var_index)

        cur_var_samples = set()
        cur_var_homozygous = set()
        cur_var_heterozygous = set()

        row_nz = sparse_gt.data[var_index]
        row_nz_col_idx = sparse_gt.rows[var_index]

        non_missing = [(sample_col_idx, el_idx)
                         for el_idx, sample_col_idx in enumerate(row_nz_col_idx)
                            if row_nz[el_idx] in [1, 2]]

        # echo(var_index)
        # echo(non_missing)

        for sample_idx, el_idx in non_missing:
            cur_var_samples.add(all_sample_ids[sample_idx])

            if row_nz[el_idx] == 1:
                cur_var_heterozygous.add(all_sample_ids[sample_idx])

            elif row_nz[el_idx] == 2:
                cur_var_homozygous.add(all_sample_ids[sample_idx])

        var_all_samples.append(','.join(cur_var_samples))
        var_homozygous.append(','.join(cur_var_homozygous))
        var_heterozygous.append(','.join(cur_var_heterozygous))
        # echo(var_all_samples, var_homozygous, var_heterozygous)

    echo('Adding new columns')

    new_info[ALL_SAMPLES] = var_all_samples
    new_info[HOMOZYGOTES] = var_homozygous
    new_info[HETEROZYGOTES] = var_heterozygous
    echo('Done')

    return VcfData(info=new_info, annotation=filtered_variants.annotation, sparse_data=filtered_variants.sparse_data)


def get_samples_for_variants(vcfdata, homozygotes_only=False, heterozygotes_only=False, include_variant_info=False):
    tag = 'all_samples'
    if homozygotes_only:
        tag = 'homozygotes'
    elif heterozygotes_only:
        tag = 'heterozygotes'

    if include_variant_info:
        var_info = {SAMPLE_ID: []}

        for row_i, row in vcfdata.info.iterrows():
            for sid in row[tag].split(','):

                var_info[SAMPLE_ID].append(sid)

                for t in row.keys():
                    if t not in var_info:
                        var_info[t] = []

                    var_info[t].append(row[t])

        return pd.DataFrame(var_info)
    else:
        return sorted(set(sid for sids in vcfdata.info[tag] for sid in sids.split(',')))


def subset_tcga_variants(sparse_data, sample_ids, verbose=True):
    """ Return all variants found in a set of sample IDs """

    var_info = sparse_data[VAR_INFO]
    var_gt = sparse_data[GENOTYPE_SDF]

    common_ids = sorted(set(var_gt.columns.values) & set(sample_ids))

    if verbose and len(common_ids) != len(sample_ids):
        echo('WARNING:', len(sample_ids) - len(common_ids), 'sample IDs could not be found')

    if verbose:
        echo('Subsetting', len(common_ids), 'sample IDs')

    sample_ids = common_ids

    sm = var_gt[sample_ids].to_coo().tocsr()
    rows_to_keep = (sm.getnnz(1) > 0)

    sm_nzrows = sm[rows_to_keep]
    subset_var_gt = pd.SparseDataFrame(sm_nzrows.tocoo(), columns=sample_ids)

    subset_var_info = pd.merge(var_info, pd.DataFrame({VARIANT_IDX:
                                                       [r_idx
                                                        for r_idx, r_to_keep in enumerate(rows_to_keep)
                                                        if r_to_keep]})).copy()

    subset_var_info[VARIANT_IDX] = list(range(len(subset_var_info)))
    subset_var_info = subset_var_info.set_index(VARIANT_IDX, drop=False)

    return subset_var_info, subset_var_gt


def subset_variants(info=None, anno=None, sparse_data=None, sample_ids=None, sample_ids_to_exculde=None, verbose=False):
    """ Return all variants found in a set of sample IDs """

    common_ids = set(list(sparse_data))

    if sample_ids is not None:
        common_ids = common_ids & set(sample_ids)

    if sample_ids_to_exculde is not None:
        common_ids = common_ids - set(sample_ids_to_exculde)

    common_ids = sorted(common_ids)

    # if verbose and len(common_ids) != len(sample_ids):
    #     echo('WARNING:', len(sample_ids) - len(common_ids), 'sample IDs could not be found')

    if verbose:
        echo('Subsetting', len(common_ids), 'sample IDs')

    sm = sparse_data[common_ids].to_coo()

    if verbose:
        echo('Creating SparseDataFrame')
    subset_sparse_data = pd.SparseDataFrame(sm, columns=common_ids)

    if verbose:
        echo('Converting to csr matrix')

    sm = sm.tocsr()
    if verbose:
        echo('Summing allele counts and missing genotypes')
    allele_counts_with_missing = np.squeeze(np.asarray(sm.sum(axis=1).transpose()))

    if verbose:
        echo('Filtering missing genotypes')
    sm = sm.multiply(sm == MISSING_GENOTYPE)

    if verbose:
        echo('Summing up missing genotypes')
    missing_samples = np.squeeze(np.asarray(sm.sum(axis=1).transpose()))

    if verbose:
        echo('Computing allele counts')
    allele_counts = allele_counts_with_missing - missing_samples

    missing_samples = missing_samples / MISSING_GENOTYPE

    allele_numbers = np.array([2 * len(common_ids)] * len(allele_counts)) - 2 * missing_samples

    new_var_stats = pd.DataFrame({VARIANT_IDX: list(range(len(sparse_data)))})

    new_var_stats[VCF_AC] = allele_counts.astype(int)
    new_var_stats[VCF_AN] = allele_numbers.astype(int)
    new_var_stats[VCF_AF] = new_var_stats[VCF_AC] / new_var_stats[VCF_AN]


    # sm = sparse_data[common_ids].to_coo().tocsr()
    #
    # subset_sparse_data = pd.SparseDataFrame(sm.tocoo(), columns=common_ids)
    #
    # allele_counts = np.squeeze(np.asarray(sm.sum(axis=1).transpose()))
    #
    # new_var_stats = pd.DataFrame({VARIANT_IDX: list(range(len(sparse_data)))})
    #
    # new_var_stats[VCF_AC] = allele_counts.astype(int)
    # new_var_stats[VCF_AN] = 2 * len(common_ids)
    # new_var_stats[VCF_AF] = new_var_stats[VCF_AC] / new_var_stats[VCF_AN]

    subset_info = pd.merge(info, new_var_stats, on=VARIANT_IDX, suffixes=['_OLD', ''])
    subset_info = subset_info[[c for c in list(subset_info) if not c.endswith('_OLD')]]

    subset_info = subset_info[subset_info[VCF_AC] > 0].copy()

    subset_anno = pd.merge(anno, subset_info, on=VARIANT_IDX)[list(anno)].copy()

    return subset_info, subset_anno, subset_sparse_data

TCGA_SAMPLE_ID = 'aliquot_submitter_id'


class TCGAmeta:

    def __getitem__(self, key):
        return self.tcga_metadata[key]

    def __init__(self):
        echo('new2')
        self.tcga_metadata = {}

        tcga_metadata_fnames = ['aliquot.tsv', 'analyte.tsv', 'clinical.tsv', 'exposure.tsv', 'portion.tsv',
                                'sample.tsv', 'slide.tsv']

        for fname in tcga_metadata_fnames:

            echo('Reading:', fname)

            self.tcga_metadata[fname.replace('.tsv', '')] = pd.read_csv(TCGA_METAINFO_PATH + '/' + fname,
                                                                        sep='\t',
                                                                        header=0,
                                                                        na_values=['--'])

        self.tcga_metadata['clinical']['age_at_diagnosis_years'] = \
            self.tcga_metadata['clinical']['age_at_diagnosis'] / 365

    def get_sample_ids_for_project(self, project_id):
        return list(
                self.tcga_metadata['aliquot'][self.tcga_metadata['aliquot']['project_id'] == project_id][TCGA_SAMPLE_ID])

    def get_all_projects(self):
        return sorted(set(self.tcga_metadata['aliquot']['project_id']))

    def get_sample_info(self, sample_ids=None):

        tcga_meta = self.tcga_metadata

        d = pd.merge(pd.merge(tcga_meta['aliquot'],
                              tcga_meta['clinical'], on='case_id', suffixes=['', '_clinical']),
                     tcga_meta['exposure'],
                     on='case_id',
                     how='left',
                     suffixes=['', '_exposure'])

        cols_to_return = [c for c in list(d) if not c.endswith('_clinical') and not c.endswith('_exposure')]

        if sample_ids is not None:
            if type(sample_ids) is not list:
                sample_ids = [sample_ids]
            return d[d[TCGA_SAMPLE_ID].isin(sample_ids)][cols_to_return]
        else:
            return d[cols_to_return]


class VcfData:

    def __init__(self,
                 fname=None,
                 annotation=None,
                 info=None,
                 n_samples=None,
                 return_annotation=True,
                 make_copy=False,
                 sparse_data=None,
                 vcf_fields=None,
                 strip_chrom_prefix=True,
                 vcfdata=None,
                 variant_types=None,
                 n_variants=None,
                 input_is_nirvana_json=False,
                 canonical_transcripts_only=True
                 ):

        """ Input is a VCF file annotated with Nirvana and, optionally, added coverage information for each variant.
            Output: 1) variant_info dataframe that has basic variant information (chrom, pos, ref, alt etc)
                    2) annotation dataframe that has variant annotation from Nirvana

        """

        if vcfdata is not None:

            self.fname = vcfdata.fname
            self.annotation = vcfdata.annotation
            self.info = vcfdata.info
            self.sparse_data = vcfdata.sparse_data

            return

        self.fname = fname
        self.sparse_data = sparse_data

        if self.fname is None:
            if make_copy:
                self.annotation = annotation.copy()
                self.info = info.copy()
            else:
                self.annotation = annotation
                self.info = info
            return

        variant_info = {VARIANT_IDX: [],
                        VCF_CHROM: [],
                        VCF_RSID: [],
                        VCF_FILTER: [],
                        VCF_POS: [],
                        VCF_REF: [],
                        VCF_ALT: []
                        }

        if vcf_fields is None:
            vcf_fields = []

        for vcf_field in vcf_fields:
            variant_info[vcf_field] = []

        if return_annotation:
            annotation = {VARIANT_IDX: [],
                          GENE_NAME: [],
                          TRANSCRIPT_ID: [],
                          CONSEQUENCE: []}

            if input_is_nirvana_json:
                annotation[BIOTYPE] = []

        echo('Reading:', self.fname)
        if n_samples is not None:
            echo('Allele frequencies will be computed based on n_samples=', n_samples)

        if variant_types is not None:
            echo('Loading information only about variant types:', variant_types)

        if input_is_nirvana_json:
            variant_info, annotation = self.parse_json(variant_info,
                                                       annotation,
                                                       variant_types,
                                                       vcf_fields,
                                                       n_variants,
                                                       return_annotation,
                                                       strip_chrom_prefix,
                                                       n_samples,
                                                       canonical_transcripts_only=canonical_transcripts_only)

        else:
            variant_info, annotation = self.parse_vcf(variant_info,
                                                      annotation,
                                                      variant_types,
                                                      vcf_fields,
                                                      n_variants,
                                                      return_annotation,
                                                      strip_chrom_prefix,
                                                      n_samples)


        echo('Converting input to dataframes')
        # for k in variant_info:
        #     echo(k, len(variant_info[k]))

        self.info = pd.DataFrame(variant_info)

        if return_annotation:
            self.annotation = pd.DataFrame(annotation)


    def parse_json(self, variant_info, annotation, variant_types, vcf_fields, n_variants, return_annotation, strip_chrom_prefix, n_samples,
                   canonical_transcripts_only=True):

        import orjson

        echo('Parsing input from Nirvana JSON with gnomAD info')
        alt_missing = 0
        annotation[PRIMATEAI_SCORE] = []
        annotation[SPLICEAI_MAX_SCORE] = []

        annotation[GNOMAD_MEDIAN_COVERAGE] = []
        annotation[GNOMAD_AF] = []
        annotation[GNOMAD_AC] = []
        annotation[GNOMAD_AN] = []

        annotation[TOPMED_AF] = []

        annotation[GNOMAD_POPMAX] = []
        annotation[GNOMAD_POPMAX_AF] = []
        annotation[GNOMAD_POPMAX_AC] = []
        annotation[GNOMAD_POPMAX_AN] = []
        annotation[IS_CANONICAL] = []

        variant_info[VCF_QUAL] = []

        with open_file(self.fname, 'rt') as in_f:
            header = in_f.readline()

            var_idx = 0

            for line in in_f:
                if line.startswith('{"chromosome":'):

                    if var_idx % 1000000 == 0:
                        echo('Variants processed so far:', var_idx)

                    if n_variants is not None and var_idx >= n_variants:
                        echo('Finished reading', n_variants)
                        break

                    rec = orjson.loads(line.rstrip(',\n'))

                    ref_allele = rec['refAllele']
                    if 'altAlleles' not in rec:
                        alt_missing += 1
                        continue

                    alt_alleles = rec['altAlleles']
                    alt_alleles = [a for a in alt_alleles if a != '*']
                    quality = float(rec.get('quality', -1))

                    if len(rec['variants']) != len(alt_alleles):
                        echo('ERROR:', line)
                        raise Exception("alt alleles don't match variants in json:" + str(alt_alleles) +
                                        ', ' + str(len(rec['variants'])) +
                                        ', ' + str(rec['variants']))

                    for (variant, alt_allele) in zip(rec['variants'], alt_alleles):

                        chrom = variant['chromosome']

                        if strip_chrom_prefix:
                            chrom = chrom.replace('chr', '')

                        variant_info[VARIANT_IDX].append(var_idx)
                        variant_info[VCF_CHROM].append(chrom)
                        variant_info[VCF_QUAL].append(quality)

                        if 'dbsnp' in variant:
                            if type(variant['dbsnp']) is dict:
                                rsid = ','.join(variant['dbsnp']['ids'])
                            else:
                                rsid = variant['dbsnp'][0] if len(variant['dbsnp']) > 0 else '.'
                        else:
                            rsid = '.'

                        variant_info[VCF_RSID].append(rsid)

                        variant_info[VCF_POS].append(rec['position'])
                        variant_info[VCF_FILTER].append(';'.join([]))

                        for vcf_field in vcf_fields:
                            vcf_value = variant.get(vcf_field, None)

                            if type(vcf_value) in [tuple, list]:
                                vcf_value = ';'.join(map(str, vcf_value))

                            variant_info[vcf_field].append(vcf_value)

                        variant_info[VCF_REF].append(ref_allele)
                        variant_info[VCF_ALT].append(alt_allele)

                        # store nirvana annotations
                        seen = set()

                        for cur_anno in variant.get('transcripts', []):
                            is_canonical = ('isCanonical' in cur_anno and cur_anno['isCanonical'])

                            if canonical_transcripts_only and not is_canonical:
                                continue

                            gene_name = cur_anno['hgnc']
                            bioType = cur_anno['bioType']
                            transcript_id = cur_anno['transcript']

                            for consequence in cur_anno['consequence']:
                                # remove duplicate annotations from Ensemble and Refseq
                                key = (gene_name, consequence)

                                if canonical_transcripts_only and key in seen:
                                    continue

                                seen.add(key)

                                annotation[VARIANT_IDX].append(var_idx)
                                annotation[GENE_NAME].append(gene_name)
                                annotation[CONSEQUENCE].append(consequence)
                                annotation[BIOTYPE].append(bioType)
                                annotation[TRANSCRIPT_ID].append(transcript_id)
                                annotation[IS_CANONICAL].append(int(is_canonical))

                                no_pAI_found = True
                                if 'primateAI' in variant:
                                    for pAI in variant['primateAI']:
                                        if pAI['hgnc'] == gene_name:
                                            annotation[PRIMATEAI_SCORE].append(float(pAI['scorePercentile']))
                                            no_pAI_found = False
                                            break

                                if no_pAI_found:
                                    annotation[PRIMATEAI_SCORE].append(None)

                                no_spliceAI_found = True
                                if 'spliceAI' in variant:
                                    for sAI in variant['spliceAI']:
                                        if sAI['hgnc'] == gene_name:

                                            splice_AI_scores = [float(sAI[k]) for k in sAI if k.endswith('Score')]
                                            max_spliceAI_score = max(splice_AI_scores) if len(splice_AI_scores) > 0 else None

                                            annotation[SPLICEAI_MAX_SCORE].append(max_spliceAI_score)
                                            no_spliceAI_found = False

                                            break

                                if no_spliceAI_found:
                                    annotation[SPLICEAI_MAX_SCORE].append(None)

                                if 'gnomad' in variant:
                                    gnomad_info = variant['gnomad']

                                    annotation[GNOMAD_MEDIAN_COVERAGE].append(float(gnomad_info['coverage']))

                                    annotation[GNOMAD_AF].append(float(gnomad_info['allAf']))
                                    annotation[GNOMAD_AC].append(int(gnomad_info['allAc']))
                                    annotation[GNOMAD_AN].append(int(gnomad_info['allAn']))

                                    gnomad_pops = [c.replace('An', '') for c in gnomad_info if c.endswith('An')]
                                    gnomad_pops = [c for c in gnomad_pops if c not in ['all', 'male', 'female']]
                                    popmax = max(gnomad_pops, key=lambda p: float(gnomad_info[p + 'Af']))

                                    annotation[GNOMAD_POPMAX].append(popmax)
                                    annotation[GNOMAD_POPMAX_AF].append(float(gnomad_info[popmax + 'Af']))
                                    annotation[GNOMAD_POPMAX_AC].append(int(gnomad_info[popmax + 'Ac']))
                                    annotation[GNOMAD_POPMAX_AN].append(int(gnomad_info[popmax + 'An']))

                                else:
                                    for k in [GNOMAD_AF, GNOMAD_AC, GNOMAD_AN,
                                              GNOMAD_POPMAX_AF, GNOMAD_POPMAX_AC, GNOMAD_POPMAX_AN,
                                              GNOMAD_MEDIAN_COVERAGE]:

                                        annotation[k].append(0)

                                    annotation[GNOMAD_POPMAX].append(None)

                                if 'topmed' in variant:
                                    topmed_info = variant['topmed']
                                    annotation[TOPMED_AF].append(float(topmed_info['allAf']))
                                else:
                                    annotation[TOPMED_AF].append(0)

                        var_idx += 1

        echo('Missing ALT alleles:', alt_missing)
        return variant_info, annotation


    def parse_vcf(self, variant_info, annotation, variant_types, vcf_fields, n_variants, return_annotation, strip_chrom_prefix, n_samples):
        from pysam import VariantFile

        cov_hist_bins = 'over_1|over_5|over_10|over_15|over_20|over_25|over_30|over_50|over_100'.split('|')

        with VariantFile(self.fname) as vcf_in:

            var_idx = 0

            for rec in vcf_in:

                if var_idx > 0 and var_idx % 1000000 == 0:
                    echo('Variants processed:', var_idx)

                if n_variants is not None and var_idx >= n_variants:
                    echo('Finished reading', n_variants)
                    break

                if rec.alts is None:
                    echo('[WARNING] Skipping because no ALTS defined:', rec)
                    continue

                n_alts = len(rec.alts)
                annotations = [[] for _ in range(n_alts)]

                if return_annotation:
                    if VCF_CSQT in rec.info:
                        for csqt in rec.info[VCF_CSQT]:
                            fields = csqt.split('|')

                            allele_no = int(fields[0]) - 1

                            gene_name = fields[1]

                            transcript_id = fields[2].split('.')[0]

                            var_consequences = fields[3].split('&')

                            for consequence in var_consequences:
                                annotations[allele_no].append((gene_name, transcript_id, consequence))

                if variant_types is not None:
                    annotations = [[a for a in allele_anno if a[2] in variant_types]
                                   for allele_anno in annotations]

                    n_annotations = sum(len(allele_anno) for allele_anno in annotations)

                    if n_annotations == 0:
                        continue

                for alt_idx in range(n_alts):

                    if VCF_AC in rec.info:
                        alt_AC = rec.info[VCF_AC][alt_idx]

                        if alt_AC == 0:
                            continue

                    variant_info[VARIANT_IDX].append(var_idx)
                    chrom = rec.chrom

                    if strip_chrom_prefix:
                        chrom = chrom.replace('chr', '')

                    variant_info[VCF_CHROM].append(chrom)
                    variant_info[VCF_RSID].append(rec.id)

                    variant_info[VCF_POS].append(rec.pos)
                    variant_info[VCF_FILTER].append(';'.join(rec.filter))

                    for vcf_field in vcf_fields:
                        vcf_value = rec.info.get(vcf_field, None)

                        if type(vcf_value) in [tuple, list]:
                            vcf_value = vcf_value[alt_idx]

                        variant_info[vcf_field].append(vcf_value)

                    variant_info[VCF_REF].append(rec.ref)
                    variant_info[VCF_ALT].append(rec.alts[alt_idx])

                    if VCF_AC in rec.info:

                        if VCF_AC not in variant_info:
                            variant_info[VCF_AC] = []
                            variant_info[VCF_AN] = []
                            variant_info[VCF_AF] = []

                        variant_info[VCF_AC].append(rec.info[VCF_AC][alt_idx])

                        if n_samples is None:
                            variant_info[VCF_AN].append(rec.info[VCF_AN][alt_idx]
                                                        if type(rec.info[VCF_AN]) is tuple else rec.info[VCF_AN])

                            variant_info[VCF_AF].append(rec.info[VCF_AF][alt_idx])

                        else:

                            variant_info[VCF_AN].append(2 * n_samples)
                            variant_info[VCF_AF].append(rec.info[VCF_AC][alt_idx] / (2 * n_samples))

                    if MEAN_COVERAGE in rec.info:

                        # initialize variant_info with the coverage tags
                        if MEAN_COVERAGE not in variant_info:
                            variant_info[MEAN_COVERAGE] = []
                            variant_info[MEDIAN_COVERAGE] = []
                            for bin_label in cov_hist_bins:
                                variant_info[COVERAGE_HISTOGRAM + '|' + bin_label] = []

                        variant_info[MEAN_COVERAGE].append(rec.info[MEAN_COVERAGE])
                        variant_info[MEDIAN_COVERAGE].append(rec.info[MEDIAN_COVERAGE])

                        cov_hist = map(float, rec.info[COVERAGE_HISTOGRAM].split('|'))

                        for bin_label, bin_value in zip(cov_hist_bins, cov_hist):
                            variant_info[COVERAGE_HISTOGRAM + '|' + bin_label].append(bin_value)

                    # store nirvana annotations
                    for gene_name, transcript_id, consequence in annotations[alt_idx]:
                        annotation[VARIANT_IDX].append(var_idx)
                        annotation[GENE_NAME].append(gene_name)
                        annotation[TRANSCRIPT_ID].append(transcript_id)
                        annotation[CONSEQUENCE].append(consequence)

                    var_idx += 1

        return variant_info, annotation

    def full(self, columns=None, columns_to_exclude=None):
        result = pd.merge(self.info, self.annotation, on=VARIANT_IDX)

        if columns is not None:
            result = result[columns]

        if columns_to_exclude is not None:
            result = result[[c for c in list(result) if c not in columns_to_exclude]]

        return result


def combine_datasets(vcfdata_1, vcfdata_2):

    indicator = pd.merge(vcfdata_1.info[[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT]],
                         vcfdata_2.info[[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT]],
                         on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                         how='outer', indicator=True).copy()

    combined_info = pd.merge(vcfdata_1.info,
                             vcfdata_2.info,
                             on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                             suffixes=['_d1', '_d2'],
                             how='outer')

    combined_info = pd.merge(combined_info, indicator)

    for k in [VCF_AC, VCF_AN]:
        for suf in ['1', '2']:
            key = k + '_d' + suf
            if k == VCF_AN:
                value = np.median(combined_info[key].dropna())
                echo('Replacing missing values for:', key, 'with', value)
            else:
                value = 0

            combined_info[key] = combined_info[key].fillna(value)

    combined_info[VCF_AC] = (combined_info[VCF_AC + '_d1'] + combined_info[VCF_AC + '_d2']).astype(int)
    combined_info[VCF_AN] = (combined_info[VCF_AN + '_d1'] + combined_info[VCF_AN + '_d2']).astype(int)
    combined_info[VCF_AF] = combined_info[VCF_AC] / combined_info[VCF_AN]
    combined_info[VARIANT_IDX] = list(range(1, len(combined_info) + 1))

    columns_to_adjust = [c.replace('_d1', '') for c in list(combined_info) if c.endswith('_d1') and
                         not (c.startswith(VCF_AC) or
                              c.startswith(VCF_AN) or
                              c.startswith(VCF_AF) or
                              c.startswith('index') or
                              any(tag in c for tag in ['all_samples', 'homozygotes', 'heterozygotes']))]

    keep_left = (combined_info['_merge'] == 'left_only')

    for c in columns_to_adjust:
        echo('Adjusting column:', c)
        combined_info[c] = np.where(keep_left, combined_info[c + '_d1'], combined_info[c + '_d2'])

    combined_info = combined_info[[col for col in list(combined_info)
                                   if not col.endswith('_d1') and
                                      not col.endswith('_d2')]].copy()

    #     combined_info[]

    anno_cols = sorted(set(list(vcfdata_1.annotation)) & set(list(vcfdata_2.annotation)) - set([VARIANT_IDX]))

    anno1 = pd.merge(vcfdata_1.info, vcfdata_1.annotation, on=VARIANT_IDX)[
        [VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT] + anno_cols]
    anno2 = pd.merge(vcfdata_2.info, vcfdata_2.annotation, on=VARIANT_IDX)[
        [VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT] + anno_cols]

    combined_annotation = pd.merge(combined_info,
                                   pd.concat([anno1,
                                              anno2]),
                                   on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])[
        [VARIANT_IDX] + anno_cols].drop_duplicates()

    return VcfData(info=combined_info, annotation=combined_annotation)


# display(ambryshare_BRCA_vs_gnomad_non_cancer_by_ethnicity_gender_joint['female_nfe'][0].info.head())
# display(ambryshare_BRCA_vs_gnomad_non_cancer_by_ethnicity_gender_joint['female_nfe'][1].info.head())
# %%capture ambryshare_BRCA_stats_out1
# pydevd.stoptrace()
# pydevd.settrace('10.112.113.22', port=21213, stdoutToServer=True, stderrToServer=True)

def compute_mean_exon_coverage(dataset, gencode_exons, consequence=None, coverage=None):

    dataset = filter_variants(dataset, consequence=consequence)
    if coverage is None:
        cov_column = 'COVERAGE_HISTOGRAM|over_20'
    else:
        cov_column = 'COVERAGE_HISTOGRAM|over_%d' % coverage

    var_info = pd.merge(dataset.info,
                        dataset.annotation,
                        on=VARIANT_IDX)[[VCF_POS, GENE_NAME, VARIANT_IDX, cov_column, VCF_AC]]

    per_exon = pd.merge(var_info,
                        gencode_exons,
                        on=GENE_NAME)

    per_exon = per_exon[(per_exon[VCF_POS] >= per_exon['start'] - 5) & (per_exon[VCF_POS] < per_exon['end'] + 5)]
    per_exon = per_exon.drop_duplicates([VARIANT_IDX, 'exon_id'])

    exon_coverage = per_exon.groupby('exon_id').agg({cov_column: 'mean', VCF_AC: ['sum', 'size']})

    return pd.merge(gencode_exons,
                    exon_coverage,
                    on='exon_id',
                    how='outer').fillna(0)


def joint_exon_coverage(cov1,
                        cov2,
                        cov1_label='|AmbryShare',
                        cov2_label='|gnomAD',
                        gene=None,
                        coverage=None):
    if coverage is None:
        cov_column = 'COVERAGE_HISTOGRAM|over_20'
    else:
        cov_column = 'COVERAGE_HISTOGRAM|over_%d' % coverage

    res = pd.merge(cov1,
                   cov2,
                   on=[col for col in list(cov1) if cov_column not in col and VCF_AC not in col],
                   suffixes=[cov1_label, cov2_label])

    if gene is not None:
        res = res[res[GENE_NAME] == gene]

    for label_to_test in ["('" + str(VCF_AC) + "', 'size')", f"(" + str(VCF_AC) + ", 'sum')"]:

        cnt1_label = label_to_test + cov1_label
        cnt2_label = label_to_test + cov2_label

        #         total_1 = np.sum(res[cnt1_label])
        #         total_2 = np.sum(res[cnt2_label])

        total_1 = np.sum(cov1[eval(label_to_test)])
        total_2 = np.sum(cov2[eval(label_to_test)])

        echo(total_1, total_2)

        def chi2_test(row):
            if row[cnt1_label] > 0 or row[cnt2_label] > 0:
                chi2, pvalue, dof, expected = scipy.stats.chi2_contingency(
                    [[row[cnt1_label], total_1 - row[cnt1_label]],
                     [row[cnt2_label], total_2 - row[cnt2_label]]])
            else:
                pvalue = np.nan

            return pvalue

        res[label_to_test + ' chi2_pvalue'] = res.apply(chi2_test, axis=1)

        res = res[~res[label_to_test + ' chi2_pvalue'].isnull()].copy()

        res[label_to_test + ' chi2_FDR'] = \
        statsmodels.stats.multitest.multipletests(res[label_to_test + ' chi2_pvalue'], method='fdr_bh')[1]

        def _odds_ratio(row):
            return odds_ratio(row[cnt1_label], total_1, row[cnt2_label], total_2)

        res[label_to_test + ' OR'] = res.apply(_odds_ratio, axis=1)

    return res[[c for c in list(res) if c not in ['.', 'transcript_id']]].sort_values(
        f"('{VCF_AC}', 'size') chi2_pvalue")


def get_icd10_phenotypes(diagnoses, icd10_diagnosis_col_name='41202'):

    n_samples = len(diagnoses)
    # n_samples = 1000
    echo('n_samples=', n_samples)

    icd10_col_names = [col for col in list(diagnoses) if col.startswith(icd10_diagnosis_col_name + '-')]

    echo('icd10:', len(icd10_col_names), icd10_col_names[:10])

    all_icd10_codes = sorted(set([v for col in icd10_col_names
                                  for v in diagnoses[col]
                                  if type(v) is not float or not math.isnan(v)]))

    echo('icd codes:', len(all_icd10_codes), all_icd10_codes[:10])

    icd10_phenotypes = dict((icd10_diagnosis_col_name + '_' + icd10_code,
                             [False] * n_samples) for icd10_code in all_icd10_codes)

    icd10_phenotypes[SAMPLE_ID] = []

    for row_idx in range(n_samples):
        userId = diagnoses.iloc[row_idx]['eid']

        if row_idx % 1000 == 0:
            echo(row_idx, 'samples processed')

        icd10_phenotypes[SAMPLE_ID].append(userId)

        for col in icd10_col_names:
            current_icd10_code = diagnoses.iloc[row_idx][col]

            # if row_idx < 3:
            #     echo(userId, col, current_icd10_code)

            if type(current_icd10_code) is not float:
                icd10_phenotypes[icd10_diagnosis_col_name + '_' + current_icd10_code][row_idx] = True

    return pd.DataFrame(icd10_phenotypes).astype({SAMPLE_ID: str})


def get_ukb_field_name(ukb_field_names, field_id):
    return ukb_field_names[ukb_field_names['FieldID'] == int(field_id)].iloc[0]['Field']


def load_genotypes(data_path, gnomad_coverage, sample_ids_to_exclude=None, load_full_annotation=False):

    DEPTH = 'depth'
    # loading metadata
    metadata = pd.read_csv(data_path + 'metadata.txt',
                           sep='\t',

                           names=[VCF_CHROM,
                                  VCF_POS,
                                  VCF_REF,
                                  VCF_ALT,
                                  DEPTH],

                           dtype={VCF_CHROM: str,
                                  VCF_POS: int,
                                  VCF_REF: str,
                                  VCF_ALT: str,
                                  DEPTH: int})

    metadata[VARIANT_IDX] = list(range(len(metadata)))

    # loading variant annotations
    if load_full_annotation:
        echo('Loading full annotation from', data_path + 'annotations_full.txt')

        annotations = pd.read_csv(data_path + 'annotations_full.txt',
                                  sep='\t',
                                  header=0,
                                  dtype={'chrom': str,
                                         VCF_POS: int,
                                         'ref': str,
                                         'alt': str,
                                         'Consequence': str,
                                         'SYMBOL': str}
                                  ).rename(
            columns={'chrom': VCF_CHROM,
                     'ref': VCF_REF,
                     'alt': VCF_ALT,
                     'SYMBOL': GENE_NAME,
                     'Consequence': VCF_CONSEQUENCE})

        annotations['CANONICAL'] = (annotations['CANONICAL'] == 'YES')

        annotations = pd.merge(annotations,
                               metadata,
                               on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])[[VARIANT_IDX] +
                                                                          [c for c in list(annotations)
                                                                           if c not in [VCF_CHROM,
                                                                                        VCF_POS,
                                                                                        VCF_REF,
                                                                                        VCF_ALT]]]

        echo('Total annotations:', len(annotations))

        _t = pd.DataFrame(annotations[VCF_CONSEQUENCE].str.split('&').tolist(),
                          index=annotations.index).stack()

        echo('Total annotations after splitting:', len(_t))

        annotations = pd.merge(annotations,
                               _t.reset_index().rename(columns={'level_0': 'row_no', 0: VCF_CONSEQUENCE}),
                               left_index=True,
                               right_on='row_no',
                               suffixes=['_ORIGINAL', ''],
                               validate='one_to_many')[list(annotations) + ['consequence_ORIGINAL']]

        echo('Total annotations after merging:', len(annotations))

    else:

        annotations = pd.read_csv(data_path + 'annotations.txt',
                                  sep='\t',
                                  names=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, VCF_CONSEQUENCE, GENE_NAME],
                                  dtype={VCF_CHROM: str,
                                         VCF_POS: int,
                                         VCF_REF: str,
                                         VCF_ALT: str,
                                         VCF_CONSEQUENCE: str,
                                         GENE_NAME: str})

        # keep only variant index, consequence and gene name in annotations
        annotations = pd.merge(annotations,
                               metadata,
                               on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])[[VARIANT_IDX, VCF_CONSEQUENCE, GENE_NAME]]

    # load all sample ids
    all_sample_ids = pd.read_csv(data_path + 'sample_ids.txt',
                                 sep='\t',
                                 names=[SAMPLE_ID],
                                 dtype={SAMPLE_ID: str})

    echo('loading sparse matrix')
    gt_sparse_matrix = scipy.sparse.load_npz(data_path + 'genotypes.npz')
    gt_sdf = pd.SparseDataFrame(gt_sparse_matrix, columns=all_sample_ids[SAMPLE_ID])

    # remove sample outliers based on number of variants
    # gt_sdf = remove_sample_outliers(gt_sdf)

    if sample_ids_to_exclude is not None:
        gt_sdf = gt_sdf[[sample_id for sample_id in list(gt_sdf) if sample_id not in sample_ids_to_exclude]]

    sample_ids = list(gt_sdf)

    def convert_to_vcfdata(sample_ids, sparse_data, metadata, annotations, gnomad_coverage):
        # create a VcfData object for the variant data

        sm = sparse_data.to_coo().tocsr()

        echo('Summing allele counts and missing genotypes')
        allele_counts_with_missing = np.squeeze(np.asarray(sm.sum(axis=1).transpose()))

        echo('Filtering missing genotypes')
        sm = sm.multiply(sm == MISSING_GENOTYPE)

        echo('Summing up missing genotypes')
        missing_samples = np.squeeze(np.asarray(sm.sum(axis=1).transpose()))

        echo('Computing allele counts')
        allele_counts = allele_counts_with_missing - missing_samples

        missing_samples = missing_samples / MISSING_GENOTYPE

        allele_numbers = np.array([2 * len(sample_ids)] * len(allele_counts)) - 2 * missing_samples

        variant_info = metadata.copy()

        variant_info[VCF_AC] = allele_counts.astype(int)
        variant_info[VCF_AN] = allele_numbers.astype(int)
        variant_info[VCF_AF] = variant_info[VCF_AC] / variant_info[VCF_AN]

        echo('Total variants before checking gnomad coverage:', len(variant_info))

        variant_info = pd.merge(variant_info,
                                gnomad_coverage,
                                on=[VCF_CHROM, VCF_POS])

        echo('Total variants with gnomad coverage:', len(variant_info))

        variant_info = variant_info[variant_info[VCF_AC] > 0]

        echo('Total variants with non-zero AC:', len(variant_info))

        return VcfData(info=variant_info,
                       annotation=pd.merge(variant_info,
                                           annotations,
                                           on=VARIANT_IDX)[list(annotations)].copy(),
                       sparse_data=sparse_data)

    echo('Converting to VcfData')
    variant_data = convert_to_vcfdata(sample_ids, gt_sdf, metadata, annotations, gnomad_coverage)

    return variant_data



def exon_coverage_qqplot(stats_object, pvalue_label, title_prefix=''):
    pvalues = [p for p in stats_object[pvalue_label] if not np.isnan(p)]

    min_pvalue = min([p for p in pvalues if p > 0])

    observed = sorted([p if p > 0 else (min_pvalue / 10) for p in pvalues])

    expected = [(i + 1) / len(pvalues) for i in range(len(pvalues))]

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    fig.suptitle(title_prefix + ": " + pvalue_label + ' n=' + str(len(stats_object)))

    ax[0].plot(expected, observed, 'r.')

    log_observed = sorted(-np.log10(observed), reverse=True)
    log_expected = -np.log10(expected)

    ax[1].plot(log_expected, log_observed, 'r.')

    # bins = np.histogram(np.hstack((stats_object[dataset_labels[0]],
    #                                stats_object[dataset_labels[1]])), bins=50)[1]

    bins = list(range(25))  # + [1000]

    for ax_idx in [0, 1]:
        axisMax = (max(max(log_expected), max(log_observed)) + 0.5) if ax_idx == 1 else 1

        ax[ax_idx].set_xlim([0, axisMax])
        ax[ax_idx].set_xlabel("Expected P-values " + ("(-log10)" if ax_idx == 1 else ""))

        ax[ax_idx].set_ylim([0, axisMax])
        ax[ax_idx].set_ylabel("Observed P-values " + ("(-log10)" if ax_idx == 1 else ""))

        ax[ax_idx].set_title("Log QQ plot" if ax_idx == 1 else 'QQ plot')
        ax[ax_idx].plot([0, axisMax], [0, axisMax], 'b-')  # blue line

    plt.show()


def filter_dataframe(df, **kwargs):
    """ Return all rows from a dataframes with fields equal to some value """
    res = df
    for key in kwargs:
        if type(kwargs[key]) is list:
            res = res[res[key].isin(kwargs[key])]
        else:
            res = res[res[key] == kwargs[key]]
    return res


def intersect_variants(vcfdata,
                       intervals,
                       offset=0,
                       int_chrom_column=VCF_CHROM,
                       int_start_column='start',
                       int_end_column='end', exclude=False):

    import pybedtools

    variant_info = vcfdata.info[[VCF_CHROM, VCF_POS, VARIANT_IDX]].copy()
    variant_info[VCF_POS + '_end'] = variant_info[VCF_POS]
    variant_info = variant_info[[VCF_CHROM, VCF_POS, VCF_POS + '_end', VARIANT_IDX]]

    tmp_intervals = intervals[[int_chrom_column, int_start_column, int_end_column]].copy()

    tmp_intervals[int_chrom_column] = tmp_intervals[int_chrom_column].apply(lambda s: s.replace('chr', ''))

    tmp_intervals[int_start_column] = tmp_intervals[int_start_column].apply(lambda p: max(0, p - offset))
    tmp_intervals[int_end_column] = tmp_intervals[int_end_column] + offset

    intervals_bed = pybedtools.BedTool.from_dataframe(tmp_intervals).sort()

    variants_bed = pybedtools.BedTool.from_dataframe(variant_info).sort()

    result = variants_bed.intersect(intervals_bed, u=True).to_dataframe().rename(
        columns={'name': VARIANT_IDX}).set_index(VARIANT_IDX)

    if exclude:
        filtered_var_info = pd.merge(vcfdata.info,
                                     result,
                                     on=VARIANT_IDX,
                                     how='outer',
                                     indicator=True)

        filtered_var_info = filtered_var_info[filtered_var_info['_merge'] == 'left_only']

        filtered_var_info = filtered_var_info[list(vcfdata.info)]
    else:
        filtered_var_info = pd.merge(vcfdata.info,
                                     result,
                                     on=VARIANT_IDX,
                                     how='inner')[list(vcfdata.info)]

    filtered_var_anno = pd.merge(filtered_var_info, vcfdata.annotation, on=VARIANT_IDX)[list(vcfdata.annotation)]

    return VcfData(info=filtered_var_info, annotation=filtered_var_anno)


def filter_variants(vcfdata,

                    external_info=None,

                    coverage=None,
                    min_frac=None,
                    coverage_tag_prefix=None,

                    filter=None,

                    max_AF=None,
                    min_AF=None,

                    min_AN=None,
                    max_AC=None,

                    max_original_AC=None,
                    max_original_AC_male=None,

                    max_original_AF=None,
                    min_original_AN=None,

                    max_joint_AC=None,
                    min_joint_AN=None,
                    max_joint_AF=None,

                    min_primateAI_score=None,

                    keep_exclusive_variants=False,
                    variants_to_exclude=None,

                    min_AC=None,

                    cancer_type=None,
                    cancer_sample_ids=None,
                    tcga_sparse_data=None,
                    tcga_meta=None,

                    SNV_only=False,
                    non_SNV_only=False,
                    consequence=None,

                    min_pAI_deleterious_score_quantile=0.5,
                    min_sAI_deleterious_score=0.2,

                    chromosomes_to_exclude=None,
                    gencode_info=None,
                    gene_set=None,
                    gene_name=None,

                    white_listed_exons=None,
                    black_listed_exons=None,
                    exclude_intervals=None,
                    is_canonical=None,

                    sample_ids=None,
                    sample_ids_to_exculde=None,

                    verbose=False,
                    all_samples=None,
                    not_in_all_samples=None,
                    keep_excusive_in_all_samples=False,
                    **kwargs
                    ):

    info = vcfdata.info
    anno = vcfdata.annotation

    if hasattr(vcfdata, 'sparse_data'):
        sparse_data = vcfdata.sparse_data
    else:
        sparse_data = None

    if gene_name is not None:
        gene_set = [gene_name]

    if gene_set is not None:
        anno = anno[anno[GENE_NAME].isin(gene_set)]
        info = pd.merge(info, anno[[VARIANT_IDX]].drop_duplicates(), on=VARIANT_IDX)

    if all_samples is not None:
        all_samples_set = set(all_samples)
        if keep_excusive_in_all_samples:
            to_keep = info.apply(lambda x: len(set(x['all_samples'].split(',')) & all_samples_set) == len(
                set(x['all_samples'].split(','))), axis=1)
        else:
            to_keep = info.apply(lambda x: len(set(x['all_samples'].split(',')) & all_samples_set) > 0, axis=1)

        info = info[to_keep].copy()

        info['all_samples'] = info['all_samples'].apply(lambda x: ','.join(sorted(set(x.split(',')) & all_samples_set)))

        info[VCF_AC] = info['all_samples'].apply(lambda x: x.count(',') + 1)
        info[VCF_AN] = 2 * len(all_samples_set)
        info[VCF_AF] = info[VCF_AC] / info[VCF_AN]

    if not_in_all_samples is not None:
        not_in_all_samples_set = set(not_in_all_samples)
        if keep_excusive_in_all_samples:
            to_keep = info.apply(lambda x: len(set(x['all_samples'].split(',')) & not_in_all_samples_set) == 0, axis=1)
        else:
            to_keep = info.apply(lambda x: len(set(x['all_samples'].split(',')) - not_in_all_samples_set) > 0, axis=1)

        info = info[to_keep].copy()

        info['all_samples'] = info['all_samples'].apply(lambda x: ','.join(sorted(set(x.split(',')) - not_in_all_samples_set)))

        info[VCF_AC] = info['all_samples'].apply(lambda x: x.count(',') + 1)
        info[VCF_AN] = np.maximum(info[VCF_AC], info[VCF_AN] - 2 * len(not_in_all_samples_set))
        info[VCF_AF] = info[VCF_AC] / info[VCF_AN]

    if sample_ids is not None or sample_ids_to_exculde is not None:
        # echo('Total samples:', len(sample_ids))
        info, anno, sparse_data = subset_variants(info=info,
                                                  anno=anno,
                                                  sparse_data=vcfdata.sparse_data,
                                                  sample_ids=sample_ids,
                                                  sample_ids_to_exculde=sample_ids_to_exculde,
                                                  verbose=verbose)

    if white_listed_exons is not None:
        info = intersect_variants(vcfdata, white_listed_exons, offset=5).info
        #
        #
        #
        # var_info = pd.merge(info, anno, on=VARIANT_IDX)
        #
        # to_keep = pd.merge(var_info, white_listed_exons, on=GENE_NAME)
        #
        # to_keep = to_keep[(to_keep[VCF_POS] >= to_keep['start'] - 5) & (to_keep[VCF_POS] < to_keep['end'] + 5)]
        # to_keep = to_keep[[VARIANT_IDX]].drop_duplicates(VARIANT_IDX)
        #
        # info = pd.merge(info, to_keep, on=VARIANT_IDX)

    if black_listed_exons is not None:
        # info = intersect_variants(vcfdata, black_listed_exons, offset=5, exclude=True).info
        #
        var_info = pd.merge(info, anno, on=VARIANT_IDX)

        to_remove = pd.merge(var_info, black_listed_exons, on=GENE_NAME)

        to_remove = to_remove[(to_remove[VCF_POS] >= to_remove['start'] - 5) & (to_remove[VCF_POS] < to_remove['end'] + 5)]
        to_remove = to_remove[[VARIANT_IDX]].drop_duplicates(VARIANT_IDX)

        info = subtract_dataframes(info, to_remove, [VARIANT_IDX])

    if exclude_intervals is not None:
        info = intersect_variants(vcfdata, exclude_intervals, exclude=True).info

    if variants_to_exclude is not None:
        _info = pd.merge(info,
                         variants_to_exclude.info,
                         on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                         how='left',
                         indicator=True,
                         suffixes=['', '_to_exclude'])

        _info = _info[_info['_merge'] == 'left_only']
        info = _info[list(info)]

    if consequence is not None:

        if consequence in [DELETERIOUS_VARIANT, DELETERIOUS_EXCEPT_PTV, DELETERIOUS_MISSENSE]:
            deleterious_variants = []
            if consequence == DELETERIOUS_VARIANT:
                ptvs = anno[anno[VCF_CONSEQUENCE].isin(ALL_PTV)]
                deleterious_variants.append(ptvs)

            primateAI = anno[(anno[VCF_CONSEQUENCE] == VCF_MISSENSE_VARIANT) &
                             (anno[PRIMATEAI_SCORE] >= anno[PRIMATEAI_SCORE + '_q' + str(min_pAI_deleterious_score_quantile)])]

            deleterious_variants.append(primateAI)

            if consequence != DELETERIOUS_MISSENSE:
                spliceAI = anno[anno[SPLICEAI_MAX_SCORE] >= min_sAI_deleterious_score]
                deleterious_variants.append(spliceAI)

            anno = pd.concat(deleterious_variants).sort_values(VARIANT_IDX).drop_duplicates()

        else:
            if type(consequence) is not list:
                consequence = [consequence]

            anno = anno[anno[VCF_CONSEQUENCE].isin(consequence)]
        info = pd.merge(info, anno[[VARIANT_IDX]].drop_duplicates(), on=VARIANT_IDX)

    if gencode_info is not None:
        anno = pd.merge(anno, gencode_info, how='inner', on=GENE_NAME)[list(anno)]
        info = pd.merge(info, anno[[VARIANT_IDX]].drop_duplicates(), on=VARIANT_IDX)

    if is_canonical is not None:
        anno = anno[anno['CANONICAL'] == is_canonical]
        info = pd.merge(info, anno[[VARIANT_IDX]].drop_duplicates(), on=VARIANT_IDX)

    if chromosomes_to_exclude is not None:
        info = info[~info[VCF_CHROM].isin(chromosomes_to_exclude)]

    if coverage is not None and min_frac is not None:
        COV_COLUMN = 'COVERAGE_HISTOGRAM|over_' + str(coverage)
        if coverage_tag_prefix is not None:
            COV_COLUMN = coverage_tag_prefix + COV_COLUMN

        info = info[info[COV_COLUMN] >= min_frac]

    if external_info is not None:
        info = pd.merge(info,
                        external_info[[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, VCF_AF]],
                        on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                        how='left',
                        suffixes=('_original', ''),
                        validate='one_to_one'
                        ).fillna(0)

    if SNV_only:
        info = info[info.apply(lambda x: len(x[VCF_REF]) == len(x[VCF_ALT]) == 1, axis=1)]

    if non_SNV_only:
        info = info[info.apply(lambda x: len(x[VCF_REF]) != len(x[VCF_ALT]), axis=1)]

    if filter is not None:
        info = info[info[VCF_FILTER] == filter]

    if max_AF is not None:
        info = info[info[VCF_AF] <= max_AF]

    if max_original_AF is not None:
        info = info[info['original_' + VCF_AF] <= max_original_AF]

    if min_original_AN is not None:
        info = info[info['original_' + VCF_AN] >= min_original_AN]

    if max_joint_AC is not None:
        info = info[info['joint_' + VCF_AC] <= max_joint_AC]

    if max_joint_AF is not None:
        info = info[info['joint_' + VCF_AF] <= max_joint_AF]

    if min_joint_AN is not None:
        info = info[info['joint_' + VCF_AN] >= min_joint_AN]

    if keep_exclusive_variants:
        info = info[info['joint_' + VCF_AC] == info[VCF_AC]]

    if min_AF is not None:
        info = info[info[VCF_AF] >= min_AF]

    if min_AN is not None:
        info = info[info[VCF_AN] >= min_AN]

    if max_AC is not None:
        info = info[info[VCF_AC] <= max_AC]

    if max_original_AC is not None:
        info = info[info['original_' + VCF_AC] <= max_original_AC]

    if max_original_AC_male is not None:
        info = info[info['original_' + VCF_AC + '_male'] <= max_original_AC_male]

    if min_AC is not None:
        info = info[info[VCF_AC] >= min_AC]

    if min_primateAI_score is not None:
        anno = anno[anno[PRIMATEAI_SCORE] >= min_primateAI_score]
        info = pd.merge(info, anno[[VARIANT_IDX]].drop_duplicates(), on=VARIANT_IDX)

    if cancer_type is not None:

        echo('Filtering by cancer type:', cancer_type)

        if type(cancer_type) is not list:
            cancer_type = [cancer_type]

        cancer_sample_ids = []

        for ct in cancer_type:

            if ct.startswith('RANDOM'):
                n_random = int(ct.split('|')[1])
                echo('Choosing', n_random, 'samples from TCGA')
                cancer_sample_ids += random.sample(list(tcga_sparse_data[GENOTYPE_SDF]), n_random)

            else:
                cancer_sample_ids += tcga_meta.get_sample_ids_for_project(ct)

        cancer_sample_ids = sorted(set(cancer_sample_ids))

    if cancer_sample_ids is not None:

        echo('Total samples:', len(cancer_sample_ids))

        ct_var_info, ct_var_genotypes = subset_tcga_variants(tcga_sparse_data,
                                                             cancer_sample_ids)

        info = pd.merge(info,
                        ct_var_info,
                        on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                        how='inner',
                        suffixes=('', '_sparse_df'))

    for key, value in kwargs.items():
        echo('Filtering custom keys:', key, value)

        if key.startswith('__max_'):
            op = np.less

        elif key.startswith('__min_'):
            op = np.greater

        elif key.startswith('__eq_'):
            op = np.equal

        else:
            raise Exception("Do not know how to parse argument:", key, value)

        key = '_'.join(key.lstrip('_').split('_')[1:])

        full = pd.merge(anno, info, on=VARIANT_IDX).dropna(subset=[key])

        to_keep = full[op(full[key], value)].drop_duplicates(VARIANT_IDX)[[VARIANT_IDX]]

        info = pd.merge(info, to_keep)
        anno = pd.merge(anno, to_keep)

    anno = pd.merge(anno, info, on=VARIANT_IDX, how='inner')[list(anno)]

    return VcfData(info=info, annotation=anno, sparse_data=sparse_data)


def compute_joint_allele_statistics(dataset_1, dataset_2):

    joint_alleles = pd.merge(dataset_1.info,
                             dataset_2.info,
                             on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                             how='outer').fillna(0)

    for key in [VCF_AN, VCF_AC]:
        joint_alleles['joint_' + key] = (joint_alleles[key + '_x'] + joint_alleles[key + '_y']).astype(int)

    joint_alleles['joint_' + VCF_AF] = joint_alleles['joint_' + VCF_AC] / joint_alleles['joint_' + VCF_AN]

    joint_alleles = joint_alleles[[VCF_CHROM,
                                   VCF_POS,
                                   VCF_REF,
                                   VCF_ALT,
                                   'joint_' + VCF_AC,
                                   'joint_' + VCF_AN,
                                   'joint_' + VCF_AF]].copy()

    res_ds1 = VcfData(info=dataset_1.info,
                      annotation=dataset_1.annotation,
                      make_copy=True)

    res_ds2 = VcfData(info=dataset_2.info,
                      annotation=dataset_2.annotation,
                      make_copy=True)

    for ds in [res_ds1, res_ds2]:

        ds.info = pd.merge(ds.info,
                           joint_alleles,
                           on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])

    gc.collect()
    return res_ds1, res_ds2


def filter_gnomad_populations(vcfdata, filter_nfe=True, gender=None, is_cancer=True, filter_segdup_lcr_decoy=False):

    echo('filter_nfe:', filter_nfe, ', gender:', gender, ', is_cancer:', is_cancer, ', filter_segdup_lcr_decoy:', filter_segdup_lcr_decoy)

    columns = [VARIANT_IDX, VCF_CHROM, VCF_FILTER, VCF_POS, VCF_REF, VCF_ALT] + [l for l in
                                                                                 list(vcfdata.info) if
                                                                                 'COVERAGE' in l]

    if gender is None:
        from_genders = ['', '_male', '_female']
        to_genders = from_genders

    else:
        from_genders = ['_' + gender]
        to_genders = ['']

    for tag in ['AC', 'AF', 'AN', 'nhomalt']:
        for _gender in to_genders:
            columns.append(tag + _gender)

    if filter_segdup_lcr_decoy:
        echo('Filtering segdup, lcr and decoy variants')

        info = vcfdata.info[(vcfdata.info['segdup'] == 0) &
                            (vcfdata.info['lcr'] == 0) &
                            (vcfdata.info['decoy'] == 0)].copy()
    else:
        info = vcfdata.info.copy()

    population = ''

    if filter_nfe:
        info = info[(info['AC_nfe'] > 0) & (info['popmax'] == 'nfe')]
        population = '_nfe'

    for cancer_status in ['', 'non_cancer_']:
        for pop in ['', '_nfe']:
            for tag in ['AC', 'AF', 'AN', 'nhomalt']:
                for _gender in ['', '_male', '_female']:

                    label = cancer_status + tag + pop + _gender
                    new_label = 'original_' + label

                    info[new_label] = info[label]

                    if tag in ['AC', 'AN', 'nhomalt']:
                        info[new_label] = info[new_label].fillna(0).astype(int)

                    columns.append(new_label)

    for to_gender, from_gender in zip(to_genders, from_genders):

        if is_cancer:
            info[VCF_AC + to_gender] = info[VCF_AC + population + from_gender] - info['non_cancer_AC' + population + from_gender]
            info[VCF_AN + to_gender] = info[VCF_AN + population + from_gender] - info['non_cancer_AN' + population + from_gender]
            info[VCF_AF + to_gender] = info[VCF_AC + from_gender] / info[VCF_AN + from_gender]
            info['nhomalt' + to_gender] = info['nhomalt' + population + from_gender] - info['non_cancer_nhomalt' + population + from_gender]

        else:

            info[VCF_AC + to_gender] = info['non_cancer_AC' + population + from_gender]
            info[VCF_AN + to_gender] = info['non_cancer_AN' + population + from_gender]
            info[VCF_AF + to_gender] = info['non_cancer_AF' + population + from_gender]
            info['nhomalt' + to_gender] = info['non_cancer_nhomalt' + population + from_gender]



    gc.collect()

    info = info.fillna(0)
    gc.collect()

    info = info[info[VCF_AC] > 0]
    gc.collect()

    info = info[columns]
    gc.collect()

    for tag in list(info):
        if any(_t in tag for _t in [VCF_AC, VCF_AN, 'nhomalt']):
            info[tag] = info[tag].astype(int)

    anno = vcfdata.annotation

    # if gencode_info is not None:
    #     anno = pd.merge(anno,
    #                     gencode_info,
    #                     left_on=['Feature', GENE_NAME],
    #                     right_on=['transcript_id', GENE_NAME])[list(anno)].copy()

    info = pd.merge(info, anno[[VARIANT_IDX]].drop_duplicates(), on=VARIANT_IDX, how='inner').copy()

    anno = pd.merge(anno, info[[VARIANT_IDX]], on=VARIANT_IDX, how='inner')

    return VcfData(info=info, annotation=anno)


# print(len(filter_variants(gnomad_exomes_cancer_only,
#                           info_condition=(gnomad_exomes_cancer_only.info['AC_nfe'] > 0)).info))


########################################################################################################################
########################################################################################################################
########################################################################################################################


def get_all_variants_for_gene(gene_name, vcfdata, consequence=None, max_AF=None, max_AC=None, max_joint_AC=None):
    vars_info = vcfdata.info
    vars_anno = vcfdata.annotation

    gene_vars = vars_anno[(vars_anno[GENE_NAME] == gene_name)]

    if consequence is not None:
        if type(consequence) is not list:
            consequence = [consequence]
        gene_vars = gene_vars[gene_vars[VCF_CONSEQUENCE].isin(consequence)]

    result = pd.merge(gene_vars,
                      vars_info,
                      on='index')

    if max_AF is not None:
        result = result[result[VCF_AF] <= max_AF]

    if max_AC is not None:
        result = result[result[VCF_AC] <= max_AC]

    if max_joint_AC is not None:
        result = result[result['joint_' + VCF_AC] <= max_joint_AC]

    return result


def get_tcga_sample_info_for_variants(vcfdata_or_vcfdatainfo, tcga_sparse_data, tcga_meta):

    if type(vcfdata_or_vcfdatainfo) is VcfData:
        vars_info = vcfdata_or_vcfdatainfo.info
    else:
        vars_info = vcfdata_or_vcfdatainfo

    n_vars = len(vars_info)

    col_names = list(vars_info)

    res = dict((col, []) for col in col_names)
    res[TCGA_SAMPLE_ID] = []

    for var_idx in range(n_vars):

        var_chrom = vars_info[VCF_CHROM].iloc[var_idx]

        var_pos = vars_info[VCF_POS].iloc[var_idx]
        var_ref = vars_info[VCF_REF].iloc[var_idx]
        var_alt = vars_info[VCF_ALT].iloc[var_idx]

        sample_ids = get_TCGA_samples_with_variant(var_chrom,
                                                   var_pos,
                                                   var_ref,
                                                   var_alt,
                                                   tcga_sparse_data)
        #         echo(var_pos, sample_ids)

        if len(sample_ids) == 0:
            sample_ids = [None]

        for s_id in sample_ids:
            for col in col_names:
                res[col].append(vars_info[col].iloc[var_idx])

            res[TCGA_SAMPLE_ID].append(s_id)

    return pd.merge(pd.DataFrame(res),
                    tcga_meta.get_sample_info(res[TCGA_SAMPLE_ID]),
                    on=TCGA_SAMPLE_ID,
                    how='left')


########################################################################################################################
########################################################################################################################
########################################################################################################################

### ENRICHMENT TESTS


def get_gene_counts(vcfdata, set_label, consequence, consequence_label=None, use_allele_counts=False):

    if type(consequence) is not list:
        consequence = [consequence]

    if consequence_label is None:
        consequence_label = ','.join(consequence)

    nvars_label = consequence_label + '|n|' + set_label

    if consequence_label.endswith('.SNV'):
        vcfdata = filter_variants(vcfdata, SNV_only=True)

    elif consequence_label.endswith('.INDEL'):
        vcfdata = filter_variants(vcfdata, non_SNV_only=True)

    if use_allele_counts:
        result = pd.merge(vcfdata.info,
                          vcfdata.annotation[vcfdata.annotation[CONSEQUENCE].isin(consequence)],
                          on=VARIANT_IDX,
                          how='inner').groupby(GENE_NAME).agg({VCF_AC: 'sum'}).reset_index()

        result.columns = [GENE_NAME, nvars_label]
        result.sort_values(nvars_label, ascending=False)

    else:

        result = pd.merge(vcfdata.info,
                          vcfdata.annotation[vcfdata.annotation[CONSEQUENCE].isin(consequence)],
                          on=VARIANT_IDX,
                          how='inner').groupby(GENE_NAME).size().reset_index(name=nvars_label).sort_values(nvars_label,
                                                                                                           ascending=False)
    return result


def chi2_or_fisher_test(table):
    """ Apply Fisher exact test in cases where at least one expect value is less than 5"""

    chi2, pvalue, dof, expected = scipy.stats.chi2_contingency(table)

    if (expected < 5).any():
        return scipy.stats.fisher_exact(table)[1]
    else:
        return pvalue


MIN_ODDS_RATIO = 10**-10


def odds_ratio(fgr, fgr_total, bgr, bgr_total):

    # fgr += .5
    # fgr_total += 1
    # bgr += .5
    # bgr_total += 1

    if fgr == 0 or bgr == 0 or fgr_total - fgr == 0 or bgr_total - bgr == 0:
    #
    #     # pseudo_count_fgr = 1
    #     # pseudo_count_bgr = pseudo_count_fgr * bgr_total / fgr_total
    #     #
    #     # fgr += pseudo_count_fgr
    #     # fgr_total += 2 * pseudo_count_fgr
    #     #
    #     # bgr += pseudo_count_bgr
    #     # bgr_total += 2 * pseudo_count_bgr

        if fgr == 0 or bgr_total - bgr == 0:
            return MIN_ODDS_RATIO

        else: # bgr == 0 or fgr_total - fgr == 0:
            return 1 / MIN_ODDS_RATIO

    def _odds(k, n):
        return k / (n - k)

    return _odds(fgr, fgr_total) / _odds(bgr, bgr_total)


def compute_test_variant_counts_on_gene_lists(vars_per_gene_1,
                                              set1_label,
                                              vars_per_gene_2,
                                              set2_label,
                                              cons_label,
                                              gene_sets=None):

    res = pd.merge(vars_per_gene_1,
                   vars_per_gene_2,
                   on=GENE_NAME,
                   how='outer').fillna(0)

    nvars_label = cons_label + '|n|'

    set1_counts_label = nvars_label + set1_label
    set2_counts_label = nvars_label + set2_label

    total_x = sum(res[set1_counts_label])
    total_y = sum(res[set2_counts_label])

    if gene_sets is not None:

        gene_sets_dict = {GENE_NAME: [],
                          set1_counts_label: [],
                          set2_counts_label: []}

        for gene_set, gene_set_label in gene_sets:

            set_var_counts = res[res[GENE_NAME].isin(gene_set)]

            total_set1_counts = sum(set_var_counts[set1_counts_label])
            total_set2_counts = sum(set_var_counts[set2_counts_label])

            gene_sets_dict[GENE_NAME].append(gene_set_label)
            gene_sets_dict[set1_counts_label].append(total_set1_counts)
            gene_sets_dict[set2_counts_label].append(total_set2_counts)

        res = pd.DataFrame(gene_sets_dict)

    # pseudo_count_x = .5
    # pseudo_count_y = pseudo_count_x * total_y / total_x

    def _chisquare(c_x, c_y):
        if (total_x > 0 and
                total_y > 0 and
                c_x + c_y > 0 and
                total_x - c_x + total_y - c_y > 0):

            #         return scipy.stats.chisquare([c_x, total_x - c_x],
            #                                      [total_x * float(c_y) / total_y,
            #                                       total_x * (float(total_y - c_y)) / total_y])[1]

            return chi2_or_fisher_test([[c_x, total_x - c_x],
                                        [c_y, total_y - c_y]])
        else:
            return np.nan

    def _binom_test(c_x, c_y, alternative):
        return scipy.stats.binom_test(c_x,
                                      n=total_x,
                                      p=c_y / total_y,
                                      alternative=alternative)

    # def _enrichment(c_x, c_y):
    #     if c_x == 0 or c_y == 0:
    #         return ((c_x + pseudo_count_x) / (total_x + pseudo_count_x)) / ((c_y + pseudo_count_y) / (total_y + pseudo_count_y))
    #     else:
    #         return (c_x / total_x) / (c_y / total_y)

    pvalue_label = cons_label + '|chi2_pvalue'
    pvalues = res.apply(lambda r: _chisquare(r[1], r[2]), axis=1)

    #     pvalue_label = cons_label + '|binom_pvalue'
    #     pvalues = res.apply(lambda r: _binom_test(r[1], r[2], alternative), axis=1)

    res[pvalue_label] = pvalues
    res[cons_label + '|chi2_FDR'] = statsmodels.stats.multitest.multipletests(pvalues, method='fdr_bh')[1]

    # enrichments = res.apply(lambda r: _enrichment(r[1], r[2]), axis=1)
    #
    # res[cons_label + '|enr'] = enrichments
    res[cons_label + '|OR'] = res.apply(lambda r: odds_ratio(r[1], total_x, r[2], total_y), axis=1)

    res[cons_label + '|total|' + set1_label] = int(total_x)
    res[cons_label + '|total|' + set2_label] = int(total_y)

    return res.sort_values(pvalue_label, ascending=True)


def variant_counts_test(vcfdata1,
                        set1_label,
                        vcfdata2,
                        set2_label,
                        consequence,
                        cons_label=None,
                        use_allele_counts=False,
                        gene_sets=None):

    if cons_label is None:
        if type(consequence) is list:
            cons_label = ','.join(consequence)
        else:
            cons_label = consequence

    set1_gene_counts = get_gene_counts(vcfdata1, set1_label, consequence, cons_label, use_allele_counts=use_allele_counts)
    set2_gene_counts = get_gene_counts(vcfdata2, set2_label, consequence, cons_label, use_allele_counts=use_allele_counts)

    return compute_test_variant_counts_on_gene_lists(set1_gene_counts,
                                                     set1_label,
                                                     set2_gene_counts,
                                                     set2_label,
                                                     cons_label,
                                                     gene_sets=gene_sets)


def get_primateai_per_gene(vcfdata, consequence, primateai_with_gene_names, median_primateai_per_gene=None, use_allele_counts=False):

    if type(consequence) is not list:
        consequence = [consequence]

    result = pd.merge(pd.merge(vcfdata.info,
                               vcfdata.annotation[vcfdata.annotation[CONSEQUENCE].isin(consequence)],
                               on=VARIANT_IDX,
                               how='inner'),
                      primateai_with_gene_names,
                      on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, GENE_NAME],
                      how='inner',
                      suffixes=['', '_primateAI'])

    if use_allele_counts:
        allele_counts = list(result[VCF_AC])

        result = result[[VARIANT_IDX, GENE_NAME, PRIMATEAI_SCORE]]

        total_alleles = sum(allele_counts)

        def replicate(column):

            # echo('replicate call')
            new_values = [None] * total_alleles
            new_values_cur_idx = 0

            for idx, v in enumerate(column):
                for j in range(allele_counts[idx]):

                    new_values[new_values_cur_idx] = v
                    new_values_cur_idx += 1

            return pd.Series(new_values)

        result = result.apply(replicate, axis=0)

    # if median_primateai_per_gene is supplied, keep only variants above the median score for each gene
    if median_primateai_per_gene is not None:
        result = pd.merge(result, median_primateai_per_gene, how='inner', on=GENE_NAME)
        result = result[result[PRIMATEAI_SCORE] > result['median_' + PRIMATEAI_SCORE]]

    return result


def rank_INT(series, c=3.0 / 8, stochastic=True):
    """ Perform rank-based inverse normal transformation on pandas series.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.
        Args:
            param1 (pandas.Series):   Series of values to transform
            param2 (Optional[float]): Constand parameter (Bloms constant)
            param3 (Optional[bool]):  Whether to randomise rank of ties

        Returns:
            pandas.Series
    """

    # Check input
    assert (isinstance(series, pd.Series))
    assert (isinstance(c, float))
    assert (isinstance(stochastic, bool))

    # Set seed
    np.random.seed(123)

    # Take original series indexes
    orig_idx = series.index

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get ranks
    if stochastic:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = scipy.stats.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = scipy.stats.rankdata(series, method="average")

    transformed = pd.Series(rank_to_normal(rank, c=c, n=len(rank)),
                            index=series.index)

    return transformed[orig_idx]


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return scipy.stats.norm.ppf(x)


def remove_sample_outliers(sparse_data, k_percent=0.025):
    # removes top 2.5% and bottom 2.5% of the data

    total_alleles_per_sample = sparse_data.to_coo().tocsc().sum(axis=0).A[0, :]
    sorted_total_alleles_per_sample = sorted(total_alleles_per_sample)

    sample_ids = list(sparse_data)
    n_samples = len(sample_ids)

    bottom_quantile = sorted_total_alleles_per_sample[int(n_samples * k_percent)]
    upper_quantile = sorted_total_alleles_per_sample[n_samples - int(n_samples * k_percent)]

    to_keep = np.where((bottom_quantile < total_alleles_per_sample) & (total_alleles_per_sample < upper_quantile))[0]

    return sparse_data.iloc[:, to_keep] #sparse_data[to_keep.index[to_keep]]




def logit_variant_count_test(cases,
                             cases_label,
                             controls,
                             controls_label,

                             consequence,
                             covariates=None,

                             cons_label=None,

                             max_joint_AC=None,
                             coverage=None,
                             min_frac=None,

                             use_per_individual_allele_counts_as_covariate=False,
                             gene_set=None):


    def logit_test(x,
                   # cases,
                   cases_label,
                   n_cases_samples,
                   all_cases_variants,
                   # controls,

                   controls_label,
                   n_controls_samples,
                   all_controls_variants,

                   logit_response,
                   constant_vector,
                   covariates_df,

                   logit_response_with_pseudo_count,
                   constant_vector_with_pseudo_count,
                   covariates_df_with_pseudo_count):

        # if logit_test.count % 1000 == 0:
        #     echo('genes processed:', logit_test.count)

        logit_test.count += 1

        # if logit_test.count < 3143:
        #     return pd.Series({cons_label + '|logit_pvalue': np.nan,
        #
        #                   cons_label + '|OR': np.nan,
        #                   cons_label + '|OR_95%_CI_low': np.nan,
        #                   cons_label + '|OR_95%_CI_high': np.nan,
        #
        #                   cons_label + '|n|' + cases_label: np.nan,
        #                   cons_label + '|n|' + controls_label: np.nan,
        #
        #                   cons_label + '|samples|' + cases_label: np.nan,
        #                   cons_label + '|samples|' + controls_label: np.nan,
        #
        #
        #                   })


        # n_variants = len(x)

        var_indexes = x[VARIANT_IDX]

        cases_variants = all_cases_variants[var_indexes, :].tocsc()
        n_cases_variants = np.sum(cases_variants)

        controls_variants = all_controls_variants[var_indexes, :].tocsc()
        n_controls_variants = np.sum(controls_variants)


        if n_cases_variants == 0 and n_controls_variants == 0:
            echo('Warning: no variants found in gene:', x.iloc[0][GENE_NAME])
            pvalue = np.nan
            OR = np.nan
            OR_95_CI_low = np.nan
            OR_95_CI_high = np.nan

        else:
            if n_cases_variants == 0 or n_controls_variants == 0:
                # echo('using pseudo counts')
                logit_test.n_with_pseudocount += 1

                logit_response = logit_response_with_pseudo_count
                constant_vector = constant_vector_with_pseudo_count
                cases_vector = np.append(cases_variants.sum(axis=0).A[0, :], 1)
                controls_vector = np.append(controls_variants.sum(axis=0).A[0, :], 1)

                covariates_df = covariates_df_with_pseudo_count

            else:

                cases_vector = cases_variants.sum(axis=0).A[0, :]
                controls_vector = controls_variants.sum(axis=0).A[0, :]

            if covariates_df is not None:
                predictors = covariates_df.copy()
            else:
                predictors = pd.DataFrame()

            # echo('genes processed with logit test:', logit_test.count - 1, np.sum(cases_vector), np.sum(controls_vector))

            predictors['const'] = constant_vector
            predictors['rare_variants'] = np.concatenate((cases_vector, controls_vector))
            rare_variants_idx = -1
            # echo(x.iloc[0][GENE_NAME])#, np.sum(cases_vector), np.sum(controls_vector))

            import statsmodels.api as sm
            logit_mod = sm.Logit(logit_response, predictors)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                logit_res = logit_mod.fit(disp=0, method='lbfgs')

            #
            beta_conf95_low,  beta_conf95_high = logit_res.conf_int().iloc[rare_variants_idx]

            pvalue = logit_res.pvalues[rare_variants_idx]
            OR = np.exp(logit_res.params[rare_variants_idx])

            OR_95_CI_low = np.exp(beta_conf95_low)
            OR_95_CI_high = np.exp(beta_conf95_high)
            #
            # echo('gene name:',
            #      x.iloc[0][GENE_NAME],
            #      np.sum(cases_vector),
            #      np.sum(controls_vector),
            #      pvalue,
            #      OR,
            #      OR_95_CI_low,
            #      OR_95_CI_high)

            if x.iloc[0][GENE_NAME] in ['LDLR', 'PCSK9']:
                echo(x.iloc[0][GENE_NAME])
                print(logit_res.summary())

        return pd.Series({cons_label + '|logit_pvalue': pvalue,

                          cons_label + '|OR': OR,
                          cons_label + '|OR_95%_CI_low': OR_95_CI_low,
                          cons_label + '|OR_95%_CI_high': OR_95_CI_high,

                          cons_label + '|n|' + cases_label: n_cases_variants,
                          cons_label + '|n|' + controls_label: n_controls_variants,

                          cons_label + '|samples|' + cases_label: n_cases_samples,
                          cons_label + '|samples|' + controls_label: n_controls_samples,


                          })

    logit_test.count = 0
    logit_test.n_with_pseudocount = 0

    if cons_label is None:
        if type(consequence) is list:
            cons_label = ','.join(consequence)
        else:
            cons_label = consequence

    echo('Filtering foreground variants')
    cases = filter_variants(cases,
                            max_joint_AC=max_joint_AC,
                            coverage=coverage,
                            min_frac=min_frac,
                            consequence=consequence,
                            gene_set=gene_set)

    echo('Filtering background variants')
    controls = filter_variants(controls,
                               max_joint_AC=max_joint_AC,
                               coverage=coverage,
                               min_frac=min_frac,
                               consequence=consequence,
                               gene_set=gene_set)

    # variants = annotations[annotations[VCF_CONSEQUENCE].isin(consequence)]
    #
    # cases_variants = pd.merge(cases.info, variants, on=VARIANT_IDX)[list(variants)]
    # controls_variants = pd.merge(controls.info, variants, on=VARIANT_IDX)[list(variants)]


    # cases_variants = cases.annotation
    # controls_variants = controls.annotation
    variants = pd.concat([cases.annotation, controls.annotation], ignore_index=True).drop_duplicates()

    n_cases_samples = len(list(cases.sparse_data))
    n_controls_samples = len(list(controls.sparse_data))

    logit_response = [1] * n_cases_samples + [0] * n_controls_samples
    constant_vector = [1] * (n_cases_samples + n_controls_samples)

    logit_response_with_pseudo_count = [1] * (n_cases_samples + 1) + [0] * (n_controls_samples + 1)

    constant_vector_with_pseudo_count = [1] * (n_cases_samples + n_controls_samples + 2)

    all_cases_variants = cases.sparse_data.to_coo().tocsr()
    all_controls_variants = controls.sparse_data.to_coo().tocsr()

    covariates_df = None
    covariates_df_with_pseudo_count = None

    if use_per_individual_allele_counts_as_covariate:
        # var_indexes =
        var_indexes = [int(ind) for ind in variants[VARIANT_IDX]]

        echo('Total variants:', len(var_indexes))

        total_alleles_per_case = all_cases_variants[var_indexes, :].tocsc().sum(axis=0).A[0, :]
        # total_alleles_per_case = all_cases_variants[var_indexes, :]
        # total_alleles_per_case = total_alleles_per_case.tocsc()
        # total_alleles_per_case = total_alleles_per_case.sum(axis=0)
        # total_alleles_per_case = total_alleles_per_case.A[0, :]


        echo('total and average AC per case sample:',
             np.sum(total_alleles_per_case),
             np.mean(total_alleles_per_case))

        total_alleles_per_control = all_controls_variants[var_indexes, :].tocsc().sum(axis=0).A[0, :]
        echo('total and average AC per control sample:',
             np.sum(total_alleles_per_control),
             np.mean(total_alleles_per_control))

        sample_ids = list(cases.sparse_data) + list(controls.sparse_data)

        n_alleles_df = pd.DataFrame({SAMPLE_ID: sample_ids,
                                     'N_ALLELES': rank_INT(pd.Series(np.concatenate((total_alleles_per_case,
                                                                                     total_alleles_per_control))))})

        if covariates is None:
            covariates = n_alleles_df
        else:
            covariates = pd.merge(covariates, n_alleles_df, on=SAMPLE_ID)


    if covariates is not None:

        covariate_names = [c for c in list(covariates) if c not in [SAMPLE_ID, SAMPLE_STATUS]]

        sample_ids = list(cases.sparse_data) + list(controls.sparse_data)
        sample_ids_df = pd.DataFrame({SAMPLE_ID: sample_ids,
                                      'sample_id_index': range(len(sample_ids))})
        covariates_df = pd.merge(covariates,
                                 sample_ids_df,
                                 on=SAMPLE_ID).sort_values('sample_id_index')[covariate_names]

        sample_ids_with_pseudo_count = list(cases.sparse_data) + ['___PSEUDO_1___'] + list(controls.sparse_data) + ['___PSEUDO_2___']
        sample_ids_df_with_pseudo_count = pd.DataFrame({SAMPLE_ID: sample_ids_with_pseudo_count,
                                                       'sample_id_index': range(len(sample_ids) + 2)})

        covariates_df_with_pseudo_count = pd.merge(covariates,
                                                   sample_ids_df_with_pseudo_count,
                                                   on=SAMPLE_ID,
                                                   how='right').fillna(0).sort_values('sample_id_index')[covariate_names]


    res = variants.groupby(GENE_NAME).apply(logit_test,
                                            # cases,
                                            cases_label,
                                            n_cases_samples,
                                            all_cases_variants,
                                            # controls,
                                            controls_label,
                                            n_controls_samples,
                                            all_controls_variants,

                                            logit_response,
                                            constant_vector,
                                            covariates_df,

                                            logit_response_with_pseudo_count,
                                            constant_vector_with_pseudo_count,
                                            covariates_df_with_pseudo_count
                                            ).sort_values(cons_label + '|logit_pvalue')

    res = res.reset_index()
    # res[GENE_NAME] = res.index

    echo('Genes with added pseudo counts:', logit_test.n_with_pseudocount)
    return res


def case_control_logit_variant_count_test_for_deleterious_variants(cases,
                                                                   cases_label,
                                                                   controls,
                                                                   controls_label,

                                                                   covariates=None,

                                                                   max_joint_AC=None,
                                                                   coverage=None,
                                                                   min_frac=None,

                                                                   primateai_scores=None,
                                                                   spliceai_scores=None,

                                                                   use_per_individual_allele_counts_as_covariate=False,
                                                                   gene_set=None):


    def logit_test(x,
                   # cases,
                   cases_label,
                   n_cases_samples,
                   all_cases_variants,
                   # controls,

                   controls_label,
                   n_controls_samples,
                   all_controls_variants,

                   logit_response,
                   constant_vector,
                   covariates_df,

                   logit_response_with_pseudo_count,
                   constant_vector_with_pseudo_count,
                   covariates_df_with_pseudo_count):

        # if logit_test.count % 1000 == 0:
        #     echo('genes processed:', logit_test.count)

        logit_test.count += 1

        # if logit_test.count < 3143:
        #     return pd.Series({cons_label + '|logit_pvalue': np.nan,
        #
        #                   cons_label + '|OR': np.nan,
        #                   cons_label + '|OR_95%_CI_low': np.nan,
        #                   cons_label + '|OR_95%_CI_high': np.nan,
        #
        #                   cons_label + '|n|' + cases_label: np.nan,
        #                   cons_label + '|n|' + controls_label: np.nan,
        #
        #                   cons_label + '|samples|' + cases_label: np.nan,
        #                   cons_label + '|samples|' + controls_label: np.nan,
        #
        #
        #                   })


        # n_variants = len(x)

        var_indexes = x[VARIANT_IDX]

        cases_variants = all_cases_variants[var_indexes, :].tocsc()
        n_cases_variants = np.sum(cases_variants)

        controls_variants = all_controls_variants[var_indexes, :].tocsc()
        n_controls_variants = np.sum(controls_variants)


        if n_cases_variants == 0 and n_controls_variants == 0:
            # echo('Warning: no variants found in gene:', x.iloc[0][GENE_NAME])
            pvalue = np.nan
            OR = np.nan
            OR_95_CI_low = np.nan
            OR_95_CI_high = np.nan

        else:
            if n_cases_variants == 0 or n_controls_variants == 0:
                # echo('using pseudo counts')
                logit_test.n_with_pseudocount += 1

                logit_response = logit_response_with_pseudo_count
                constant_vector = constant_vector_with_pseudo_count
                cases_vector = np.append(cases_variants.sum(axis=0).A[0, :], 1)
                controls_vector = np.append(controls_variants.sum(axis=0).A[0, :], 1)

                covariates_df = covariates_df_with_pseudo_count

            else:

                cases_vector = cases_variants.sum(axis=0).A[0, :]
                controls_vector = controls_variants.sum(axis=0).A[0, :]

            if covariates_df is not None:
                predictors = covariates_df.copy()
            else:
                predictors = pd.DataFrame()

            # echo('genes processed with logit test:', logit_test.count - 1, np.sum(cases_vector), np.sum(controls_vector))

            predictors['const'] = constant_vector
            predictors['rare_variants'] = np.concatenate((cases_vector, controls_vector))
            rare_variants_idx = -1
            # echo(x.iloc[0][GENE_NAME])#, np.sum(cases_vector), np.sum(controls_vector))

            import statsmodels.api as sm
            logit_mod = sm.Logit(logit_response, predictors)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                logit_res = logit_mod.fit(disp=0, method='lbfgs')

            #
            beta_conf95_low,  beta_conf95_high = logit_res.conf_int().iloc[rare_variants_idx]

            pvalue = logit_res.pvalues[rare_variants_idx]
            OR = np.exp(logit_res.params[rare_variants_idx])

            OR_95_CI_low = np.exp(beta_conf95_low)
            OR_95_CI_high = np.exp(beta_conf95_high)
            #
            # echo('gene name:',
            #      x.iloc[0][GENE_NAME],
            #      np.sum(cases_vector),
            #      np.sum(controls_vector),
            #      pvalue,
            #      OR,
            #      OR_95_CI_low,
            #      OR_95_CI_high)

            if x.iloc[0][GENE_NAME] in ['LDLR', 'PCSK9']:
                echo(x.iloc[0][GENE_NAME])
                print(logit_res.summary())

        return pd.Series({cons_label + '|logit_pvalue': pvalue,

                          cons_label + '|OR': OR,
                          cons_label + '|OR_95%_CI_low': OR_95_CI_low,
                          cons_label + '|OR_95%_CI_high': OR_95_CI_high,

                          cons_label + '|n|' + cases_label: n_cases_variants,
                          cons_label + '|n|' + controls_label: n_controls_variants,

                          cons_label + '|samples|' + cases_label: n_cases_samples,
                          cons_label + '|samples|' + controls_label: n_controls_samples,


                          })

    logit_test.count = 0
    logit_test.n_with_pseudocount = 0

    cons_label = 'deleterious_variants'

    echo('Filtering foreground variants')
    cases = filter_variants(cases,
                            max_joint_AC=max_joint_AC,
                            coverage=coverage,
                            min_frac=min_frac,
                            gene_set=gene_set)

    echo('Filtering background variants')
    controls = filter_variants(controls,
                               max_joint_AC=max_joint_AC,
                               coverage=coverage,
                               min_frac=min_frac,
                               gene_set=gene_set)

    def get_primateai_scores(vcfdata, primateai_scores):
        result = pd.merge(pd.merge(vcfdata.info,
                                 vcfdata.annotation[vcfdata.annotation[CONSEQUENCE] == VCF_MISSENSE_VARIANT],
                                 on=VARIANT_IDX,
                                 how='inner'),
                        primateai_scores,
                        on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, GENE_NAME],
                        how='inner',
                        suffixes=['', '_primateAI'])

        return result

    cases_primateai = get_primateai_scores(cases, primateai_scores)
    controls_primateai = get_primateai_scores(controls, primateai_scores)

    def get_spliceai_scores(vcfdata, spliceai_scores):
        anno = vcfdata.annotation[~vcfdata.annotation[VCF_CONSEQUENCE].isin([VCF_SPLICE_ACCEPTOR_VARIANT,
                                                                             VCF_SPLICE_DONOR_VARIANT])]

        info = pd.merge(vcfdata.info, anno, on=VARIANT_IDX)

        result = pd.merge(info,
                          spliceai_scores,
                          on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, GENE_NAME])

        return result

    cases_spliceai = get_spliceai_scores(cases, spliceai_scores)
    controls_spliceai = get_spliceai_scores(controls, spliceai_scores)

    def get_weights_for_regression(sample_genotypes, primateai, spliceai, ptvs):
        weights = scipy.sparse.dok_matrix(sample_genotypes.shape, dtype=np.float)

        for df in [spliceai, primateai, ptvs]:

            for index, row in df.iterrows():
                if df is ptvs:
                    score = 1
                else:
                    score = row['max spliceAI score' if df is spliceai else 'primateAI score']

                variant_idx = row[VARIANT_IDX]
                row_genotypes = sample_genotypes[variant_idx, :].nonzero()
                for row_idx, col_idx in zip(*row_genotypes):
                    weights[variant_idx, col_idx] = score

        weights = weights.tocsr()

        return weights



    all_cases_scores = cases.sparse_data.to_coo().tocsr()
    all_controls_scores = controls.sparse_data.to_coo().tocsr()

    cases_ptvs = filter_variants(cases, consequence=ALL_PTV)
    controls_ptvs = filter_variants(controls, consequence=ALL_PTV)

    all_cases_weights = get_weights_for_regression(all_cases_scores, cases_primateai, cases_spliceai, cases_ptvs.info)
    all_controls_weights = get_weights_for_regression(all_controls_scores, controls_primateai, controls_spliceai, controls_ptvs.info)


    # variants = annotations[annotations[VCF_CONSEQUENCE].isin(consequence)]
    #
    # cases_variants = pd.merge(cases.info, variants, on=VARIANT_IDX)[list(variants)]
    # controls_variants = pd.merge(controls.info, variants, on=VARIANT_IDX)[list(variants)]


    # cases_variants = cases.annotation
    # controls_variants = controls.annotation
    variants = pd.concat([cases.annotation, controls.annotation], ignore_index=True).drop_duplicates()

    n_cases_samples = len(list(cases.sparse_data))
    n_controls_samples = len(list(controls.sparse_data))

    logit_response = [1] * n_cases_samples + [0] * n_controls_samples
    constant_vector = [1] * (n_cases_samples + n_controls_samples)

    logit_response_with_pseudo_count = [1] * (n_cases_samples + 1) + [0] * (n_controls_samples + 1)

    constant_vector_with_pseudo_count = [1] * (n_cases_samples + n_controls_samples + 2)

    all_cases_variants = cases.sparse_data.to_coo().tocsr()
    all_controls_variants = controls.sparse_data.to_coo().tocsr()

    covariates_df = None
    covariates_df_with_pseudo_count = None

    if use_per_individual_allele_counts_as_covariate:
        # var_indexes =
        var_indexes = [int(ind) for ind in variants[VARIANT_IDX]]

        echo('Total variants:', len(var_indexes))

        total_alleles_per_case = all_cases_variants[var_indexes, :].tocsc().sum(axis=0).A[0, :]
        # total_alleles_per_case = all_cases_variants[var_indexes, :]
        # total_alleles_per_case = total_alleles_per_case.tocsc()
        # total_alleles_per_case = total_alleles_per_case.sum(axis=0)
        # total_alleles_per_case = total_alleles_per_case.A[0, :]


        echo('total and average AC per case sample:',
             np.sum(total_alleles_per_case),
             np.mean(total_alleles_per_case))

        total_alleles_per_control = all_controls_variants[var_indexes, :].tocsc().sum(axis=0).A[0, :]
        echo('total and average AC per control sample:',
             np.sum(total_alleles_per_control),
             np.mean(total_alleles_per_control))

        sample_ids = list(cases.sparse_data) + list(controls.sparse_data)

        n_alleles_df = pd.DataFrame({SAMPLE_ID: sample_ids,
                                     'N_ALLELES': rank_INT(pd.Series(np.concatenate((total_alleles_per_case,
                                                                                     total_alleles_per_control))))})

        if covariates is None:
            covariates = n_alleles_df
        else:
            covariates = pd.merge(covariates, n_alleles_df, on=SAMPLE_ID)


    if covariates is not None:

        covariate_names = [c for c in list(covariates) if c not in [SAMPLE_ID, SAMPLE_STATUS]]

        sample_ids = list(cases.sparse_data) + list(controls.sparse_data)
        sample_ids_df = pd.DataFrame({SAMPLE_ID: sample_ids,
                                      'sample_id_index': range(len(sample_ids))})
        covariates_df = pd.merge(covariates,
                                 sample_ids_df,
                                 on=SAMPLE_ID).sort_values('sample_id_index')[covariate_names]

        sample_ids_with_pseudo_count = list(cases.sparse_data) + ['___PSEUDO_1___'] + list(controls.sparse_data) + ['___PSEUDO_2___']
        sample_ids_df_with_pseudo_count = pd.DataFrame({SAMPLE_ID: sample_ids_with_pseudo_count,
                                                       'sample_id_index': range(len(sample_ids) + 2)})

        covariates_df_with_pseudo_count = pd.merge(covariates,
                                                   sample_ids_df_with_pseudo_count,
                                                   on=SAMPLE_ID,
                                                   how='right').fillna(0).sort_values('sample_id_index')[covariate_names]


    res = variants.groupby(GENE_NAME).apply(logit_test,
                                            # cases,
                                            cases_label,
                                            n_cases_samples,
                                            all_cases_weights,
                                            # controls,
                                            controls_label,
                                            n_controls_samples,
                                            all_controls_weights,

                                            logit_response,
                                            constant_vector,
                                            covariates_df,

                                            logit_response_with_pseudo_count,
                                            constant_vector_with_pseudo_count,
                                            covariates_df_with_pseudo_count
                                            ).sort_values(cons_label + '|logit_pvalue')

    res = res.reset_index()
    # res[GENE_NAME] = res.index

    echo('Genes with added pseudo counts:', logit_test.n_with_pseudocount)
    return res


def ranksum_test(x, fgr_dataset, bgr_dataset):
    fgr_scores = x[x['dataset'] == fgr_dataset][PRIMATEAI_SCORE]
    bgr_scores = x[x['dataset'] == bgr_dataset][PRIMATEAI_SCORE]

    u, p_value = scipy.stats.mannwhitneyu(fgr_scores, bgr_scores, alternative='greater')

    return pd.Series({'ranksum_pvalue': p_value,
                      'U': u,

                      fgr_dataset + ' #': len(fgr_scores),
                      bgr_dataset + ' #': len(bgr_scores),

                      fgr_dataset + ' m_score': np.mean(fgr_scores),
                      bgr_dataset + ' m_score': np.mean(bgr_scores)

                      })


def ranksum_primateai(fgr_vcfdata,
                      fgr_label,

                      bgr_vcfdata,
                      bgr_label,

                      consequence,
                      primateai_with_gene_names,
                      use_allele_counts=False):

    fgr_primateai = get_primateai_per_gene(fgr_vcfdata, consequence, primateai_with_gene_names, use_allele_counts=use_allele_counts)
    bgr_primateai = get_primateai_per_gene(bgr_vcfdata, consequence, primateai_with_gene_names, use_allele_counts=use_allele_counts)

    d = pd.concat([fgr_primateai.assign(dataset=fgr_label),
                   bgr_primateai.assign(dataset=bgr_label)], copy=False) \
        .groupby(GENE_NAME) \
        .apply(ranksum_test, fgr_label, bgr_label).sort_values('ranksum_pvalue')

    d = d.reset_index()

    return d


def primateai_test(fgr_vcfdata,
                   fgr_label,

                   bgr_vcfdata,
                   bgr_label,

                   primateai_with_gene_names,
                   primateai_quantile=0.5,

                   use_allele_counts=False):

    PRIMATEAI_OVER_MEDIAN_TEST = 'primateAI_over_%.2lf_quantile_within_gene' % primateai_quantile

    def significance_test(x, fgr_label, bgr_label, median_primateai_per_gene):

        fgr_scores = x[x['dataset'] == fgr_label][PRIMATEAI_SCORE]
        bgr_scores = x[x['dataset'] == bgr_label][PRIMATEAI_SCORE]

        gene_name = x.iloc[0][GENE_NAME]

        # transcript_id = x.iloc[0][TRANSCRIPT_ID]
        #
        # print(gene_name)

        #         print('AAAAAA')
        #         print(x)

        #         if gene_name != 'A1BG':
        #             raise "AA"

        score_threshold = median_primateai_per_gene.loc[gene_name]['median_' + PRIMATEAI_SCORE]

        count_geq = lambda a: len([el for el in a if el > score_threshold])

        fgr_greater_than_threshold = count_geq(fgr_scores)
        bgr_greater_than_threshold = count_geq(bgr_scores)

        total_fgr = len(fgr_scores)
        total_bgr = len(bgr_scores)

        if (total_fgr > 0 and
            total_bgr > 0 and
            fgr_greater_than_threshold + bgr_greater_than_threshold > 0 and
            total_fgr - fgr_greater_than_threshold + total_bgr - bgr_greater_than_threshold > 0):

            #             test_pvalue = scipy.stats.binom_test(fgr_greater_than_threshold,
            #                                                   n=total_fgr,
            #                                                   p=bgr_greater_than_threshold / total_bgr,
            #                                                   alternative='greater')

            #             test_pvalue = scipy.stats.chisquare([fgr_greater_than_threshold,
            #                                             total_fgr - fgr_greater_than_threshold],

            #                                            [total_fgr * bgr_greater_than_threshold / total_bgr,
            #                                             total_fgr * (total_bgr - bgr_greater_than_threshold) / total_bgr])[1]

            test_pvalue = chi2_or_fisher_test([[fgr_greater_than_threshold, total_fgr - fgr_greater_than_threshold],
                                               [bgr_greater_than_threshold, total_bgr - bgr_greater_than_threshold]])

            odds_r = odds_ratio(fgr_greater_than_threshold,
                                total_fgr,
                                bgr_greater_than_threshold,
                                total_bgr)

            # test_pvalue = 1
            # odds_r = 1
        else:
            test_pvalue = np.nan
            # enr = np.nan
            odds_r = np.nan


        return pd.Series({  # 'chi^2_pvalue': p_value,
            # PRIMATEAI_OVER_MEDIAN_TEST + '|enr': enr,
            PRIMATEAI_OVER_MEDIAN_TEST + '|OR': odds_r,
            #                           'p_value2':p_value2,

            PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_pvalue': test_pvalue,

            #                           'primateAI|' + consequence + '|binom_pvalue': binom_pvalue,
            PRIMATEAI_OVER_MEDIAN_TEST + '|t': score_threshold,

            PRIMATEAI_OVER_MEDIAN_TEST + '|>t #|' + fgr_label: fgr_greater_than_threshold,
            PRIMATEAI_OVER_MEDIAN_TEST+ '|total #|' + fgr_label: total_fgr,

            PRIMATEAI_OVER_MEDIAN_TEST + '|>t #|' + bgr_label: bgr_greater_than_threshold,
            PRIMATEAI_OVER_MEDIAN_TEST + '|total #|' + bgr_label: total_bgr
        })

    fgr_primateai = get_primateai_per_gene(fgr_vcfdata, VCF_MISSENSE_VARIANT, primateai_with_gene_names, use_allele_counts=use_allele_counts)
    bgr_primateai = get_primateai_per_gene(bgr_vcfdata, VCF_MISSENSE_VARIANT, primateai_with_gene_names, use_allele_counts=use_allele_counts)

    median_primateai_per_gene = primateai_with_gene_names.loc[:, [GENE_NAME, PRIMATEAI_SCORE]].groupby(
        GENE_NAME).quantile(q=primateai_quantile).rename(columns={PRIMATEAI_SCORE: 'median_' + PRIMATEAI_SCORE})

    res = pd.concat([fgr_primateai.assign(dataset=fgr_label),
                     bgr_primateai.assign(dataset=bgr_label)], copy=False) \
                    .groupby(GENE_NAME) \
                    .apply(significance_test, fgr_label, bgr_label, median_primateai_per_gene).sort_values(
                    PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_pvalue')

    res[PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_FDR'] = np.nan

    non_nan_genes = ~res[PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_pvalue'].isnull()
    non_nan_pvalues = res[non_nan_genes]

    res[PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_FDR'][non_nan_genes] = statsmodels.stats.multitest.multipletests(non_nan_pvalues[PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_pvalue'], method='fdr_bh')[1]

    # res[PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_FDR'] = statsmodels.stats.multitest.multipletests(res[PRIMATEAI_OVER_MEDIAN_TEST + '|chi2_pvalue'], method='fdr_bh')[1]

    res = res.reset_index()

    return res


PRIMATEAI_BEST_THRESHOLD_TEST = 'primateAI_best_threshold_global'

def primateai_best_threshold_test(fgr_vcfdata,
                                  fgr_label,

                                  bgr_vcfdata,
                                  bgr_label,

                                  primateai_with_gene_names,
                                  use_allele_counts=False):

    def best_threshold_significance_test(x, fgr_label, fgr_total, bgr_label, bgr_total):

        fgr_scores = sorted(x[x['dataset'] == fgr_label][PRIMATEAI_SCORE])
        bgr_scores = sorted(x[x['dataset'] == bgr_label][PRIMATEAI_SCORE])

        best_threshold = np.nan
        best_pvalue = 1
        # best_enrichment = 1
        best_odds_ratio = np.nan

        fgr_idx = 0
        bgr_idx = 0

        best_fgr_greater = 0
        best_bgr_greater = 0

        MAX_NUMBER_OF_THRESHOLDS = 25

        distinct_scores = sorted(set(fgr_scores + bgr_scores + [0]))

        if len(distinct_scores) <= MAX_NUMBER_OF_THRESHOLDS:
            thresholds = distinct_scores
        else:
            thresholds = np.linspace(0, 1, MAX_NUMBER_OF_THRESHOLDS)

        for score_threshold in thresholds:

            while fgr_idx < len(fgr_scores) and fgr_scores[fgr_idx] <= score_threshold:
                fgr_idx += 1

            fgr_greater = len(fgr_scores) - fgr_idx

            while bgr_idx < len(bgr_scores) and bgr_scores[bgr_idx] <= score_threshold:
                bgr_idx += 1

            bgr_greater = len(bgr_scores) - bgr_idx

            if fgr_greater + bgr_greater > 0:

                pvalue = chi2_or_fisher_test([[fgr_greater, fgr_total - fgr_greater], [bgr_greater, bgr_total - bgr_greater]])

                if pvalue <= best_pvalue:

                    best_pvalue = pvalue
                    best_threshold = score_threshold

                    best_fgr_greater = fgr_greater
                    best_bgr_greater = bgr_greater

                    best_odds_ratio = odds_ratio(fgr_greater,
                                                 fgr_total,
                                                 bgr_greater,
                                                 bgr_total)

                    # if fgr_greater == 0 or bgr_greater == 0:
                    #
                    #     pseudo_count_x = .5
                    #     pseudo_count_y = pseudo_count_x * bgr_total / fgr_total
                    #
                    #     best_enrichment = ((pseudo_count_x + fgr_greater) / (pseudo_count_x + fgr_total)) / (
                    #                        (pseudo_count_y + bgr_greater) / (pseudo_count_y + bgr_total))
                    # else:
                    #
                    #     best_enrichment = (fgr_greater / fgr_total) / (bgr_greater / bgr_total)

        return pd.Series({  #PRIMATEAI_BEST_THRESHOLD_TEST + '|enr': best_enrichment,
                            PRIMATEAI_BEST_THRESHOLD_TEST + '|OR': best_odds_ratio,

                            PRIMATEAI_BEST_THRESHOLD_TEST + '|chi2_pvalue': min(best_pvalue * len(thresholds), 1),  # correct for multiple testing

                            PRIMATEAI_BEST_THRESHOLD_TEST + '|t': best_threshold,

                            PRIMATEAI_BEST_THRESHOLD_TEST + '|>t #|' + fgr_label: best_fgr_greater,
                            PRIMATEAI_BEST_THRESHOLD_TEST + '|total #|' + fgr_label: fgr_total,

                            PRIMATEAI_BEST_THRESHOLD_TEST + '|>t #|' + bgr_label: best_bgr_greater,
                            PRIMATEAI_BEST_THRESHOLD_TEST + '|total #|' + bgr_label: bgr_total
                            })

    fgr_primateai = get_primateai_per_gene(fgr_vcfdata, VCF_MISSENSE_VARIANT, primateai_with_gene_names, use_allele_counts=use_allele_counts)
    bgr_primateai = get_primateai_per_gene(bgr_vcfdata, VCF_MISSENSE_VARIANT, primateai_with_gene_names, use_allele_counts=use_allele_counts)

    fgr_total = len(fgr_primateai)
    bgr_total = len(bgr_primateai)

    res = pd.concat([fgr_primateai.assign(dataset=fgr_label),
                     bgr_primateai.assign(dataset=bgr_label)], copy=False) \
        .groupby(GENE_NAME) \
        .apply(best_threshold_significance_test, fgr_label, fgr_total, bgr_label, bgr_total).sort_values(
        PRIMATEAI_BEST_THRESHOLD_TEST + '|chi2_pvalue')

    res[PRIMATEAI_BEST_THRESHOLD_TEST + '|chi2_FDR'] = statsmodels.stats.multitest.multipletests(res[PRIMATEAI_BEST_THRESHOLD_TEST + '|chi2_pvalue'], method='fdr_bh')[1]

    res = res.reset_index()

    return res


def missense_counts_test_over_median_primateai_score(fgr_vcfdata,
                                                     fgr_label,

                                                     bgr_vcfdata,
                                                     bgr_label,

                                                     primateai_with_gene_names,
                                                     primateai_quantile=0.5,
                                                     use_allele_counts=False,
                                                     gene_sets=None):

    # This method counts all missense variants with higher primateAI score than the median score for each gene
    # and performs a chi2 test using all variants in the cohort

    consequence_label = 'primateAI_over_%.2lf_quantile_global' % primateai_quantile

    median_primateai_per_gene = primateai_with_gene_names.loc[:, [GENE_NAME, PRIMATEAI_SCORE]].groupby(
        GENE_NAME).quantile(q=primateai_quantile).rename(columns={PRIMATEAI_SCORE: 'median_' + PRIMATEAI_SCORE})

    fgr_primateai = get_primateai_per_gene(fgr_vcfdata, VCF_MISSENSE_VARIANT, primateai_with_gene_names, median_primateai_per_gene, use_allele_counts=use_allele_counts)
    bgr_primateai = get_primateai_per_gene(bgr_vcfdata, VCF_MISSENSE_VARIANT, primateai_with_gene_names, median_primateai_per_gene, use_allele_counts=use_allele_counts)

    def count_variants_per_gene(varinfo, set_label):
        nvars_label = consequence_label + '|n|' + set_label
        return varinfo.groupby(GENE_NAME).size().reset_index(name=nvars_label).sort_values(nvars_label,
                                                                                           ascending=False)

    fgr_variants_per_gene = count_variants_per_gene(fgr_primateai, fgr_label)
    bgr_variants_per_gene = count_variants_per_gene(bgr_primateai, bgr_label)

    result = compute_test_variant_counts_on_gene_lists(fgr_variants_per_gene,
                                                       fgr_label,
                                                       bgr_variants_per_gene,
                                                       bgr_label,
                                                       consequence_label,
                                                       gene_sets=gene_sets)

    return result


def count_spliceai_variants_per_gene(vcfinfo, set_label, use_allele_counts, consequence_label, filtered_spliceai, exclude_known_splice_junctions=False):

    nvars_label = consequence_label + '|n|' + set_label

    if exclude_known_splice_junctions:

        anno = vcfinfo.annotation[~vcfinfo.annotation[VCF_CONSEQUENCE].isin([VCF_SPLICE_ACCEPTOR_VARIANT,
                                                                             VCF_SPLICE_DONOR_VARIANT])]

        info = pd.merge(vcfinfo.info, anno[[VARIANT_IDX]].drop_duplicates(), on=VARIANT_IDX)

        vcfinfo = VcfData(info=info, annotation=anno)

    result = pd.merge(vcfinfo.info,
                      filtered_spliceai,
                      on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])

    if use_allele_counts:
        allele_counts = list(result[VCF_AC])

        result = result[[VARIANT_IDX, GENE_NAME, SPLICEAI_MAX_SCORE]]

        total_alleles = sum(allele_counts)

        def replicate(column):

            # echo('replicate call')
            new_values = [None] * total_alleles
            new_values_cur_idx = 0

            for idx, v in enumerate(column):
                for j in range(allele_counts[idx]):
                    new_values[new_values_cur_idx] = v
                    new_values_cur_idx += 1

            return pd.Series(new_values)

        result = result.apply(replicate, axis=0)

    result = result.groupby(GENE_NAME).size().reset_index(name=nvars_label).sort_values(nvars_label, ascending=False)

    return result


def test_spliceai_for_gnomad_exomes(fgr_vcfdata,
                                    fgr_label,

                                    bgr_vcfdata,
                                    bgr_label,

                                    spliceai_info,
                                    spliceai_cutoff=0.5,
                                    use_allele_counts=False,
                                    gene_sets=None):

    consequence_label = 'spliceAI_over_' + str(spliceai_cutoff)
    filtered_spliceai = spliceai_info[spliceai_info[SPLICEAI_MAX_SCORE] > spliceai_cutoff]

    fgr_spliceai_per_gene = count_spliceai_variants_per_gene(fgr_vcfdata, fgr_label, use_allele_counts, consequence_label, filtered_spliceai)
    bgr_spliceai_per_gene = count_spliceai_variants_per_gene(bgr_vcfdata, bgr_label, use_allele_counts, consequence_label, filtered_spliceai)

    result = compute_test_variant_counts_on_gene_lists(fgr_spliceai_per_gene,
                                                       fgr_label,
                                                       bgr_spliceai_per_gene,
                                                       bgr_label,
                                                       consequence_label,
                                                       gene_sets=gene_sets)

    return result



def counts_test_for_deleterious_variants(fgr_vcfdata,
                                         fgr_label,

                                         bgr_vcfdata,
                                         bgr_label,

                                         primateai_with_gene_names,

                                         spliceai_info=None,
                                         spliceai_cutoff=0.5,

                                         use_allele_counts=False,
                                         gene_sets=None):

    consequence_label = 'deleterious_variants'

    primateai_test_results = primateai_best_threshold_test(fgr_vcfdata,
                                                           fgr_label,

                                                           bgr_vcfdata,
                                                           bgr_label,

                                                           primateai_with_gene_names,
                                                           use_allele_counts=use_allele_counts)

    # get PTV variants per gene
    fgr_PTVs = get_gene_counts(fgr_vcfdata, fgr_label, ALL_PTV, consequence_label, use_allele_counts=use_allele_counts)
    bgr_PTVs = get_gene_counts(bgr_vcfdata, bgr_label, ALL_PTV, consequence_label, use_allele_counts=use_allele_counts)

    def add_variant_counts(vars1, vars2, set_label):

        n_vars_label = consequence_label + '|n|' + set_label

        res = pd.merge(vars1, vars2, on=GENE_NAME, how='outer').fillna(0)
        res[n_vars_label] = res[n_vars_label + '_x'] + res[n_vars_label + '_y']

        return res[[GENE_NAME, n_vars_label]]

    fgr_deleterious_missense_variants_per_gene = primateai_test_results[
        [GENE_NAME, PRIMATEAI_BEST_THRESHOLD_TEST + '|>t #|' + fgr_label]] \
        .rename(columns={PRIMATEAI_BEST_THRESHOLD_TEST + '|>t #|' + fgr_label: consequence_label + '|n|' + fgr_label})

    bgr_deleterious_missense_variants_per_gene = primateai_test_results[
        [GENE_NAME, PRIMATEAI_BEST_THRESHOLD_TEST + '|>t #|' + bgr_label]] \
        .rename(columns={PRIMATEAI_BEST_THRESHOLD_TEST + '|>t #|' + bgr_label: consequence_label + '|n|' + bgr_label})

    fgr_variants_per_gene = add_variant_counts(fgr_PTVs, fgr_deleterious_missense_variants_per_gene, fgr_label)
    bgr_variants_per_gene = add_variant_counts(bgr_PTVs, bgr_deleterious_missense_variants_per_gene, bgr_label)

    if spliceai_info is not None:
        ## fetch spliceAI variants
        filtered_spliceai = spliceai_info[spliceai_info[SPLICEAI_MAX_SCORE] >= spliceai_cutoff]

        fgr_spliceai_per_gene = count_spliceai_variants_per_gene(fgr_vcfdata, fgr_label, use_allele_counts, consequence_label, filtered_spliceai, exclude_known_splice_junctions=True)
        bgr_spliceai_per_gene = count_spliceai_variants_per_gene(bgr_vcfdata, bgr_label, use_allele_counts, consequence_label, filtered_spliceai, exclude_known_splice_junctions=True)

        fgr_variants_per_gene = add_variant_counts(fgr_variants_per_gene, fgr_spliceai_per_gene, fgr_label)
        bgr_variants_per_gene = add_variant_counts(bgr_variants_per_gene, bgr_spliceai_per_gene, bgr_label)

    result = compute_test_variant_counts_on_gene_lists(fgr_variants_per_gene,
                                                       fgr_label,
                                                       bgr_variants_per_gene,
                                                       bgr_label,
                                                       consequence_label,
                                                       gene_sets=gene_sets)

    return result


def dataset_stats(fgr_vcfdata=None,

                  bgr_vcfdata=None,

                  af_set=None,

                  max_AF=None,
                  max_AC=None,

                  max_joint_AC=None,

                  max_original_AC=None,
                  max_original_AF=None,
                  max_original_AC_male=None,

                  coverage=None,

                  min_frac_samples_with_coverage=None,
                  chromosomes_to_exclude=None,

                  primateai_with_gene_names=None,

                  gencode_info=None,

                  cancer_type=None,
                  gnomad_samples_per_cancer_type=None,
                  tcga_sparse_data=None,
                  tcga_meta=None,
                  ):

    protein_coding_genes = None

    if primateai_with_gene_names is not None:
        protein_coding_genes = primateai_with_gene_names[[GENE_NAME]].drop_duplicates()

    if gencode_info is not None:
        if protein_coding_genes is not None:
            protein_coding_genes = pd.merge(protein_coding_genes, gencode_info[gencode_info['gene_type'] == 'protein_coding'])
        else:
            protein_coding_genes = gencode_info[gencode_info['gene_type'] == 'protein_coding']

    protein_coding_genes = protein_coding_genes[[GENE_NAME]].drop_duplicates()

    n_fgr_samples = max(fgr_vcfdata.info[VCF_AN]) / 2

    fgr_variants = filter_variants(fgr_vcfdata,

                                   coverage=coverage,
                                   min_frac=min_frac_samples_with_coverage,

                                   external_info=af_set,
                                   max_AF=max_AF,
                                   max_AC=max_AC,

                                   max_original_AC=max_original_AC,
                                   max_original_AF=max_original_AF,
                                   max_original_AC_male=max_original_AC_male,
                                   max_joint_AC=max_joint_AC,
                                   cancer_type=cancer_type,
                                   tcga_sparse_data=tcga_sparse_data,
                                   tcga_meta=tcga_meta,

                                   chromosomes_to_exclude=chromosomes_to_exclude,
                                   gencode_info=protein_coding_genes
                                   )

    echo('Total foreground variants:', len(fgr_variants.info))
    echo('Total foreground samples:', n_fgr_samples)

    for var_type in [None, VCF_MISSENSE_VARIANT, VCF_SYNONYMOUS_VARIANT, ALL_PTV]:
        fgr = filter_variants(fgr_variants, consequence=var_type)
        echo((str(var_type) if var_type is not None else 'All variants'), 'per sample:', sum(fgr.info[VCF_AC]) / n_fgr_samples)

    # if gnomad_samples_per_cancer_type is not None:
    #     if cancer_type is None:
    #         n_fgr_samples_estimate = sum(len(gnomad_samples_per_cancer_type[k]) for k in gnomad_samples_per_cancer_type)
    #     else:
    #         n_fgr_samples_estimate = len(gnomad_samples_per_cancer_type[cancer_type])
    #
    #     echo('Estimated n_samples:', n_fgr_samples_estimate)
    #     echo('Foreground alleles per sample (estimated):', sum(fgr_variants.info[VCF_AC]) / n_fgr_samples_estimate)

    n_bgr_samples = max(bgr_vcfdata.info[VCF_AN]) / 2

    bgr_variants = filter_variants(bgr_vcfdata,

                                   coverage=coverage,
                                   min_frac=min_frac_samples_with_coverage,

                                   external_info=af_set,
                                   max_AF=max_AF,
                                   max_AC=max_AC,
                                   max_original_AC=max_original_AC,
                                   max_original_AF=max_original_AF,
                                   max_original_AC_male=max_original_AC_male,

                                   max_joint_AC=max_joint_AC,

                                   chromosomes_to_exclude=chromosomes_to_exclude,
                                   gencode_info=protein_coding_genes)

    echo('Total background variants:', len(bgr_variants.info))
    echo('Total background samples:', n_bgr_samples)

    for var_type in [None, VCF_MISSENSE_VARIANT, VCF_SYNONYMOUS_VARIANT, ALL_PTV]:
        bgr = filter_variants(bgr_variants, consequence=var_type)
        echo((str(var_type) if var_type is not None else 'All variants'), 'per sample:', sum(bgr.info[VCF_AC]) / n_bgr_samples)

    # echo('Background alleles per sample:', sum(bgr_variants.info[VCF_AC]) / n_bgr_samples)



def compute_variant_stats(fgr_vcfdata=None,
                          fgr_label=None,

                          bgr_vcfdata=None,
                          bgr_label=None,

                          af_set=None,

                          max_AF=None,
                          max_AC=None,

                          max_joint_AC=None,

                          keep_exclusive_variants=False,

                          max_original_AC=None,
                          max_original_AF=None,
                          max_original_AC_male=None,
                          min_AN=None,


                          coverage=None,
                          filter_fgr_by_coverage=True,

                          min_frac_samples_with_coverage=None,
                          chromosomes_to_exclude=None,

                          primateAI_tests=True,
                          primateai_with_gene_names=None,
                          primateai_quantile=0.5,

                          gencode_info=None,

                          cancer_type=None,
                          gnomad_samples_per_cancer_type=None,
                          tcga_sparse_data=None,
                          tcga_meta=None,
                          use_allele_counts=False,


                          spliceai_info=None,
                          spliceai_cutoff=0.5,

                          gene_sets=None,

                          variant_types_to_test=None,
                          white_listed_exons=None,
                          black_listed_exons=None
                          ):

    echo('Cancer type:', 'ALL' if cancer_type is None else cancer_type)

    protein_coding_genes = None

    if primateai_with_gene_names is not None:
        protein_coding_genes = primateai_with_gene_names[[GENE_NAME]].drop_duplicates()

    if gencode_info is not None:
        if protein_coding_genes is not None:
            protein_coding_genes = pd.merge(protein_coding_genes, gencode_info[gencode_info['gene_type'] == 'protein_coding'])
        else:
            protein_coding_genes = gencode_info[gencode_info['gene_type'] == 'protein_coding']

    protein_coding_genes = protein_coding_genes[[GENE_NAME]].drop_duplicates()

    n_fgr_samples = max(fgr_vcfdata.info[VCF_AN]) / 2

    fgr_variants = filter_variants(fgr_vcfdata,

                                   coverage=coverage if filter_fgr_by_coverage else None,
                                   min_frac=min_frac_samples_with_coverage if filter_fgr_by_coverage else None,

                                   external_info=af_set,
                                   max_AF=max_AF,
                                   max_AC=max_AC,
                                   min_AN=min_AN,

                                   max_original_AC=max_original_AC,
                                   max_original_AF=max_original_AF,
                                   max_original_AC_male=max_original_AC_male,
                                   max_joint_AC=max_joint_AC,

                                   keep_exclusive_variants=keep_exclusive_variants,

                                   cancer_type=cancer_type,
                                   tcga_sparse_data=tcga_sparse_data,
                                   tcga_meta=tcga_meta,

                                   chromosomes_to_exclude=chromosomes_to_exclude,
                                   gencode_info=protein_coding_genes,
                                   white_listed_exons=white_listed_exons,
                                   black_listed_exons=black_listed_exons)


    echo('Total foreground variants:', len(fgr_variants.info))
    echo('Total foreground samples:', n_fgr_samples)
    echo('Foreground alleles per sample:', sum(fgr_variants.info[VCF_AC]) / n_fgr_samples)

    if gnomad_samples_per_cancer_type is not None:
        if cancer_type is None:
            n_fgr_samples_estimate = sum(len(gnomad_samples_per_cancer_type[k]) for k in gnomad_samples_per_cancer_type)
        else:
            n_fgr_samples_estimate = len(gnomad_samples_per_cancer_type[cancer_type])

        echo('Estimated n_samples:', n_fgr_samples_estimate)
        echo('Foreground alleles per sample (estimated):', sum(fgr_variants.info[VCF_AC]) / n_fgr_samples_estimate)

    n_bgr_samples = max(bgr_vcfdata.info[VCF_AN]) / 2

    bgr_variants = filter_variants(bgr_vcfdata,

                                   coverage=coverage,
                                   min_frac=min_frac_samples_with_coverage,

                                   external_info=af_set,
                                   max_AF=max_AF,
                                   max_AC=max_AC,
                                   max_original_AC=max_original_AC,
                                   max_original_AF=max_original_AF,
                                   max_original_AC_male=max_original_AC_male,
                                   min_AN=min_AN,

                                   max_joint_AC=max_joint_AC,

                                   keep_exclusive_variants=keep_exclusive_variants,

                                   chromosomes_to_exclude=chromosomes_to_exclude,
                                   gencode_info=protein_coding_genes,
                                   white_listed_exons=white_listed_exons,
                                   black_listed_exons=black_listed_exons
                                   )

    echo('Total background variants:', len(bgr_variants.info))
    echo('Total background samples:', n_bgr_samples)
    echo('Background alleles per sample:', sum(bgr_variants.info[VCF_AC]) / n_bgr_samples)

    result = None

    def merge_results(prev_result, cur_result):
        if prev_result is None:
            return cur_result
        else:
            return pd.merge(prev_result,
                            cur_result,
                            on=GENE_NAME,
                            how='outer')

    if spliceai_info is not None:
        echo('Comparing: SpliceAI variants')
        cur_result = test_spliceai_for_gnomad_exomes(fgr_variants,
                                                     fgr_label,

                                                     bgr_variants,
                                                     bgr_label,

                                                     spliceai_info,
                                                     spliceai_cutoff=spliceai_cutoff,
                                                     use_allele_counts=use_allele_counts,
                                                     gene_sets=gene_sets)

        result = merge_results(result, cur_result)

    if primateAI_tests:
        # perform all tests that involve primateAI scores
        if gene_sets is None:

            echo('Comparing: deleterious variants')
            cur_result = counts_test_for_deleterious_variants(fgr_variants,
                                                              fgr_label,

                                                              bgr_variants,
                                                              bgr_label,

                                                              primateai_with_gene_names,
                                                              spliceai_info=spliceai_info,
                                                              spliceai_cutoff=spliceai_cutoff,
                                                              use_allele_counts=use_allele_counts)

            result = merge_results(result, cur_result)

        if gene_sets is None:

            echo('Comparing: missense variants above gene-specific best threshold for PrimateAI scores')
            cur_result = primateai_best_threshold_test(fgr_variants,
                                                       fgr_label,

                                                       bgr_variants,
                                                       bgr_label,

                                                       primateai_with_gene_names,
                                                       use_allele_counts=use_allele_counts)
            result = merge_results(result, cur_result)

        # echo('Comparing missense variants above gene-specific median threshold for PrimateAI scores')
        # cur_result = missense_counts_test_over_median_primateai_score( fgr_variants,
        #                                                                fgr_label,
        #
        #                                                                bgr_variants,
        #                                                                bgr_label,
        #
        #                                                                primateai_with_gene_names,
        #                                                                primateai_quantile=primateai_quantile,
        #                                                                use_allele_counts=use_allele_counts,
        #                                                                gene_sets=gene_sets)
        #
        # result = merge_results(result, cur_result)
        #
        # if gene_sets is None:
        #     echo('Comparing PrimateAI scores within the same gene for missense variants')
        #     cur_result = primateai_test(fgr_variants,
        #                                 fgr_label,
        #
        #                                 bgr_variants,
        #                                 bgr_label,
        #
        #                                 primateai_with_gene_names=primateai_with_gene_names,
        #                                 primateai_quantile=primateai_quantile,
        #                                 use_allele_counts=use_allele_counts)
        #
        #     result = merge_results(result, cur_result)

    if variant_types_to_test is None:
        variant_types_to_test = [(VCF_MISSENSE_VARIANT, 'missense'),
                                 # ([VCF_STOP_GAINED, VCF_FRAMESHIFT_VARIANT], 'stop,frameshift'),
                                 # (VCF_SPLICE_REGION_VARIANT, 'splice_region'),
                                 # ([VCF_SPLICE_DONOR_VARIANT, VCF_SPLICE_ACCEPTOR_VARIANT], 'splice_do,ac'),
                                 # (VCF_NMD_TRANSCRIPT_VARIANT, 'NMD'),
                                 (ALL_PTV, 'all_PTV'),
                                 # (ALL_PTV, 'all_PTV.SNV'),
                                 # (ALL_PTV, 'all_PTV.INDEL'),
                                 (VCF_SYNONYMOUS_VARIANT, 'syn')]

    for consequence, cons_label in variant_types_to_test:

        echo('Comparing:', consequence)

        cur_result = variant_counts_test(fgr_variants,
                                         fgr_label,
                                         bgr_variants,
                                         bgr_label,
                                         consequence,
                                         cons_label=cons_label,
                                         use_allele_counts=use_allele_counts,
                                         gene_sets=gene_sets)  # ,
#                                          alternative='two-sided' if consequence == VCF_SYNONYMOUS_VARIANT else 'greater')

        result = merge_results(result, cur_result)

    return result


def compute_combined_pvalues(compute_variant_stats_result, COLUMNS_TO_COMBINE=None):

    result = compute_variant_stats_result

    if COLUMNS_TO_COMBINE is None:
        COLUMNS_TO_COMBINE= [#'primateAI|missense_variant|',
                              'missense|',
                              'all_PTV|']

    echo('Computing combined p-values for:', COLUMNS_TO_COMBINE)

    def combine_pvalues(row):
        pvals = [row[col_name + 'chi2_pvalue'] if not pd.isnull(row[col_name + 'chi2_pvalue']) else 1 for col_name
                 in COLUMNS_TO_COMBINE]

        combined_pvalue = scipy.stats.combine_pvalues(pvals, method='fisher')[1]
        return combined_pvalue

    def combine_odds_ratios(row):
        ORs = [row[col_name + 'OR'] if not pd.isnull(row[col_name + 'OR']) else 1 for col_name in
                COLUMNS_TO_COMBINE]

        combined_OR = np.exp(sum(np.log(ORs)) / len(ORs))

        return combined_OR

    combined_label = ','.join([c.replace('|', '') for c in COLUMNS_TO_COMBINE])

    result[combined_label + '|combined_pvalue'] = result.apply(lambda row: combine_pvalues(row), axis=1)
    result[combined_label + '|combined_OR'] = result.apply(lambda row: combine_odds_ratios(row), axis=1)

    result = result[~pd.isnull(result[combined_label + '|combined_pvalue'])]
    result[combined_label + '|combined_fdr'] = statsmodels.stats.multitest.multipletests(result[combined_label + '|combined_pvalue'], method='fdr_bh')[1]

    result = result.sort_values(by=[combined_label + '|combined_pvalue', combined_label + '|combined_OR'], ascending=[True, False])

    return result


# def _compare_to_gwas(summary_stats, test_label, gwas_snps, finemapped_gene_column='eQTL_gene_name'):
#     EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN = 'EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN'
#     GWAS_OR = 'OR'
#     GWAS_ID = 'ID'
#     GWAS_PVALUE = 'P-VALUE'
#
#     if finemapped_gene_column == 'eQTL_gene_name':
#         gwas_snps = gwas_snps[gwas_snps['eQTL_total_eQTLs'] == 1]
#         cols_to_slice = [finemapped_gene_column,
#                          GWAS_ID,
#                          GWAS_PVALUE,
#                          'eQTL_tissues',
#                          'eQTL_max_effect_size',
#                          GWAS_OR,
#
#                          EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN]
#     else:
#         cols_to_slice = [finemapped_gene_column,
#                          GWAS_ID,
#                          'CHROM',
#                          'pos',
#                          'REF',
#                          'ALT',
#                          'pics2_Linked_SNP',
#                          'pics2_Linked_RefAlt',
#                          'pics2_Linked_position',
#                          GWAS_OR,
#                          GWAS_PVALUE,
#                          EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN]
#
#     res = pd.merge(gwas_snps[cols_to_slice].rename(columns={'CHROM': 'INDEX_CHROM',
#                                                             'pos': 'INDEX_POS',
#                                                             'REF': 'INDEX_REF',
#                                                             'ALT': 'INDEX_ALT',
#                                                             'OR': 'GWAS_OR',
#                                                             'P-VALUE': 'GWAS_P-VALUE'}),
#                    sort_variant_stats_by_test(summary_stats,
#                                               pvalue_label=test_label + '|chi2_pvalue'),
#
#                    right_on=GENE_NAME,
#                    left_on=finemapped_gene_column)
#
#     if 'pics2_Linked_position' in list(res):
#         res['pics2_Linked_position'] = res['pics2_Linked_position'].str.replace('|', ':')
#
#     res['WIN'] = (((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '+') & (res[test_label + '|OR'] > 1)) |
#                   ((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '-') & (res[test_label + '|OR'] < 1)))
#
#     cols = res.columns.tolist()
#
#     #     n_total = len(res)
#
#     n_pos = sum(res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '+')
#     n_neg = sum(res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '-')
#
#     n_pos_wins = sum((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '+') & res['WIN'])
#     n_neg_wins = sum((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '-') & res['WIN'])
#
#     p = sum(res[test_label + '|OR'] > 1) / len(res)
#
#     p_value = sum(sum(scipy.special.binom(n_pos, k) *
#                       scipy.special.binom(n_neg, l) *
#                       p ** (k + n_neg - l) *
#                       (1 - p) ** (n_pos - k + l)
#                       for l in range(n_neg_wins, n_neg + 1))
#                   for k in range(n_pos_wins, n_pos + 1))
#
#     print (n_pos_wins + n_neg_wins, 'wins out of', n_pos + n_neg, 'P(OR>1)=', p, ', p-value=', p_value)
#
#     return res[cols[-1:] + cols[:-1]].drop([GENE_NAME], axis=1)


def compare_to_gwas(summary_stats,
                    test_label,
                    gwas_snps,
                    EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN='EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN',
                    GWAS_OR='OR',
                    GWAS_ID='variant',
                    GWAS_PVALUE='pvalue'

                    ):

    summary_stats = summary_stats[~summary_stats[test_label + '|chi2_pvalue'].isnull()]

    PROBLEMS = 'PROBLEMS'
    NO_PROBLEMS = ['OK', '.']


    if PROBLEMS in list(gwas_snps):
        gwas_snps = gwas_snps[gwas_snps[PROBLEMS].isin(NO_PROBLEMS)]

    cols_to_slice = [GENE_NAME,
                     GWAS_ID,
                     VCF_CHROM,
                     VCF_POS,
                     VCF_REF,
                     VCF_ALT,
                     GWAS_OR,
                     GWAS_PVALUE,
                     EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN]


    res = pd.merge(gwas_snps[cols_to_slice],
                   sort_variant_stats_by_test(summary_stats,
                                              pvalue_label=test_label + '|chi2_pvalue'),
                   on=GENE_NAME)

    # echo(len(res))

    res['WIN'] = (((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '+') & (res[test_label + '|OR'] > 1)) |
                  ((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '-') & (res[test_label + '|OR'] < 1)))

    cols = list(res)

    n_pos = sum(res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '+')
    n_neg = sum(res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '-')

    n_pos_wins = sum((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '+') & res['WIN'])
    n_neg_wins = sum((res[EXPECTED_RARE_VARIANTS_LOG_ODDS_RATIO_SIGN] == '-') & res['WIN'])

    # p = sum(res[test_label + '|OR'] > 1) / len(res)
    p_pos = sum(summary_stats[test_label + '|OR'] > 1) / len(summary_stats)

    p_value = sum(sum(scipy.special.binom(n_pos, k) *
                      scipy.special.binom(n_neg, l) *
                      p_pos ** (k + n_neg - l) *
                        (1 - p_pos) ** (n_pos - k + l)
                    for l in range(n_neg_wins, n_neg + 1))
                for k in range(n_pos_wins, n_pos + 1))

    echo('custom pvalue:', p_value)

    p_value_pos = scipy.stats.binom_test(n_pos_wins, n_pos, p=p_pos, alternative='greater')
    p_value_neg = scipy.stats.binom_test(n_neg_wins, n_neg, p=1-p_pos, alternative='greater')

    p_value = scipy.stats.combine_pvalues([p_value_pos, p_value_neg])[1]

    echo('n_pos_wins:', n_pos_wins, 'n_pos:', n_pos)
    echo('n_neg_wins:', n_neg_wins, 'n_neg:', n_neg)
    echo('pos p-value:', p_value_pos, 'neg p-value:', p_value_neg, 'product:', p_value_pos*p_value_neg, 'combined:', p_value)

    echo(n_pos_wins + n_neg_wins, 'wins out of', n_pos + n_neg, 'P(OR>1)=', p_pos, ', p-value=', p_value)
    echo('binomial p-value:', scipy.stats.binom_test(n_pos_wins + n_neg_wins, len(res), alternative='greater'))
    # echo('pos p-value:', p_value_pos)

    res = res[[c for c in cols if c in list(gwas_snps) or c == 'WIN' or c.startswith(test_label)]]

    return res



def sort_variant_stats_by_test(compute_variant_stats_result,
                               pvalue_label=None,
                               gene_names=None,
                               keep_columns=True,
                               ascending=True,
                               min_synonymous_pvalue=None):

    res = compute_variant_stats_result

    if pvalue_label is not None:
        test_type = pvalue_label.split('|')[0]
        or_label = test_type + '|OR'

        cols_to_keep = [c for c in list(res) if c.startswith(test_type) and c not in [pvalue_label, or_label]]

        if keep_columns:
            cols_to_keep += [c for c in list(res) if c not in cols_to_keep + [GENE_NAME, pvalue_label, or_label]]

        res = res.sort_values(pvalue_label, ascending=ascending)[[GENE_NAME, pvalue_label, or_label] + cols_to_keep]

    if gene_names is not None:
        res = res[res[GENE_NAME].isin(gene_names)]
        # res = res.sort_values(by=GENE_NAME)

    if min_synonymous_pvalue is not None:
        res = res[res['syn|chi2_pvalue'] >= min_synonymous_pvalue]

    return res


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


def qqplot(stats_object, pvalue_label, title_prefix=''):
    from matplotlib import pyplot as plt

    test_label = pvalue_label.split('|')[0]

    dataset_labels = sorted([col_label for col_label in list(stats_object) if col_label.startswith(test_label + '|n|')])
    if len(dataset_labels) == 0:
        dataset_labels = sorted([col_label for col_label in list(stats_object) if col_label.startswith(test_label + '|total')])
    # print(dataset_labels, test_label)

    stats_object = stats_object[~stats_object[pvalue_label].isnull()]

    # stats_object = stats_object[(stats_object[dataset_labels[0]] > 0) & (stats_object[dataset_labels[1]] > 0)]
    # stats_object = stats_object[(stats_object[dataset_labels[0]] > 0) & (stats_object[dataset_labels[1]] > 0)]

    pvalues = [p for p in stats_object[pvalue_label] if not np.isnan(p)]

    min_pvalue = min([p for p in pvalues if p > 0])

    observed = sorted([p if p > 0 else (min_pvalue / 10) for p in pvalues])

    expected = [(i + 1) / len(pvalues) for i in range(len(pvalues))]

    fig, ax = plt.subplots(ncols=4, figsize=(25, 5))
    fig.suptitle(title_prefix + ": " + pvalue_label + ' n=' + str(len(stats_object)))

    ax[0].plot(expected, observed, 'r.')
    ax[0].set_title("QQ plot")

    log_observed = sorted(-np.log10(observed), reverse=True)
    log_expected = -np.log10(expected)

    ax[1].plot(log_expected, log_observed, 'r.')

    fdr = statsmodels.stats.multitest.multipletests(observed, method='fdr_bh')[1]

    ax[1].set_title("n_fdr_5prc=%d, n_fdr_1prc=%d, 0|1/2|1ord=%d|%d|%d" %
                    (len([f for f in fdr if f <= 0.05]),
                     len([f for f in fdr if f <= 0.01]),
                     len([1 for o, e in zip(log_observed, log_expected) if o - e >= 0]),
                     len([1 for o, e in zip(log_observed, log_expected) if o - e >= 0.5]),
                     len([1 for o, e in zip(log_observed, log_expected) if o - e >= 1])
                     ))

    sorted_log_odds_ratios = sorted(np.log(stats_object[test_label + '|OR']))
    ax[2].hist(sorted_log_odds_ratios, bins=50, log=True)
    ax[2].set_title('median= %.4lf' % np.median(sorted_log_odds_ratios))

    ax[2].set_xlabel('Log (' + test_label + '|OR)')
    ax[2].set_ylabel('# genes')

    # bins = np.histogram(np.hstack((stats_object[dataset_labels[0]],
    #                                stats_object[dataset_labels[1]])), bins=50)[1]

    bins = list(range(25)) #+ [1000]

    # ax[3].set_title(dataset_labels[0].split('|')[-1])

    if len(dataset_labels) == 2:
        ax[3].hist(stats_object[dataset_labels[0]], bins, log=True, alpha=0.3, label=dataset_labels[0].split('|')[-1].replace('_colstripped', ''))
        ax[3].hist(stats_object[dataset_labels[1]], bins, log=True, alpha=0.3, label=dataset_labels[1].split('|')[-1].replace('_colstripped', ''))
        ax[3].set_xlabel('# variants')
        ax[3].set_ylabel('# genes')

        ax[3].legend()

    # ax[4].set_title(dataset_labels[1].split('|')[-1])
    # ax[4].hist(stats_object[dataset_labels[1]], bins=50, log=True)

    for ax_idx in [0, 1]:
        axisMax = (max(max(log_expected), max(log_observed)) + 0.5) if ax_idx == 1 else 1

        ax[ax_idx].set_xlim([0, axisMax])
        ax[ax_idx].set_xlabel("Expected P-values " + ("(-log10)" if ax_idx == 1 else ""))

        ax[ax_idx].set_ylim([0, axisMax])
        ax[ax_idx].set_ylabel("Observed P-values " + ("(-log10)" if ax_idx == 1 else ""))


        ax[ax_idx].plot([0, axisMax], [0, axisMax], 'b-')  # blue line

    plt.show()


def qqplot1(stats_object, pvalue_label, title_prefix='', log_only=False, exclude_ones=False, upper_limit=None):
    from matplotlib import pyplot as plt

    stats_object = stats_object[~stats_object[pvalue_label].isnull()]

    if exclude_ones:
        stats_object = stats_object[stats_object[pvalue_label] < 1]

    pvalues = [p for p in stats_object[pvalue_label] if not np.isnan(p)]

    # min_pvalue = min([p for p in pvalues if p > 0])

    observed = sorted([p if p > 1e-300 else 1e-300 for p in pvalues])

    expected = [(i + 1) / len(pvalues) for i in range(len(pvalues))]

    if log_only:
        fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
        ax.set_title(title_prefix + ": " + pvalue_label + ', n=' + str(len(stats_object)))

        log_observed = sorted(-np.log10(observed), reverse=True)
        log_expected = -np.log10(expected)

        ax.plot(log_expected, log_observed, 'r.')
        ax_max = max(max(log_observed), max(log_expected))
        ax.plot([0, ax_max], [0, ax_max], color='grey')

        # fdr = statsmodels.stats.multitest.multipletests(observed, method='fdr_bh')[1]
        ax.set_xlabel('-log10 Expected')
        ax.set_ylabel('-log10 Observed')

    else:
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        fig.suptitle(title_prefix + ": " + pvalue_label + ', n=' + str(len(stats_object)))

        ax[0].plot(expected, observed, 'r.')
        ax[0].set_title("QQ plot")
        ax[0].set_xlabel('Expected')
        ax[0].set_ylabel('Observed')

        ax[0].plot([0, 1], [0, 1], color='grey')

        log_observed = sorted(-np.log10(observed), reverse=True)
        log_expected = -np.log10(expected)

        ax[1].plot(log_expected, log_observed, 'r.')
        ax_max = max(max(log_observed), max(log_expected))
        ax[1].plot([0, ax_max], [0, ax_max], color='grey')

        # fdr = statsmodels.stats.multitest.multipletests(observed, method='fdr_bh')[1]
        ax[1].set_xlabel('Expected')
        ax[1].set_ylabel('Observed')

        ax[1].set_title('Log QQ plot')
    # ax[1].set_title("n_fdr_5prc=%d, n_fdr_1prc=%d, 0|1/2|1ord=%d|%d|%d" %
    #                 (len([f for f in fdr if f <= 0.05]),
    #                  len([f for f in fdr if f <= 0.01]),
    #                  len([1 for o, e in zip(log_observed, log_expected) if o - e >= 0]),
    #                  len([1 for o, e in zip(log_observed, log_expected) if o - e >= 0.5]),
    #                  len([1 for o, e in zip(log_observed, log_expected) if o - e >= 1])
    #                  ))

    if upper_limit is not None:
        ax[1].set_xlim((0, upper_limit))
        ax[1].set_ylim((0, upper_limit))

    plt.show()

# In order to define dataset specific variants: subtract TCGA from gnomad and gnomad from TCGA

def subtract_tcga_from_gnomad_and_vice_versa( tcga_meta,
                                              tcga_sparse_data,
                                              tcga_all,
                                              gnomad_exomes_non_cancer_all,
                                              nfe_gnomad_exomes_non_cancer_only):


    # subtract gnomad non_cancer_all from TCGA

    tcga_sample_info_all = tcga_meta.get_sample_info(sample_ids=list(tcga_sparse_data[GENOTYPE_SDF]))
    tcga_sample_info_europeans = tcga_sample_info_all[tcga_sample_info_all['race'] == 'white']

    tcga_all_AC1 = filter_variants(tcga_all, max_AC=1, chromosomes_to_exclude=['X', 'Y'])

    tcga_all_europeans_info, tcga_all_europeans_info_genotypes = subset_tcga_variants(tcga_sparse_data,
                                                                                 tcga_sample_info_europeans[
                                                                                     TCGA_SAMPLE_ID])

    tcga_all_AC1_europeans_info = pd.merge(tcga_all_AC1.info,
                                           tcga_all_europeans_info[[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT]],
                                           how='inner',
                                           on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])

    tcga_all_AC1_europeans_annotation = pd.merge(tcga_all_AC1.annotation,
                                                 tcga_all_AC1_europeans_info[[VARIANT_IDX]],
                                                 how='inner',
                                                 on=VARIANT_IDX)

    #     tcga_all_AC1_europeans = VcfData(info=tcga_all_AC1_europeans_info,
    #                                      annotation=tcga_all_AC1_europeans_annotation)

    # tcga_all_AC1_europeans_minus_gnomad_info = tcga_all_AC1_europeans_info[pd.merge(tcga_all_AC1_europeans_info,
    #                                                                                 gnomad_exomes_non_cancer_all.info,
    #                                                                                 on=(VCF_CHROM, VCF_POS, VCF_REF,
    #                                                                                     VCF_ALT),
    #                                                                                 how='left',
    #                                                                                 indicator=True)[
    #                                                                            '_merge'] == 'left_only']


    tcga_all_AC1_europeans_minus_gnomad_info = subtract_dataframes(tcga_all_AC1_europeans_info,
                                                                   gnomad_exomes_non_cancer_all.info,
                                                                   [VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])


    tcga_all_AC1_europeans_minus_gnomad_annotation = pd.merge(tcga_all_AC1_europeans_annotation,
                                                              tcga_all_AC1_europeans_minus_gnomad_info,
                                                              on=VARIANT_IDX,
                                                              how='inner')[list(tcga_all_AC1_europeans_annotation)]

    tcga_all_AC1_europeans_minus_gnomad = VcfData(info=tcga_all_AC1_europeans_minus_gnomad_info,
                                                  annotation=tcga_all_AC1_europeans_minus_gnomad_annotation)

    # subtract TCGA from gnomad non_cancer_only

    # nfe_gnomad_exomes_non_cancer_only_minus_tcga_info = nfe_gnomad_exomes_non_cancer_only.info[
    #     pd.merge(nfe_gnomad_exomes_non_cancer_only.info,
    #              tcga_all.info,
    #              on=(VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT),
    #              how='left',
    #              indicator=True)['_merge'] == 'left_only']


    nfe_gnomad_exomes_non_cancer_only_minus_tcga_info = subtract_dataframes(nfe_gnomad_exomes_non_cancer_only.info,
                                                                            tcga_all.info,
                                                                            [VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])


    nfe_gnomad_exomes_non_cancer_only_minus_tcga_annotation = pd.merge(nfe_gnomad_exomes_non_cancer_only.annotation,
                                                                       nfe_gnomad_exomes_non_cancer_only_minus_tcga_info,
                                                                       on=VARIANT_IDX,
                                                                       how='inner')[
        list(nfe_gnomad_exomes_non_cancer_only.annotation)]

    nfe_gnomad_exomes_non_cancer_only_minus_tcga = VcfData(info=nfe_gnomad_exomes_non_cancer_only_minus_tcga_info,
                                                           annotation=nfe_gnomad_exomes_non_cancer_only_minus_tcga_annotation)

    return tcga_all_AC1_europeans_minus_gnomad, nfe_gnomad_exomes_non_cancer_only_minus_tcga


def subtract_dataframes(from_dataframe, to_subtract, columns_for_join):
    res = pd.merge(from_dataframe,
                   to_subtract[columns_for_join],
                   on=columns_for_join,
                   how='left',
                   indicator=True)

    res = res[res['_merge'] == 'left_only']
    return res[[c for c in list(res) if c != '_merge']]


def read_GWAS_SNPs_finemapped_by_Jeremy(fname):
    dtype = {}
    for c in ['gwas_chrom', 'gwas_pos', 'coding_pos', 'peak_pos']:
        dtype[c] = np.int64
    for c in ['coding_p', 'coding_odds', 'peak_p', 'gwas_to_coding_r_squared']:
        dtype[c] = np.float64

    with open_file(fname, 'rt') as in_f:
        header = in_f.readline().strip().split()
        d = dict((k, []) for k in header)
        d['finemapped_to_coding'] = []

        finemapped_to_coding = True

        for l in in_f:
            l = l.strip()
            if l == '------any-----':
                finemapped_to_coding = False
            else:

                buf = l.split()
                for k, v in zip(header, buf):
                    d[k].append(dtype.get(k, lambda x: x)(v))

                d['finemapped_to_coding'].append(finemapped_to_coding)

    res = pd.DataFrame(d).rename(columns={'symbol': GENE_NAME})

    o = res.groupby(GENE_NAME).agg({'coding_odds': ['min', 'max']}).dropna()['coding_odds'].reset_index()

    o['max_abs_effect'] = o.apply(lambda r: max(r['max'], 1 / r['min']), axis=1)

    res = pd.merge(res, o[[GENE_NAME, 'max_abs_effect']], on=GENE_NAME, how='left').sort_values('max_abs_effect', ascending=False)

    return res


def test_ranks(ranks,
               n_all,
               return_empirical=False,
               multiple_testing_correction=True,
               verbose=False,
               return_all_pvalues=False,
               use_scipy_hypergeom=True,
               return_expected_value=False):

    import random
    from matplotlib import pyplot as plt

    ranks = sorted(ranks)
    n_ranks = len(ranks)

    def min_max_index(a, pick_first=False):

        m_i = 0
        m = a[m_i]

        for i in range(1, len(a)):
            if (a[i] < m) if pick_first else (a[i] <= m):
                m_i = i
                m = a[i]

        return m_i + 1, a[m_i]

    if return_empirical:
        N_PERM = 100000

        n_false_pos_array = [1] * n_ranks
        all_ranks = list(range(1, n_all + 1))

        for k in range(N_PERM):
            random_ranks = sorted(random.sample(all_ranks, n_ranks))

            for i in range(len(random_ranks)):
                if random_ranks[i] <= ranks[i]:
                    n_false_pos_array[i] += 1
        #                 else:
        #                     break

        pvalues = [fp / N_PERM for fp in n_false_pos_array]

    #         echo(ranks)
    #         echo(pvalues)

    else:

        if use_scipy_hypergeom:
            if verbose:
                echo('Using scipy hypergeom.sf method!')
            pvalues = [scipy.stats.hypergeom.sf(rho,
                                                n_all,
                                                r,
                                                n_ranks
                                                )
                       for rho, r in enumerate(ranks)]

        else:

            import hypergeom
            if verbose:
                echo('Using cythonized hypergeom.sf method!')
            pvalues = [hypergeom.sf(rho,
                                    n_all,
                                    r,
                                    n_ranks
                                    )
                        for rho, r in enumerate(ranks)]

    # echo(ranks, n_all, n_ranks, pvalues)
    non_zero_pvalues = [p for p in pvalues if p > 0]
    if len(non_zero_pvalues) > 0:
        min_p = min(non_zero_pvalues)
    else:
        min_p = 1e-300

    pvalues = [p if p > 0 else min_p / 10 for p in pvalues]

    if multiple_testing_correction:
        _, pvalues = statsmodels.stats.multitest.fdrcorrection(pvalues)

    best_n_variants, best_pval = min_max_index(pvalues)

    if verbose:
        # echo('p-values:', pvalues)
        plt.figure()
        plt.plot(-np.log10(pvalues))
        plt.ylabel('-log10 P-value')
        plt.xlabel('Rank')
        plt.show()

    if return_all_pvalues:
        if return_expected_value:
            return best_n_variants, best_pval, pvalues, [n_ranks * ranks[i] / n_all for i in range(n_ranks)]
        else:
            return best_n_variants, best_pval, pvalues

    else:
        if return_expected_value:
            return best_n_variants, best_pval, n_ranks * ranks[best_n_variants - 1] / n_all
        else:
            return best_n_variants, best_pval


def get_effective_number_of_dimensions(df):
    echo('[get_effective_number_of_dimensions]')
    # df = df.subtract(df.mean())

    df = pd.DataFrame(scipy.stats.zscore(df, axis=0), columns=df.columns)

    n_columns = len(list(df))

    sv = np.linalg.svd(df, compute_uv=False)

    eigv = np.square(sv) / (n_columns - 1)

    var_expl = eigv / np.sum(eigv)

    exp_var = 1 / n_columns

    var_enrichment = var_expl / exp_var

    effective_n_dims = sum([1 if v > 1 else v for v in var_enrichment])

    echo('Observed number of dimensions:', n_columns, ', effective:', effective_n_dims)

    return effective_n_dims


def annotate_variants_with_pAI_and_sAI(variant_data):

    echo('Annotating variants with primateAI and spliceAI scores')

    echo('Reading primateAI scores from pickle file')
    primateai_with_gene_names = pd.read_pickle(PRIMATEAI_PATH + '/PrimateAI_scores_v1.0.with_gene_names.pickle')

    echo('Reading spliceAI scores')

    spliceai_scores_all = pd.read_pickle(DATA_PATH + 'spliceai/spliceai_scores.hg19.min_score_0.pandas.pickle')
    spliceai_scores_all = spliceai_scores_all.sort_values(SPLICEAI_MAX_SCORE, ascending=False)
    spliceai_scores_all = spliceai_scores_all.drop_duplicates(
        [VCF_CHROM, VCF_POS, GENE_NAME, VCF_REF, VCF_ALT]).sort_values(
        [VCF_CHROM, VCF_POS, GENE_NAME, VCF_REF, VCF_ALT])

    echo('Positions with Splice AI scores:', len(spliceai_scores_all))

    variant_data.annotation = pd.merge(variant_data.full(),
                                       primateai_with_gene_names,
                                       on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, GENE_NAME],
                                       how='left',
                                       suffixes=['', '_pAI'])[[c for c in list(variant_data.annotation)
                                                               if c not in [PRIMATEAI_SCORE, 'UCSC gene']] +
                                                              [PRIMATEAI_SCORE, 'UCSC gene']]
    qs = {'min': 'min',
          'max': 'max',
          'q0.2': lambda x: x.quantile(q=0.2),
          'q0.5': lambda x: x.quantile(q=0.5),
          'q0.7': lambda x: x.quantile(q=0.75),
          'q0.9': lambda x: x.quantile(q=0.9)
          }

    res = primateai_with_gene_names[[GENE_NAME, PRIMATEAI_SCORE]].groupby(GENE_NAME).agg({PRIMATEAI_SCORE: qs})

    res.columns = ["_".join(x) for x in res.columns.ravel()]
    res = res.reset_index()

    variant_data.annotation = pd.merge(variant_data.annotation,
                                       res,
                                       on=GENE_NAME,
                                       how='left')

    variant_data.annotation = pd.merge(variant_data.full(),
                                       spliceai_scores_all,
                                       on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, GENE_NAME],
                                       how='left',
                                       suffixes=['', '_sAI'])[list(variant_data.annotation) + [SPLICEAI_MAX_SCORE]]

    return variant_data


def clump(snps,
          window=10000,
          consequence=None,
          pvalue_label='pval',
          max_p_value=None,
          clump_by_zscores=False,
          r2_table=None,
          max_r2=None,
          clump_by_pos_only=False):

    if consequence is not None and type(consequence) is not list:
        consequence = [consequence]

    result = dict((c, []) for c in list(snps))
    seen = {}

    if max_p_value is not None:
        snps = snps[snps[pvalue_label] <= max_p_value]

    if clump_by_zscores:
        sorted_snps = snps.copy()

        # echo('Sorting by:', '__sort_by' + pvalue_label)
        sorted_snps['__sort_by' + pvalue_label] = np.abs(sorted_snps[pvalue_label])
        sorted_snps = sorted_snps.sort_values('__sort_by' + pvalue_label, ascending=False)

    else:
        sorted_snps = snps.sort_values(pvalue_label)

    for row_idx, row in sorted_snps.iterrows():

        chrom = row[VCF_CHROM]
        pos = row[VCF_POS]

        if clump_by_pos_only:
            varid = ':'.join([chrom, str(pos)])
        else:
            varid = ':'.join([chrom, str(pos), row[VCF_REF], row[VCF_ALT]])

        if chrom not in seen:
            seen[chrom] = set()

        skip = False
        for (seen_varid, seen_pos) in seen[chrom]:
            if r2_table is None:
                if abs(seen_pos - pos) <= window:
                    skip = True
            else:
                r2 = r2_table.loc[varid][seen_varid]
                if r2 >= max_r2:
                    skip = True

        if consequence is not None:
            if row[VCF_CONSEQUENCE] not in consequence:
                skip = True

        if not skip:
            seen[chrom].add((varid, pos))
            for c in result:
                result[c].append(row[c])

    return pd.DataFrame(result)


def vcorrcoef_old(X, y):
    """Correlates rows of X with y"""
    np_array_type = type(np.array([]))
    if type(X) is not np_array_type:
        X = np.array(X)

    if type(y) is not np_array_type:
        y = np.array(y)

    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)

    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))

    r = r_num / r_den

    n = len(y)

    ab = n / 2 - 1

    prob = 2 * scipy.special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))

    return r, prob


def vcorrcoef(X, Y=None, axis='columns', return_single_df=False, index_label=None):
    """Correlates rows of X with rows of Y"""

    if Y is not None and len(Y.shape) == 1:
        Y = pd.DataFrame(Y)

    if axis == 'columns':
        X = X.T
        if Y is not None:
            Y = Y.T

    if Y is None:
        Y = X

    return_dataframe = False
    index = None
    columns = None

    if type(X) is type(pd.DataFrame({})) and type(Y) is type(pd.DataFrame({})):
        return_dataframe = True
        columns = X.index
        index = Y.index

    np_array_type = type(np.array([]))
    if type(X) is not np_array_type:
        X = np.array(X)

    if type(Y) is not np_array_type:
        Y = np.array(Y)

    if axis not in ['rows', 'columns']:
        echo('ERROR: axis must be one of "rows" or "columns"')
        return None, None

    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))

    r_num = np.matmul((X - Xm), (Y - Ym).T).T

    r_den = np.sqrt(np.matmul(np.reshape(np.sum((X - Xm) ** 2, axis=1), (X.shape[0], 1)),
                              np.reshape(np.sum((Y - Ym) ** 2, axis=1), (Y.shape[0], 1)).T)).T

    r = r_num / r_den

    # replace overflows
    r = np.where(r > 1, 1., r)
    r = np.where(r < -1, -1., r)

    n = Y.shape[1]

    ab = n / 2 - 1

    prob = 2 * scipy.special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))

    if return_dataframe:
        r = pd.DataFrame(r, columns=columns, index=index)
        prob = pd.DataFrame(prob, columns=columns, index=index)

    if return_single_df:
        final_result = pd.merge(r.rename(columns=dict((c, c + '/r') for c in columns)),
                                prob.rename(columns=dict((c, c + '/pvalue') for c in columns)),
                                left_index=True,
                                right_index=True)
        if index_label is not None:
            final_result = final_result.reset_index().rename(columns={'index': index_label})

        return final_result

    else:
        return r, prob

def multiproc_wrapper(func):
    def wrapped(batch_params):
        try:
            return func(*batch_params)

        except Exception as e:

            echo('Caught exception in output worker thread (pid: %d):' % os.getpid())
            echo(e)

            if hasattr(open_log, 'logfile'):
                traceback.print_exc(file=open_log.logfile)

            traceback.print_exc()

            # print

            raise e

    return wrapped


def get_variant_genotypes_from_vcf(fname, variants, sample_ids=None, replace_missing=False, replace_missing_with=None,
                                   append_chr_prefix=True,
                                   chrom_label=VCF_CHROM, pos_label=VCF_POS, ref_label=VCF_REF, alt_label=VCF_ALT,
                                   index=None):

    import tabix
    echo('Reading variants from:', fname)

    tb = tabix.open(fname)

    def get_single_variant_genotypes(tb, chrom, pos, ref, alt, sample_ids, replace_missing=False):

        records = list(tb.querys(f"{chrom}:{pos}-{pos}"))

        gt = [0] * len(sample_ids)

        has_missing = False

        if len(records) > 0:
            rec = [r for r in records if r[3] == ref and r[4] == alt]
            if len(rec) > 0:

                for s_idx, c_gt in enumerate(rec[0][9:]):
                    c_gt = c_gt.split(':')[0]

                    if c_gt == '0/0':
                        gt[s_idx] = 0
                    elif c_gt == '0/1':
                        gt[s_idx] = 1
                    elif c_gt == '1/1':
                        gt[s_idx] = 2
                    else:
                        gt[s_idx] = None
                        has_missing = True

        if replace_missing and has_missing:
            if replace_missing_with is None:
                repl = np.mean([g for g in gt if g is not None])
            else:
                repl = replace_missing_with
            for i in range(len(gt)):
                if gt[i] is None:
                    gt[i] = repl

        return pd.Series(dict((s_id, g) for s_id, g in zip(sample_ids, gt)))
    if index is None:
        index_array = variants.index
    else:
        index_array = variants[index]

    if sample_ids is None:
        echo('Extracting sample_ids from:', fname)
        with open_file(fname, 'rt') as in_f:
            for l in in_f:
                if l.startswith('#CHROM'):
                    buf = l.strip().split('\t')
                    sample_ids = buf[9:]
                    echo('sample_ids found:', len(sample_ids), sample_ids[:10], sample_ids[-10:])
                    break

    if sample_ids is None:
        echo('[ERROR]: no sample ids found!')
        return None

    res = pd.DataFrame([get_single_variant_genotypes(tb,
                                                     ('chr' if append_chr_prefix else '') + v[chrom_label],
                                                     v[pos_label],
                                                     v[ref_label],
                                                     v[alt_label],
                                                     sample_ids,
                                                     replace_missing
                                                     ) for _, v in variants.iterrows()],
                       index=index_array)

    echo(np.sum(np.sum(res, axis=1) == 0), 'missing variants')
    return res

def get_variant_sample_ids_from_vcf(fname, variants, sample_ids=None,
                                    append_chr_prefix=True,
                                    chrom_label=VCF_CHROM, pos_label=VCF_POS, ref_label=VCF_REF, alt_label=VCF_ALT):

    import tabix
    echo('Reading variants from:', fname)

    tb = tabix.open(fname)

    def get_single_variant_sample_ids(tb, chrom, pos, ref, alt, sample_ids):

        records = list(tb.querys(f"{chrom}:{pos}-{pos}"))

        samples_with_variant = []

        homozygotes = []
        heterozygotes = []

        if len(records) > 0:
            rec = [r for r in records if r[3] == ref and r[4] == alt]
            if len(rec) > 0:

                for s_idx, c_gt in enumerate(rec[0][9:]):
                    c_gt = c_gt.split(':')[0]
                    if c_gt in ['0/1', '1/1']:
                        samples_with_variant.append(sample_ids[s_idx])
                        if c_gt == '0/1':
                            heterozygotes.append(sample_ids[s_idx])
                        else:
                            homozygotes.append(sample_ids[s_idx])

        return ','.join(samples_with_variant), ','.join(homozygotes), ','.join(heterozygotes)

    if sample_ids is None:
        echo('Extracting sample_ids from:', fname)
        with open_file(fname, 'rt') as in_f:
            for l in in_f:
                if l.startswith('#CHROM'):
                    buf = l.strip().split('\t')
                    sample_ids = buf[9:]
                    echo('sample_ids found:', len(sample_ids), sample_ids[:10], sample_ids[-10:])
                    break

    if sample_ids is None:
        echo('[ERROR]: no sample ids found!')
        return None

    res = variants.copy()

    all_samples = []
    homozygotes = []
    heterozygotes = []

    for v_no, (_, v) in enumerate(variants.iterrows()):
        if v_no % 1000 == 0:
            echo(v_no, 'variants processed')
        v_all_samples, v_homozygotes, v_heterozygotes = get_single_variant_sample_ids(tb,
                                                                                        ('chr' if append_chr_prefix else '') + v[chrom_label],
                                                                                        v[pos_label],
                                                                                        v[ref_label],
                                                                                        v[alt_label],
                                                                                        sample_ids)

        all_samples.append(v_all_samples)
        homozygotes.append(v_homozygotes)
        heterozygotes.append(v_heterozygotes)


    res['all_samples'] = all_samples
    res['homozygotes'] = homozygotes
    res['heterozygotes'] = heterozygotes


    echo(np.sum(np.sum(res, axis=1) == 0), 'missing variants')
    return res

# gt = get_variant_genotypes(tb, annotated_gtex_variants.info.head(), sample_ids=sample_ids)
# gt = get_variant_genotypes(gtex_fname, variants, sample_ids=sample_ids, replace_missing=True)
# echo(gt.shape)
# gt.head()


def get_prs_params(prs_fname):
    echo('Reading PRS from:', prs_fname)
    with open(prs_fname, 'rb') as in_f:
        prs_model = pickle.load(open(prs_fname, 'rb'))

    CONST_TERM = '__CONST__'

    prs_params = prs_model.params.reset_index().rename(columns={0: 'PRS_beta', 'index': 'varid'})
    const_term = prs_params[prs_params['varid'] == CONST_TERM].iloc[0]['PRS_beta']

    echo('const_term:', const_term)

    prs_params = prs_params[prs_params['varid'] != CONST_TERM].copy()
    echo(len(prs_params), 'PRS variants')

    for i, l in enumerate([VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT]):

        prs_params[l] = prs_params['varid'].apply(lambda x: x.split(':')[i])

        if l == VCF_POS:
            prs_params[l] = prs_params[l].astype(int)
        else:
            prs_params[l] = prs_params[l].astype(str)

    return prs_params, const_term


def get_residuals(data, response_labels, covariate_labels, interactions=False, make_sample_ids_index=True, verbose=True):
    import sklearn
    import sklearn.linear_model

    if type(response_labels) is not list:
        response_labels = [response_labels]

    if type(covariate_labels) is not list:
        covariate_labels = [covariate_labels]

    interaction_labels = []

    if interactions:
        if verbose:
            echo('Including interaction terms')
        data = data.copy()

        for c1_idx in range(len(covariate_labels)):
            c1 = covariate_labels[c1_idx]
            for c2_idx in range(c1_idx, len(covariate_labels)):
                c2 = covariate_labels[c2_idx]
                int_label = c1 + ' X ' + c2
                data[int_label] = data[c1] * data[c2]
                interaction_labels.append(int_label)

    covariate_labels = list(covariate_labels) + interaction_labels

    if verbose:
        echo('Responses:', response_labels)
        echo('Covariates:', covariate_labels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        skm = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(data[covariate_labels], data[response_labels])

        predictions = skm.predict(data[covariate_labels])
        resid = data[response_labels] - predictions
        res = pd.DataFrame(resid, columns=response_labels)

        if SAMPLE_ID in list(data):
            res[SAMPLE_ID] = data[SAMPLE_ID]

    if SAMPLE_ID in list(res) and make_sample_ids_index:
        res = res.set_index(SAMPLE_ID)

    return res


def max_r2(af1, af2):
    # as per: https://genepi.qimr.edu.au/contents/p/staff/wray_r2_2005.pdf

    max_r2 = (af1 * (1 - af2)) / ((1 - af1) * af2)

    max_r2 = min(max_r2, 1 / max_r2)

    return max_r2


def batchify(array, batch_size):
    for batch_idx in range(0, len(array), batch_size):
        yield array[batch_idx: batch_idx + batch_size]


def list_hist(a):
    d = {}
    for el in a:
        if el not in d:
            d[el] = 0
        d[el] += 1

    total = sum(d.values())

    res = pd.DataFrame({'element': sorted(d),
                        'count': [d[el] for el in sorted(d)]}).sort_values('count', ascending=False)
    res['freq'] = res['count'] / total

    return res


def pandas_memory_footprint():
    gg = globals()
    for k in gg:
        if type(gg[k]) == pd.core.frame.DataFrame:
            mf = gg[k].memory_usage(deep=True)
            echo(k, np.sum(mf) / 10 ** 9, '\n', mf)


def reorder_rare_variants_result(rv_res,
                                 vtype='del',
                                 exclude_chromosomes=['X', 'Y'],
                                 filter_genes=None,
                                 sort_label='ALL/%s/carrier/pvalue/fdr_corr',
                                 verbose=False,
                                 recompute_fdr=True):

    if verbose:
        echo('Reordering by', vtype, ', excluding chromosomes:', exclude_chromosomes)

    if filter_genes is not None:
        rv_res = rv_res[rv_res[GENE_NAME].isin(filter_genes)].copy()

    rv_res = rv_res.sort_values(sort_label % vtype)

    if exclude_chromosomes is not None:
        rv_res = rv_res[~rv_res[VCF_CHROM].isin(exclude_chromosomes)].copy()

    rv_res['rank/' + vtype] = list(range(1, len(rv_res) + 1))

    if recompute_fdr:
        _r = rv_res[[GENE_NAME, sort_label % vtype]].dropna()
        _r[vtype + '/fdr'] = statsmodels.stats.multitest.multipletests(_r[sort_label % vtype], method='fdr_bh')[1]

        del _r[sort_label % vtype]

        rv_res = pd.merge(_r, rv_res, on=GENE_NAME, how='right')

    return rv_res[[GENE_NAME, 'rank/' + vtype] + ([vtype + '/fdr'] if recompute_fdr else []) +
                  sorted([c for c in list(rv_res) if vtype in c and c not in ['rank/' + vtype, vtype + '/fdr']],
                         key=lambda k: 1 if f'{vtype}/carrier/pvalue' in k else 2 if 'beta' in k else 3) +
                  [c for c in list(rv_res) if vtype not in c and c != GENE_NAME]]


def read_rv_results(fname,
                    gencode_fname=ROOT_PFIZIEV_PATH + '/rare_variants/data/gencode/gencode.v39.annotation.gene_info.csv',
                    verbose=False,
                    recompute_fdr=True):

    if fname.endswith('.pickle'):
        rv_res = pd.read_pickle(fname)
    else:
        rv_res = pd.read_csv(fname, sep='\t')

    if verbose:
        echo('rv_res:', rv_res.shape)
    if gencode_fname is not None:
        if verbose:
            echo('Keeping genes in gencode:', gencode_fname)

        gencode_genes = set(pd.read_csv(gencode_fname, sep='\t')[GENE_NAME])
        rv_res = rv_res[rv_res[GENE_NAME].isin(gencode_genes)].copy()

    if recompute_fdr:
        vtypes = set(k.split('/')[1] for k in rv_res if k.startswith('ALL/'))
        if verbose:
            echo('vtypes:', vtypes)
        for vt in sorted(vtypes, key=lambda k: 10 if k == 'del' else 0):
            rv_res = reorder_rare_variants_result(rv_res, vt, verbose=verbose)
    if verbose:
        echo('after filtering:', rv_res.shape)
    return rv_res

def dump_to_tmp_file(data, output_dir=None):
    if output_dir is None:
        output_dir = os.getcwd()

    fd, fname = tempfile.mkstemp(suffix='.' + str(os.getpid()) + '.pickle', dir=output_dir)
    echo('Dumping data into temp file:', fname)

    with open(fd, 'wb') as outf:
        pickle.dump(data, outf, 4)

    return fname


def load_from_tmp_file(fname):

    echo('Loading data from temp file:', fname)

    with open(fname, 'rb') as in_f:
        data = pickle.load(in_f)

    return data


def read_sqlite(fname):
    conn = sqlite3.connect(fname)
    tables = pd.read_sql_query(f"SELECT * from sqlite_master where type = 'table'", conn)

    table_names = list(tables['name'])

    if len(table_names) != 1:
        echo('[WARNING] More than one table in the sqlite database:', table_names)

    res = pd.read_sql('select * from "' + table_names[0] + '"', conn)

    conn.close()

    return res
