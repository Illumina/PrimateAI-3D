

from dataclasses import dataclass
import logging

import numpy
from regressor import linregress
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, ElasticNetCV, RidgeCV

from rvPRS.rare.consequence import group_by_consequence
from rvPRS.rare.filter_variants import get_rare_variants, filter_by_ac

@dataclass
class TempVar:
    chrom: str
    pos: int
    ref: str
    alt: str
    af: float
    pathogenicity: float

def get_max_values_for_carriers(variants):
    ''' find the max allele frequency and missense pathogenicity score for carriers of a gene
    '''
    samples = {}
    for var in sorted(variants, key=lambda x: x.af, reverse=True):
        var_af = var.af
        if var_af > 0.5:
            var_af = 1 - var_af
        tmpvar = TempVar(var.chrom, var.pos, var.ref, var.alt, var_af, var.primateai)
        for sample in var.all_samples:
            if sample not in samples:
                samples[sample] = tmpvar
            if var.primateai > samples[sample].pathogenicity:
                samples[sample] = tmpvar
    return samples

def af_arrays(table):
    ''' convert AF data into numpy arrays for regression/plotting
    
    Args:
        table: list of entries per person for samples with a rare variant for a
            gene. Each person has AF, and phenotype (adjusted for 
            missense pathogenicity).
    
    Returns:
        two numpy arrays, one for x_values (log10(AF) and score, 
        and one for y_values (phenotype values).
    '''
    cols = list(table[0].keys())
    y_vals = numpy.array([x['pheno'] for x in table])
    
    cols = [x for x in cols if x not in set(['pheno', 'sample_id'])]  # drop pheno column, now in y_vals
    x_vals = numpy.array([[x[k] for k in cols] for x in table])
    
    log10_cols = set(['af'])
    for col in log10_cols:
        if col in cols:
            idx = cols.index(col)
            x_vals[:, idx] = numpy.log10(x_vals[:, idx])  # log10 transform cols
    
    # drop columns with no
    keep = []
    for idx in range(x_vals.shape[1]):
        unique = numpy.unique(x_vals[:, idx])
        keep.append(len(unique) > 1)
    x_vals = x_vals[:, keep].copy()
    cols = [x for x, k in zip(cols, keep) if k]
    
    return x_vals, y_vals, cols, log10_cols

def get_alpha_plus_1se(fit):
    ''' get a more conservative estimate for the optimal alpha (lambda) 
    '''
    # get the average and stderr of the MSE across the folds for each alpha
    fit.mse_path_avg_ = fit.mse_path_.mean(axis=1)
    fit.mse_path_se_ = fit.mse_path_.std(axis=1) / fit.mse_path_.shape[1]
    # find the index to the best performer
    idx = fit.mse_path_avg_.argmin()
    mse_1se = fit.mse_path_avg_[idx] + fit.mse_path_se_[idx]
    subset = fit.mse_path_avg_ <= mse_1se
    return max(fit.alphas_[subset])

def fit_model(x_vals, y_vals, name='lasso', cv=True, use_1se=True):
    ''' fit AF versus phenotype
    
    Args:
        model: type of linear model (linear, lasso, ridge, elasticnet)
        cv: whether to use cross-validation to optimize for non-linreg models
        use_1se: whether to use more conservative alpha for cross validation 
                 models, that is, the alpha + 1 standard error (from the k-fold 
                 cross validations)
    
    Returns:
        fitted model with beta parameters
    '''
    models = {False: 
                {'linreg': linregress, 'lasso': Lasso, 'ridge': Ridge,  'elasticnet': ElasticNet},
            True:
                {'linreg': linregress, 'lasso': LassoCV, 'ridge': RidgeCV,  'elasticnet': ElasticNetCV},
            }
    model = models[cv][name]
    x_vals = numpy.concatenate([x_vals, numpy.ones(len(x_vals), dtype=numpy.float32)[:, None]], axis=1)
    if name == 'linreg':
        fit = model(x_vals, y_vals, has_intercept=True)
    else:
        if cv:
            x_vals = x_vals.astype(numpy.float64)
            fit = model(fit_intercept=True).fit(x_vals, y_vals)
        else:
            fit = model(alpha=0.1, fit_intercept=True).fit(x_vals, y_vals)
    
    if cv and hasattr(fit, 'alpha_'):
        alpha_1se = get_alpha_plus_1se(fit)
        logging.info(f'optimal alpha: {fit.alpha_} vs 1se estimate: {alpha_1se}')
        if use_1se:
            model = models[False][name]
            fit = model(alpha=alpha_1se, fit_intercept=True).fit(x_vals, y_vals)
    
    if name in ['lasso', 'lassocv', 'ridge', 'ridgecv', 'elasticnet', 'elasticnetcv']:
        fit.beta = fit.coef_
    
    return fit

def get_trait_data(phenotype, gene, conn, score_col, var_type, exome_samples):
    ''' get trait data for a single gene
    
    Args:
        phenotype: dict of sample ID (as int) to phenotype value. This is just 
            for the subset in the training data, and with a non-null phenotype
        gene: rare variant gene result for a single consequence e.g. 'del'
        exome_db_path: path to exome variant database
    
    Returns:
        list of relevant data for variant/samples for the trait/gene
    '''
    symbol = gene['symbol']
    variants = get_rare_variants(conn, symbol, score_type=score_col, exome_samples=exome_samples)
    variants = filter_by_ac(variants, gene['ac_threshold'])
    by_cq = group_by_consequence(variants)
    
    above_threshold = [x for x in by_cq[var_type] if x.primateai >= gene.pathogenicity_threshold]
    
    table = []
    for sample, var in get_max_values_for_carriers(above_threshold).items():
        if sample not in phenotype:
            continue
        pheno_val = phenotype[sample]
        table.append({
            'sample_id': sample,
            'af': var.af,
            'pathogenicity': var.pathogenicity,
            'pheno': pheno_val,
            })
    return table

class MockFit:
    def __init__(self, labels):
        self.beta = [0] * len(labels)
        self.pvalue = [1] * len(labels)

def fit_gene(phenotype, gene, conn, score_col, var_type, exome_samples):
    ''' fits a model for allele frequency and missense pathogenicity for a trait
    
    Args:
        phenotype: dict of sample ID to phenotype value for samples to include in model
        genes: iterable of genes significant for rare variant test. Each gene has
            symbol, ac_threshold, pathogenicity_threshold attributes
        conn: connection to sqlite3 database with exome data
    
    Returns:
        regression model with intercept, beta and pvalue attributes
    '''
    table = get_trait_data(phenotype, gene, conn, score_col, var_type, exome_samples)
    if len(table) > 10:
        x_vals, y_vals, labels, log10_cols = af_arrays(table)
        fit = fit_model(x_vals, y_vals, name='linreg')
    else:
        labels = ['af', 'pathogenicity']
        log10_cols = ['af']
        fit = MockFit(labels + ['intercept'])
    
    labels += ['intercept']
    
    fit.labels = labels
    fit.log10_cols = log10_cols
    
    return fit

def fit_genes(phenotype, genes, conn, score_col, var_type, exome_samples, model='lasso'):
    ''' fits a model for allele frequency and missense pathogenicity for a trait
    
    Args:
        phenotype: dict of sample ID to phenotype value for samples to include in model
        genes: iterable of genes significant for rare variant test. Each gene has
            symbol, ac_threshold, pathogenicity_threshold attributes
        conn: connection to sqlite3 database with exome data
    
    Returns:
        regression model with intercept, beta and pvalue attributes
    '''
    # score_col = score_type
    # var_type = cq_type
    
    genes = list(genes)
    x_arr = numpy.zeros((len(phenotype), 2 * len(genes)), dtype=numpy.float32)
    sample_ids = {k: i for i, k in enumerate(sorted(phenotype))}
    y_arr = numpy.array([phenotype[x] for x in sample_ids], dtype=numpy.float32)
    i = 0
    labels = []
    inverse_sqrt_cols = []
    for gene in genes:
        table = get_trait_data(phenotype, gene, conn, score_col, var_type, exome_samples)
        for item in table:
            idx = sample_ids[item['sample_id']]
            x_arr[idx, i] = item['af']
            x_arr[idx, i+1] = item['pathogenicity']
        labels += [(gene.symbol, 'af'), (gene.symbol, 'pathogenicity')]
        inverse_sqrt_cols += [(gene.symbol, 'af')]
        i += 2
    
    # inverse sqrt the allele frequencies, but reassign missing AFs to 0, to 
    # account samples without rare variants in a gene
    for col in inverse_sqrt_cols:
        idx = labels.index(col)
        x_arr[:, idx] = 1 / (x_arr[:, idx] ** 0.5)
    x_arr[numpy.isinf(x_arr)] = 0
    
    labels += ['intercept']
    
    fit = fit_model(x_arr, y_arr, model)
    fit.labels = labels
    fit.inverse_sqrt_cols = inverse_sqrt_cols
    return fit

def freq_model_predict(fit, var):
    ''' predict per-variant effect size given freq, pathogenicity score etc model
    
    Get an adjusted effect size for a variant, given the variant allele frequency, 
    missense pathogenicity score and local mutation rate.
    
    Args:
        fit: regression object for multiple regression, with numpy arrays for
            betas, labels, log10_cols (whether the value needs to be log10-scaled)
        var: Variant object, with chrom, pos, ref, alt attributes
    
    Returns:
        an adjusted effect size for the variant
    '''
    attributes = (['af', 'pathogenicity'])
    
    effect = 0
    for idx, label in enumerate(fit.labels):
        if label in attributes:
            value = getattr(var, label)
        elif label == 'intercept':
            value = 1
        if label in fit.log10_cols:
            value = numpy.log10(value)
        effect += fit.beta[idx] * value
    
    return float(effect)
