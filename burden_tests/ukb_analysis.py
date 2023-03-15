from jutils import *
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import bgen
from bgen.reader import BgenFile
import pandas as pd
import re
import sklearn
import sklearn.linear_model
import statsmodels.stats.multitest
from constants import *
import os

DNANEXUS_PATH = '/mnt/project/'
UKB_COMMON_VARIANTS_ON_DNANEXUS_PATH = DNANEXUS_PATH + '/Bulk/Imputation/UKB imputation from genotype/'
UKB_COMMON_VARIANTS_ON_DNANEXUS_MAPPING = dict((chrom, (UKB_COMMON_VARIANTS_ON_DNANEXUS_PATH + '/' + f'ukb22828_c{chrom}_b0_v3.bgen',
                                                        UKB_COMMON_VARIANTS_ON_DNANEXUS_PATH + '/' + f'ukb22828_c{chrom}_b0_v3.sample',
                                                        UKB_COMMON_VARIANTS_ON_DNANEXUS_PATH + '/' + f'ukb22828_c{chrom}_b0_v3.mfi.txt'
                                                   )) for chrom in list(map(str, range(1, 23))) + ['X'])


CATEGORICAL_PHENOTYPE = 'CATEGORICAL_PHENOTYPE'
DATE_PHENOTYPE = 'DATE_PHENOTYPE'
HOUR_PHENOTYPE = 'HOUR_PHENOTYPE'
UKB_GPC_PREFIX = 'gPC'

med_array_range = range(48)

MEDICATIONS_FIRST_VISIT = ['20003-0.' + str(i) for i in med_array_range]
MEDICATIONS_SECOND_VISIT = ['20003-1.' + str(i) for i in med_array_range]

def combine_phenotypes_by_prefix(ukb_phenotypes,
                                 prefixes,
                                 phenotype_name=None,
                                 vcfdata=None,
                                 generate_random_phenotypes=False,
                                 is_quantitative=False,
                                 binarize_quantiles=None,
                                 covariates_to_regress=None,
                                 inverse_rank_normal_tranform=False):

    import statsmodels.api as sm
    # collect info for columns needed to compute phenotype values
    col_names = []
    if type(prefixes) is not list:
        prefixes=[prefixes]

    if is_quantitative:
        col_names = prefixes
        if len(col_names) != 1:
            echo('ERROR: too many column names:', col_names)
            return None
    else:
        for c in list(ukb_phenotypes):

            for prefix in prefixes:
                if c.startswith(prefix):
                    col_names.append(c)

    result = ukb_phenotypes[[SAMPLE_ID] + col_names].copy()

    if phenotype_name is None:
        phenotype_name = prefixes[0] + '_combined'

    if not is_quantitative:
        result[phenotype_name] = result[col_names].any(axis='columns')
    else:
        result[phenotype_name] = result[col_names]

    result = result[~result[phenotype_name].isnull()]

    # regress out covariates from phenotype, if covariates are provided
    if covariates_to_regress is not None:
        cov_values = {}

        for cov in covariates_to_regress:

            if type(cov) is tuple:
                if len(cov) == 3:
                    cov_col_name, cov_func, cov_label = cov
                else:
                    cov_col_name, cov_label = cov
                    cov_func = lambda v: v
            else:
                cov_col_name = cov_label = cov
                cov_func = lambda v: v

            if type(cov_func) is str:
                if cov_func == CATEGORICAL_PHENOTYPE:

                    dummies = pd.get_dummies(ukb_phenotypes[cov_col_name], prefix=cov_label)
                    for d_label in list(dummies):
                        cov_values[d_label] = dummies[d_label]

                elif cov_func == DATE_PHENOTYPE:
                    extract_date = lambda x: x.split('T')[0] if not pd.isnull(x) else x

                    dummies = pd.get_dummies(ukb_phenotypes[cov_col_name].apply(extract_date), prefix=cov_label)
                    for d_label in list(dummies):
                        cov_values[d_label] = dummies[d_label]

                elif cov_func == HOUR_PHENOTYPE:

                    def extract_hour(x):

                        if pd.isnull(x):
                            return None

                        buf = x.split('T')
                        if len(buf) == 2:
                            return buf[1].split(':')[0]
                        else:
                            return None

                    dummies = pd.get_dummies(ukb_phenotypes[cov_col_name].apply(extract_hour), prefix=cov_label)
                    for d_label in list(dummies):
                        cov_values[d_label] = dummies[d_label]

            else:
                cov_values[cov_label] = list(map(float, cov_func(ukb_phenotypes[cov_col_name])))

        CONST_LABEL = '__CONST__'

        cov_values[CONST_LABEL] = [1] * len(ukb_phenotypes)

        predictors = pd.DataFrame(cov_values)
        predictors = predictors.dropna()

        predictors = pd.merge(predictors, result, left_index=True, right_index=True)[list(predictors)]
        result = pd.merge(predictors, result, left_index=True, right_index=True)[list(result)]

        # exclude covariates that have too few different values
        cov_to_exclude = []
        for cov_label in list(predictors):

            if cov_label == CONST_LABEL:
                continue

            all_distinct_values = predictors[cov_label].value_counts()

            if (len(all_distinct_values) == 1 or
                (len(all_distinct_values) == 2 and any(val_count < 10 for val_count in all_distinct_values))):
                cov_to_exclude.append(cov_label)

        echo('Excluding covariates due to insufficient variability:', cov_to_exclude, logfile_only=True)
        predictors = predictors.drop(cov_to_exclude, axis=1)

        echo('Fitting regression with n=', len(result), 'samples')
        regression_model = sm.OLS(result[phenotype_name], predictors)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            regression_result = regression_model.fit()

        echo(regression_result.summary(), logfile_only=True)

        result[phenotype_name] = regression_result.resid

        if is_quantitative and inverse_rank_normal_tranform:
            echo('Inverse rank-normalizing residuals')
            result[phenotype_name] = rank_INT(result[phenotype_name])

    if generate_random_phenotypes:

        echo('Generating random binary phenotypes with # cases:', int(np.sum(result[phenotype_name])))
        result = result.apply(np.random.permutation)

    if vcfdata is not None:
        result = pd.merge(pd.DataFrame({SAMPLE_ID: list(vcfdata.sparse_data)}), result)

    if binarize_quantiles is not None:

        (q_lo, q_hi) = binarize_quantiles

        sorted_phenotypes = sorted(result[phenotype_name])

        v_lo = sorted_phenotypes[int(q_lo * len(sorted_phenotypes))]
        v_hi = sorted_phenotypes[int(q_hi * len(sorted_phenotypes))]

        echo('Binarizing based thresholds:', v_lo, v_hi)

        result = result[(result[phenotype_name] <= v_lo) | (result[phenotype_name] >= v_hi)]
        result[phenotype_name] = (result[phenotype_name] >= v_hi)

        echo('cases:', np.sum(result[phenotype_name]), 'controls:', len(result) - np.sum(result[phenotype_name]))

    return result[[SAMPLE_ID, phenotype_name]].sort_values(SAMPLE_ID)


CONST_LABEL = '__CONST__'


def get_covariates(ukb_phenotypes, covariates, bias=True):

    cov_values = {SAMPLE_ID: ukb_phenotypes[SAMPLE_ID]}

    for cov in covariates:

        if type(cov) is tuple:
            if len(cov) == 3:
                cov_col_name, cov_func, cov_label = cov
            else:
                cov_col_name, cov_label = cov
                cov_func = lambda v: v
        else:
            cov_col_name = cov_label = cov
            cov_func = lambda v: v

        if type(cov_func) is str:
            if cov_func == CATEGORICAL_PHENOTYPE:

                dummies = pd.get_dummies(ukb_phenotypes[cov_col_name], prefix=cov_label)
                for d_label in list(dummies):
                    cov_values[d_label] = dummies[d_label]

            elif cov_func == DATE_PHENOTYPE:

                extract_date = lambda x: x.split('T')[0] if not pd.isnull(x) else x

                dummies = pd.get_dummies(ukb_phenotypes[cov_col_name].apply(extract_date), prefix=cov_label)
                for d_label in list(dummies):
                    cov_values[d_label] = dummies[d_label]

            elif cov_func == HOUR_PHENOTYPE:

                def extract_hour(x):

                    if pd.isnull(x):
                        return np.nan

                    buf = x.split('T')
                    if len(buf) == 2:
                        return int(buf[1].split(':')[0])
                    else:
                        return np.nan

                hours = ukb_phenotypes[cov_col_name].apply(extract_hour)

                if len(hours.dropna().unique()) > 0:
                    cov_values[cov_label + '_sin'] = np.sin(hours)
                    cov_values[cov_label + '_cos'] = np.cos(hours)
                else:
                    echo('Warning: Could not extract hours for:', cov_col_name, ', ', cov_label)

        else:
            cov_values[cov_label] = list(map(float, cov_func(ukb_phenotypes[cov_col_name])))

    if bias:
        cov_values[CONST_LABEL] = [1] * len(ukb_phenotypes)

    return pd.DataFrame(cov_values)


def save_regression_results(y_label, model_type, regression_result, out_fname_prefix):

    df = pd.DataFrame({'beta': regression_result.params,
                       'pvalue': regression_result.pvalues,
                       'stderr': regression_result.bse,
                       'tstat': regression_result.tvalues}).reset_index().rename(columns={'index': 'predictor'})

    df['y_label'] = y_label
    df['model_type'] = model_type
    df['rsquared'] = regression_result.rsquared
    df['rsquared_adj'] = regression_result.rsquared_adj

    df['AIC'] = regression_result.aic
    df['BIC'] = regression_result.bic

    df['mse_model'] = regression_result.mse_model
    df['mse_resid'] = regression_result.mse_resid
    df['mse_total'] = regression_result.mse_total
    df['df_model'] = regression_result.df_model
    df['df_resid'] = regression_result.df_resid
    df['n_observations'] = regression_result.nobs
    df['log_likelihood'] = regression_result.llf

    echo('Saving regression results:', out_fname_prefix)
    df.to_pickle(out_fname_prefix + '.pickle')
    df.to_csv(out_fname_prefix + '.csv.gz', sep='\t', index=False)


def regress_out_covariates_from_ukb_phenotype(ukb_phenotypes,
                                              phenotype_code,
                                              phenotype_name,
                                              covariates,
                                              sex,
                                              bias=True,
                                              log_dir=None,
                                              outlier_cutoff=None,
                                              irnt_transform=False,
                                              additional_covariates=None,
                                              fig_label='',
                                              phenotype_is_binary=False,
                                              impute_missing_covatiates=False,
                                              return_covariates=False):

    import statsmodels.api as sm

    echo('Imputing missing covariate values:', impute_missing_covatiates)

    phenotype = ukb_phenotypes[[SAMPLE_ID, phenotype_code]].dropna(how='any').copy()

    echo(phenotype_name, '(', phenotype_code, ')', ' was measured in:', len(phenotype), 'samples')

    # remove outliers
    if outlier_cutoff is not None:
        before = len(phenotype)
        phenotype = phenotype[np.abs(scipy.stats.zscore(phenotype[phenotype_code])) <= outlier_cutoff]
        echo('Removed', before - len(phenotype), 'outliers father than', outlier_cutoff, 'standard deviations')

    phenotype[phenotype_name + '.original'] = phenotype[phenotype_code]

    predictors = get_covariates(ukb_phenotypes, covariates, bias=bias)

    predictor_labels = [c for c in list(predictors) if c != SAMPLE_ID]

    predictor_means = {}
    for p in list(predictor_labels):
        predictor_means[p] = predictors[p].mean(skipna=True)
        echo('Covariate=', p, ', non-missing=', len(predictors) - np.sum(predictors[p].isnull()), ', mean=', predictor_means[p], logfile_only=True)

    if additional_covariates is not None:

        predictors = pd.merge(predictors, additional_covariates, on=SAMPLE_ID)
        predictor_labels += [c for c in list(additional_covariates) if c != SAMPLE_ID]

    predictor_labels = sorted(set(predictor_labels))

    echo('n_covariates=', len(predictor_labels))
    echo('Total rows:', len(predictors))

    if not impute_missing_covatiates:
        predictors = predictors.dropna()
        echo('Rows after removing NaN values:', len(predictors))

    data = pd.merge(phenotype, predictors, on=SAMPLE_ID)

    data = data[[SAMPLE_ID, phenotype_code] + predictor_labels]
    # exclude covariates that have too few different values
    cov_to_exclude = []
    for cov_label in predictor_labels:

        if cov_label == CONST_LABEL:
            continue

        all_distinct_values = data[cov_label].dropna().value_counts()

        if (len(all_distinct_values) == 1 or
            (len(all_distinct_values) == 2 and any(val_count < 10 for val_count in all_distinct_values))):
            cov_to_exclude.append(cov_label)

    echo('Excluding covariates due to insufficient variability:', cov_to_exclude, logfile_only=True)
    data = data.drop(cov_to_exclude, axis=1)

    predictor_labels = [c for c in predictor_labels if c in list(data)]

    if impute_missing_covatiates:

        echo('Replacing missing values')

        is_null = data[predictor_labels].isnull()

        non_missing = data[~is_null.any(axis=1)]

        echo('Samples with no missing values:', len(non_missing))

        missing_columns = sorted([c for c in predictor_labels if is_null[c].any()])
        echo('Columns with missing values:', missing_columns)

        for col in missing_columns:
            echo('Filling missing values in:', col, ':', np.sum(is_null[col]), 'with mean=', predictor_means[col])
            data[col] = data[col].fillna(predictor_means[col])

    echo('Fitting regression with n=', len(data), 'samples')
    if phenotype_is_binary:
        echo('Fitting a logit model')
        regr_model_type = sm.Logit
        model_type = 'Logit'

    else:
        echo('Fitting an OLS model')
        regr_model_type = sm.OLS
        model_type = 'OLS'

    regression_model = regr_model_type(data[phenotype_code].astype(float), data[predictor_labels].astype(float))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        regression_result = regression_model.fit()

    echo(regression_result.summary(), logfile_only=True)

    i = 0

    for i in range(min(len(phenotype_code), len(phenotype_name))):
        if phenotype_code[i] != phenotype_name[i]:
            break

    ph_name_label = phenotype_name[i:]

    if log_dir is not None:
        model_res_fname_prefix = log_dir + f'/{phenotype_code}-{ph_name_label}' + '.covariate_correction_model_result'
        save_regression_results(phenotype_code, model_type, regression_result, model_res_fname_prefix)

    from matplotlib import pyplot as plt
    import seaborn as sns

    FIG_SIZE = (12, 6)
    fig, ax = plt.subplots(1, 2, figsize=FIG_SIZE)

    fig.suptitle(phenotype_name + ', ' + 'IRNT= ' + str(irnt_transform) + ', ' + fig_label)
    ax[0].set_title('Before correction, n= ' + str(len(data)))
    sns.distplot(data[phenotype_code], bins=30, ax=ax[0])

    ax[1].set_title('After correction, n= ' + str(len(data)))
    sns.distplot(regression_result.resid, bins=30, ax=ax[1])
    plt.show()

    if log_dir is not None:
        plt.savefig(log_dir + '/' + str(ph_name_label) + '.' + 'IRNT_' + str(irnt_transform) + '.' + fig_label + '.corrected.png', dpi=100)

    corrected = pd.DataFrame({SAMPLE_ID: data[SAMPLE_ID],
                              phenotype_name: regression_result.resid})

    corrected = pd.merge(corrected, phenotype[[SAMPLE_ID, phenotype_name + '.original']], on=SAMPLE_ID)

    if return_covariates:
        corrected = pd.merge(corrected,
                             data[[SAMPLE_ID] + [c for c in predictor_labels if c != CONST_LABEL]],
                             on=SAMPLE_ID)

    if return_covariates:
        return corrected, [c for c in predictor_labels if c != CONST_LABEL]
    else:
        return corrected


def recode_covariates(covariates, visit):
    import copy

    recoded = copy.deepcopy(covariates)

    for cov_idx in range(len(recoded)):

        cov = recoded[cov_idx]

        if type(cov) is tuple:
            if len(cov) == 3:
                cov_type = 3
                cov_col_name, cov_func, cov_label = cov
            else:
                cov_type = 2
                cov_col_name, cov_label = cov
                cov_func = lambda v: v
        else:
            cov_type = 1
            cov_col_name = cov_label = cov
            cov_func = lambda v: v

        if type(cov_col_name) is int:
            cov_col_name = str(cov_col_name)

        if type(cov_col_name) is str:
            if '-' not in cov_col_name:
                cov_col_name += '-%d.0' % (visit - 1)
        else:
            for c_idx in range(len(cov_col_name)):
                if '-' not in cov_col_name[c_idx]:
                    cov_col_name[c_idx] += '-%d.0' % (visit - 1)

        if cov_type == 1:
            recoded[cov_idx] = cov_col_name

        elif cov_type == 2:
            recoded[cov_idx] = (cov_col_name, cov_label)

        elif cov_type == 3:
            recoded[cov_idx] = (cov_col_name, cov_func, cov_label)

        else:
            echo('Error:', cov)

    return recoded


def _correct_phenotype(ph_data,
                       original_ph_name,
                       cor_ph_name,
                       covariate_labels_first_timepoint,
                       sex,
                       log_dir,
                       irnt_transform,
                       phenotype_is_binary):

    corrected_ph_data = regress_out_covariates_from_ukb_phenotype(ph_data,
                                                                  original_ph_name,
                                                                  cor_ph_name,
                                                                  [],
                                                                  sex,
                                                                  bias=True,
                                                                  # norm_factor=ph_std,
                                                                  log_dir=log_dir,
                                                                  irnt_transform=irnt_transform,
                                                                  fig_label='1st_visit',
                                                                  phenotype_is_binary=phenotype_is_binary,
                                                                  impute_missing_covatiates=False,
                                                                  additional_covariates=covariate_labels_first_timepoint)

    ph_data = pd.merge(corrected_ph_data[[SAMPLE_ID, cor_ph_name]], ph_data, on=SAMPLE_ID)
    echo('ph_data:', ph_data.shape)

    return ph_data


def correct_phenotype_for_covariates_and_drug_use(ukb_phenotypes,
                                                  sex,

                                                  phenotype_code,
                                                  phenotype_name,

                                                  medications,

                                                  covariates_to_regress,
                                                  log_dir=None,
                                                  min_subjects_per_category=50,
                                                  phenotype_is_binary=False,

                                                  impute_missing_covatiates=False,
                                                  first_visit_index=0):

    echo('[correct_phenotype_for_covariates_and_drug_use]')

    import statsmodels.api as sm
    import seaborn as sns
    from matplotlib import pyplot as plt

    # collect info for columns needed to compute phenotype values

    # correct raw phenotype for drug use
    first_visit_phenotype_label = str(phenotype_code) + f'-{first_visit_index}.0'
    cor_phenotype_first_visit, covariate_labels_first_timepoint = regress_out_covariates_from_ukb_phenotype(
                                                                          ukb_phenotypes,
                                                                          first_visit_phenotype_label,
                                                                          phenotype_name,
                                                                          recode_covariates(covariates_to_regress,
                                                                                            visit=1),
                                                                          sex,
                                                                          bias=True,
                                                                          # norm_factor=ph_std,
                                                                          log_dir=log_dir,
                                                                          irnt_transform=False,
                                                                          fig_label='1st_visit',
                                                                          phenotype_is_binary=phenotype_is_binary,
                                                                          impute_missing_covatiates=impute_missing_covatiates,
                                                                          return_covariates=True)

    cor_phenotype_first_visit = cor_phenotype_first_visit.rename(columns={phenotype_name + '.original': phenotype_name + '.original.RAW'})

    second_visit_index = first_visit_index + 1
    second_visit_phenotype_label = str(phenotype_code) + f'-{second_visit_index}.0'
    second_visit_present = second_visit_phenotype_label in list(ukb_phenotypes)

    if second_visit_present:
        cor_phenotype_second_visit = regress_out_covariates_from_ukb_phenotype(ukb_phenotypes,
                                                                               second_visit_phenotype_label,
                                                                               phenotype_name,
                                                                               recode_covariates(covariates_to_regress,
                                                                                                 visit=2),
                                                                               sex,
                                                                               bias=True,
                                                                               # norm_factor=ph_std,
                                                                               log_dir=log_dir,
                                                                               irnt_transform=False,
                                                                               fig_label='2nd_visit',
                                                                               phenotype_is_binary=phenotype_is_binary,
                                                                               additional_covariates=None,
                                                                               impute_missing_covatiates=impute_missing_covatiates)

        cor_phenotype_second_visit = cor_phenotype_second_visit.rename(
            columns={phenotype_name + '.original': phenotype_name + '.2nd_visit.original.RAW'})

    else:

        echo('WARNING: phenotype was not measured during the second visit:', phenotype_name)

        cor_phenotype_second_visit = pd.merge(cor_phenotype_first_visit[[SAMPLE_ID, phenotype_name, phenotype_name + '.original.RAW']],
                                              ukb_phenotypes[~ukb_phenotypes[AGE_2nd_visit].isnull()][[SAMPLE_ID]])

        cor_phenotype_second_visit = cor_phenotype_second_visit.rename(
            columns={phenotype_name + '.original.RAW': phenotype_name + '.2nd_visit.original.RAW'})

    (associated_meds,
     corrected_phenotype,
     med_info_for_static_correction,
     on_med_first_visit_labels_dict) = correct_effects_of_drugs_on_phenotype(cor_phenotype_first_visit,
                                                                             cor_phenotype_second_visit,
                                                                             ukb_phenotypes,
                                                                             medications,
                                                                             min_subjects_per_category,
                                                                             phenotype_is_binary,
                                                                             phenotype_name,
                                                                             sex,
                                                                             AGE_1st_visit,
                                                                             AGE_2nd_visit,
                                                                             log_dir=log_dir)

    cols_to_keep = [SAMPLE_ID, 'correction_ON_meds_t1'] + [c for c in list(corrected_phenotype)
                                                           if c.startswith('on_med.') or
                                                              c.startswith('med.') or
                                                              c.startswith('is_associated_med.')]
    del cor_phenotype_first_visit[phenotype_name]
    del cor_phenotype_second_visit[phenotype_name]

    echo('Merging corrected and corrected phenotypes')
    cor_phenotype_first_visit = pd.merge(cor_phenotype_first_visit,
                                         cor_phenotype_second_visit[[SAMPLE_ID, phenotype_name + '.2nd_visit.original.RAW']],
                                         on=SAMPLE_ID,
                                         how='left')

    cor_phenotype_first_visit = pd.merge(cor_phenotype_first_visit,
                                         corrected_phenotype[cols_to_keep],
                                         on=SAMPLE_ID)

    echo('cor_phenotype_first_visit:', cor_phenotype_first_visit.shape)

    cor_phenotype_first_visit[phenotype_name + '.original.with_med_correction.RAW'] = cor_phenotype_first_visit[phenotype_name + '.original.RAW'] - cor_phenotype_first_visit['correction_ON_meds_t1']
    cor_phenotype_first_visit[phenotype_name + '.original.with_med_correction.IRNT'] = rank_INT(cor_phenotype_first_visit[phenotype_name + '.original.with_med_correction.RAW'])
    cor_phenotype_first_visit[phenotype_name + '.original.IRNT'] = rank_INT(cor_phenotype_first_visit[phenotype_name + '.original.RAW'])

    echo('cor_phenotype_first_visit:', cor_phenotype_first_visit.shape)

    static_covariates = cor_phenotype_first_visit[[SAMPLE_ID] + covariate_labels_first_timepoint]
    covariates_with_drug_use = pd.merge(static_covariates,
                                        med_info_for_static_correction,
                                        on=SAMPLE_ID)

    corrected_ph_data = cor_phenotype_first_visit

    echo('Correcting for all covariates EXCEPT for medications')
    corrected_ph_data = _correct_phenotype(corrected_ph_data,
                                           phenotype_name + '.original.RAW',
                                           phenotype_name + '.cov_corr_except_meds.RAW',
                                           static_covariates,
                                           sex,
                                           log_dir,
                                           False,
                                           phenotype_is_binary)

    corrected_ph_data = _correct_phenotype(corrected_ph_data,
                                           phenotype_name + '.original.IRNT',
                                           phenotype_name + '.cov_corr_except_meds.IRNT',
                                           static_covariates,
                                           sex,
                                           log_dir,
                                           True,
                                           phenotype_is_binary)

    echo('Correcting for all covariates INCLUDING medications by using only first time point estimates for medication effects')
    corrected_ph_data = _correct_phenotype(corrected_ph_data,
                                           phenotype_name + '.original.RAW',
                                           phenotype_name + '.static_med_corrected.RAW',
                                           covariates_with_drug_use,
                                           sex,
                                           log_dir,
                                           False,
                                           phenotype_is_binary)

    corrected_ph_data = _correct_phenotype(corrected_ph_data,
                                           phenotype_name + '.original.IRNT',
                                           phenotype_name + '.static_med_corrected.IRNT',
                                           covariates_with_drug_use,
                                           sex,
                                           log_dir,
                                           True,
                                           phenotype_is_binary)


    if second_visit_present:
        echo('Correcting for all covariates INCLUDING medications by using second time point estimates of medication effects')
        corrected_ph_data = _correct_phenotype(corrected_ph_data,
                                               phenotype_name + '.original.with_med_correction.RAW',
                                               phenotype_name + '.dynamic_med_corrected.RAW',
                                               static_covariates,
                                               sex,
                                               log_dir,
                                               False,
                                               phenotype_is_binary)

        corrected_ph_data = _correct_phenotype(corrected_ph_data,
                                               phenotype_name + '.original.with_med_correction.IRNT',
                                               phenotype_name + '.dynamic_med_corrected.IRNT',
                                               static_covariates,
                                               sex,
                                               log_dir,
                                               True,
                                               phenotype_is_binary)
    else:

        echo('Setting', phenotype_name + '.dynamic_med_corrected.RAW', 'to', phenotype_name + '.static_med_corrected.RAW')
        corrected_ph_data[phenotype_name + '.dynamic_med_corrected.RAW'] = corrected_ph_data[phenotype_name + '.static_med_corrected.RAW']

        echo('Setting', phenotype_name + '.dynamic_med_corrected.IRNT', 'to', phenotype_name + '.static_med_corrected.IRNT')
        corrected_ph_data[phenotype_name + '.dynamic_med_corrected.IRNT'] = corrected_ph_data[phenotype_name + '.static_med_corrected.IRNT']

    return corrected_ph_data


def correct_effects_of_drugs_on_phenotype(cor_phenotype_first_visit,
                                          cor_phenotype_second_visit,
                                          ukb_phenotypes,
                                          medications,
                                          min_subjects_per_category,
                                          phenotype_is_binary,
                                          phenotype_name,
                                          sex,
                                          AGE_1st_visit,
                                          AGE_2nd_visit,
                                          log_dir=None):

    phenotype = pd.merge(cor_phenotype_first_visit,
                         cor_phenotype_second_visit,
                         on=SAMPLE_ID,
                         suffixes=['.1st_visit', '.2nd_visit'])

    phenotype[phenotype_name + '.diff'] = phenotype[phenotype_name + '.2nd_visit'] - phenotype[phenotype_name + '.1st_visit']

    phenotype = phenotype[[SAMPLE_ID, phenotype_name + '.diff']].dropna()
    phenotype = pd.merge(phenotype,
                         pd.DataFrame({SAMPLE_ID: ukb_phenotypes[SAMPLE_ID],
                                      'time': ukb_phenotypes[AGE_2nd_visit] - ukb_phenotypes[AGE_1st_visit]}).dropna())
    phenotype[CONST_LABEL] = 1
    # try to fix regression to the mean problem
    centered_baseline = cor_phenotype_first_visit[phenotype_name] - np.mean(cor_phenotype_first_visit[phenotype_name])

    echo('Mean static_no_corrected.1st_visit=', np.mean(cor_phenotype_first_visit[phenotype_name]))

    diff_df = pd.DataFrame({SAMPLE_ID: cor_phenotype_first_visit[SAMPLE_ID],
                            phenotype_name + '.1st_visit.centered': centered_baseline}).dropna()

    phenotype = pd.merge(phenotype, diff_df, on=SAMPLE_ID)

    meds_all_cols = ukb_phenotypes[[SAMPLE_ID] + MEDICATIONS_FIRST_VISIT + MEDICATIONS_SECOND_VISIT].copy()

    meds = {SAMPLE_ID: list(meds_all_cols[SAMPLE_ID])}
    first_meds_int = {SAMPLE_ID: list(meds_all_cols[SAMPLE_ID])}

    on_med_first_visit_labels_dict = {}
    medication_dummies = {}

    for medication_name, medication_codes in medications.items():

        on_med_first_visit = 'on_med.' + medication_name + '.1st_visit.' + sex
        on_med_first_visit_labels_dict[medication_name] = on_med_first_visit

        on_meds = 'med.' + medication_name + '.' + sex

        fst = meds_all_cols[MEDICATIONS_FIRST_VISIT].isin(medication_codes).any(axis=1).astype(int)
        first_meds_int[on_med_first_visit] = list(fst)

        sec = meds_all_cols[MEDICATIONS_SECOND_VISIT].isin(medication_codes).any(axis=1).astype(int)

        on_meds_dummies = pd.get_dummies(fst + 2 * sec, prefix=on_meds)

        for label in list(on_meds_dummies):
            meds[label] = list(on_meds_dummies[label])

        medication_dummies[medication_name] = list(on_meds_dummies)

    first_meds_int = pd.DataFrame(first_meds_int)

    meds = pd.DataFrame(meds)
    phenotype = pd.merge(phenotype, meds, on=SAMPLE_ID)
    to_exclude = set()

    for medication_name in medications:
        for i in range(4):
            d_label = 'med.' + medication_name + '.' + sex + '_%d' % i
            if d_label not in list(phenotype) or np.sum(phenotype[d_label]) < min_subjects_per_category:
                to_exclude.add(medication_name)

    for medication_name in sorted(to_exclude):
        echo('Excluding medications due to insufficient data:', medication_name)
        del medication_dummies[medication_name]

    all_medications = list(medication_dummies)
    all_med_dummies = [d for m in medication_dummies for d in medication_dummies[m]]

    echo('Regressing out medication effects and time between visits with n=', len(phenotype), 'samples')
    y_label = phenotype_name + '.diff'
    static_covariates = [c for c in list(phenotype) if c not in [SAMPLE_ID, y_label] and not c.startswith('med.')]

    def fit_regression(phenotype, y_label, covariates, return_pvalue_for, phenotype_is_binary):

        if phenotype_is_binary:
            regr_model_type = sm.Logit
        else:
            regr_model_type = sm.OLS

        for c in covariates:
            null_cnt = np.sum(phenotype[c].isnull())
            if null_cnt > 0:
                echo(c, 'contains null!', null_cnt)
                echo(phenotype[phenotype[c].isnull()].head())
            inf_cnt = np.sum(np.isinf(phenotype[c]))
            if inf_cnt > 0:
                echo(c, 'contains inf!', inf_cnt)
                echo(phenotype[np.isinf(phenotype[c])].head())

        regression_model = regr_model_type(phenotype[y_label],
                                           phenotype[covariates])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            regression_result = regression_model.fit()

        return regression_result.pvalues[return_pvalue_for]

    P_VALUE_THRESHOLD = 1e-3

    associated_med_dummies = []
    associated_meds = []

    best_pvalue = 0
    best_med = None

    while best_pvalue <= P_VALUE_THRESHOLD:

        medication_pvalues = {}

        if best_med is not None:
            associated_med_dummies.extend(medication_dummies[best_med])
            associated_meds.append(best_med)

            del medication_dummies[best_med]

        if len(medication_dummies) == 0:
            break

        for medication_name in medication_dummies:
            med_dummies = medication_dummies[medication_name]
            medication_pvalues[medication_name] = fit_regression(phenotype,
                                                                 y_label,
                                                                 static_covariates + associated_med_dummies + med_dummies,
                                                                 med_dummies[2],
                                                                 phenotype_is_binary)

        best_med, best_pvalue = min(medication_pvalues.items(), key=lambda kv: kv[1])

    corrected_phenotype = pd.merge(cor_phenotype_first_visit,
                                   cor_phenotype_second_visit.rename(
                                       columns={phenotype_name: phenotype_name + '.2nd_visit.static_no_med_corrected',
                                                phenotype_name + '.original': phenotype_name + '.2nd_visit.original'}),
                                   on=SAMPLE_ID,
                                   how='left')

    corrected_phenotype = pd.merge(corrected_phenotype,
                                   phenotype[[SAMPLE_ID] + all_med_dummies],
                                   on=SAMPLE_ID,
                                   how='left')

    corrected_phenotype = pd.merge(corrected_phenotype,
                                   first_meds_int[[SAMPLE_ID] + [on_med_first_visit_labels_dict[m] for m in all_medications]],
                                   on=SAMPLE_ID)

    for medication_name in all_medications:
        corrected_phenotype['on_med.' + medication_name + '.2nd_visit.' + sex] = corrected_phenotype[
                                                                                     'med.' + medication_name + '.' + sex + '_2'] + \
                                                                                 corrected_phenotype[
                                                                                     'med.' + medication_name + '.' + sex + '_3']

    for medication_name in all_medications:
        corrected_phenotype['is_associated_med.' + medication_name] = int(medication_name in associated_meds)

    if len(associated_meds) > 0:

        correction_NO_meds_labels = ['med.' + medication_name + '.' + sex + '_0' for medication_name in associated_meds]
        correction_ON_meds_labels = ['med.' + medication_name + '.' + sex + '_2' for medication_name in associated_meds]

        regression_model = sm.OLS(phenotype[y_label],
                                  phenotype[static_covariates + associated_med_dummies])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            regression_result = regression_model.fit()

        echo(regression_result.summary())

        if log_dir is not None:
            model_res_fname_prefix = log_dir + f'/{y_label}.drug_effects_model_results'
            model_type = 'OLS'
            save_regression_results(y_label, model_type, regression_result, model_res_fname_prefix)

        on_med_first_visit_labels_assoc_meds = [on_med_first_visit_labels_dict[m] for m in associated_meds]

        # compute the difference of differences for the effect of each drug:
        # first get effects of starting the drugs
        correction_ON_meds = regression_result.params[correction_ON_meds_labels].to_numpy()
        # subtract effects of not taking the drug
        correction_ON_meds -= regression_result.params[correction_NO_meds_labels].to_numpy()

        echo('medications=', correction_ON_meds_labels,
             'on_meds_betas=', list(regression_result.params[correction_ON_meds_labels]),
             'on_meds_p-value=', list(regression_result.pvalues[correction_ON_meds_labels]),
             'no_meds_betas=', list(regression_result.params[correction_NO_meds_labels]),
             'no_meds_p-value=', list(regression_result.pvalues[correction_NO_meds_labels]),
             'corrections=', list(correction_ON_meds),
             'phenotype=', phenotype_name)

        correction_ON_meds = correction_ON_meds * corrected_phenotype[on_med_first_visit_labels_assoc_meds].to_numpy()
        correction_ON_meds = correction_ON_meds.sum(axis=1)

        to_report = pd.DataFrame({'p-value': regression_result.pvalues[[c for c in associated_med_dummies]],
                                  'beta': regression_result.params[[c for c in associated_med_dummies]],
                                  'phenotype': [phenotype_name] * len(associated_med_dummies),
                                  'med_label': associated_med_dummies})

        for r_idx, row in to_report.iterrows():
            echo('FINDINGS:', 'pvalue= %lf' % row['p-value'], 'beta= %lf' % row['beta'],
                 'phenotype= %s' % row['phenotype'], 'medication= %s' % row['med_label'],
                 'n= %d' % np.sum(phenotype[r_idx]), sep='\t')

    else:
        echo('NO MEDICATIONS WERE FOUND TO AFFECT THIS PHENOTYPE')
        correction_ON_meds = 0

    on_med_first_visit_labels = [on_med_first_visit_labels_dict[m] for m in all_medications]
    med_info_for_static_correction = corrected_phenotype[[SAMPLE_ID] + on_med_first_visit_labels]

    corrected_phenotype[phenotype_name + '.corrected'] = (corrected_phenotype[phenotype_name] - correction_ON_meds)

    corrected_phenotype['correction_ON_meds_t1'] = correction_ON_meds

    return associated_meds, corrected_phenotype, med_info_for_static_correction, on_med_first_visit_labels_dict


def get_cases_for_ICD10_group(phenotypes, icd10_group, icd10_column_ids=None):

    if icd10_column_ids is None:
        icd10_column_ids = ['41202', '41204', '40006']
    echo('ICD10 column ids:', icd10_column_ids)

    echo('Reading ICD10 tree:', UKB_DATA_PATH + '/coding19.tsv')
    icd10_tree = pd.read_csv(UKB_DATA_PATH + '/coding19.tsv', sep='\t')

    echo('Computing children information in the ICD10 tree')
    icd10_children = {}

    icd10_child_to_parent = dict((r['node_id'], r['parent_id']) for _, r in icd10_tree.iterrows())
    node_id_to_icd10 = dict((r['node_id'], r['coding']) for _, r in icd10_tree.iterrows())

    for node_id in icd10_child_to_parent:

        c_node_icd10 = node_id_to_icd10[node_id]
        if c_node_icd10 not in icd10_children:
            icd10_children[c_node_icd10] = set()

        icd10_children[c_node_icd10].add(c_node_icd10)

        c_node = node_id

        while icd10_child_to_parent[c_node] != 0:
            c_parent = icd10_child_to_parent[c_node]
            c_parent_icd10 = node_id_to_icd10[c_parent]
            if c_parent_icd10 not in icd10_children:
                icd10_children[c_parent_icd10] = set()

            icd10_children[c_parent_icd10].add(c_node_icd10)

            c_node = c_parent

    echo('Processing the input group')
    if type(icd10_group) is not list:
        icd10_group = [icd10_group]

    children = set()
    for icd10_name in icd10_group:
        icd10_name = remove_special_chars(icd10_name)
        icd10_coding = icd10_tree[(icd10_tree['meaning'].apply(remove_special_chars) == icd10_name) |
                                  (icd10_tree['coding'].apply(remove_special_chars) == icd10_name)].iloc[0]['coding']

        echo('icd10 name=', icd10_name, ', coding=', icd10_coding)
        children |= icd10_children[icd10_coding]

    icd10_group_name = '-'.join([remove_special_chars(icd10_name) for icd10_name in icd10_group])
    echo('Selecting cases for:', icd10_group, ', name=', icd10_group_name, ', codes=', children)

    ICD10_hospital_columns = [c for c in list(phenotypes) if any(c.startswith(col_id) for col_id in icd10_column_ids)]
    is_case = phenotypes[ICD10_hospital_columns].isin(children).any(axis=1).astype(int)

    echo('cases=', np.sum(is_case), ', controls=', len(is_case) - np.sum(is_case))

    return pd.DataFrame({SAMPLE_ID: phenotypes[SAMPLE_ID],
                         icd10_group_name: is_case})


def get_job_info(min_samples=100):

    everyones_jobs = pd.read_csv(ROOT_PATH + '/pfiziev/ukbiobank/data/occupation_geography_geo.csv', dtype={'eid': str}).rename(columns={'eid': SAMPLE_ID})
    job_column_ids = ['132-0.0', '132-1.0', '132-2.0']

    echo('job column ids:', job_column_ids)
    everyones_jobs = everyones_jobs[[SAMPLE_ID] + job_column_ids].copy()

    everyones_jobs['has_job_entry'] = (~everyones_jobs[job_column_ids].isnull().all(axis=1)).astype(int)

    echo('Reading jobs tree:', UKB_DATA_PATH + '/coding2.tsv')
    jobs_tree = pd.read_csv(UKB_DATA_PATH + '/coding2.tsv', sep='\t')

    echo('Computing children information in the jobs tree')
    job_codes_children = {}

    node_id_to_job_id = dict((r['node_id'], r['coding']) for _, r in jobs_tree.iterrows())

    jobs_child_to_parent_codings = dict((node_id_to_job_id[r['node_id']],
                                         node_id_to_job_id[r['parent_id']] if r['parent_id'] != 0 else None)  for _, r in jobs_tree.iterrows())

    job_id_to_name = dict((r['coding'], remove_special_chars(r['meaning'])) for _, r in jobs_tree.iterrows())

    job_counts = {}
    job_id_children = {}

    for col in job_column_ids:

        counts = everyones_jobs.groupby([col]).size()

        for job_id, job_count in counts.items():
            try:
                job_id = int(job_id)
            except:
                continue

            c_node = job_id
            children = set()
            while c_node is not None:
                if c_node not in job_counts:
                    job_counts[c_node] = 0
                    job_id_children[c_node] = set()

                job_counts[c_node] += job_count

                children.add(c_node)

                job_id_children[c_node] |= children

                c_node = jobs_child_to_parent_codings[c_node]

    echo('Returning jobs with at least', min_samples, 'samples')

    for job_id in job_counts:
        job_name = job_id_to_name[job_id]

        if job_counts[job_id] >= min_samples:
            job_children = job_id_children[job_id]
            everyones_jobs['job.' + job_name] = np.where(everyones_jobs['has_job_entry'] == 1,
                                                         everyones_jobs[job_column_ids].isin(job_children).any(axis=1).astype(int),
                                                         np.nan)

    jobs_table = everyones_jobs[[c for c in list(everyones_jobs) if c not in job_column_ids]].copy()

    return jobs_table


def plot_top_genes_per_UKB_phenotype(blood_biomarkers_results, corrected_phenotypes, all_gene_var_info, phenotypes=None):

    from matplotlib import pyplot as plt
    import seaborn as sns

    def get_samples(vcfdata, homozygotes_only=False, heterozygotes_only=False):
        tag = 'all_samples'
        if homozygotes_only:
            tag = 'homozygotes'
        elif heterozygotes_only:
            tag = 'heterozygotes'

        samples = sorted(set(sid for sids in vcfdata.info[tag] for sid in sids.split(',')))
        return samples

    P_VALUE_THRESHOLD = 1e-5
    if phenotypes is None:
        phenotypes = blood_biomarkers_results.keys()
    for ph_sex in phenotypes:

        ph, sex = ph_sex.split('.')

        echo(ph_sex)

        rvt = blood_biomarkers_results[ph_sex]

        for pval_label in ['ptv|rs_pval', 'missense|rs_pval']:

            echo(pval_label)

            vt_genes = rvt[rvt[pval_label] <= P_VALUE_THRESHOLD]

            if len(vt_genes) < 3:
                vt_genes = rvt.sort_values(pval_label).head(3)

            f, ax = plt.subplots(1, 6, figsize=(35, 9))

            plt.suptitle(ph + ', sex=' + sex)

            all_values = corrected_phenotypes[ph][sex]

            sorted_all_values = sorted(all_values[ph])
            q_idx = int(len(sorted_all_values) * 1.0 / 100)

            lo_q = sorted_all_values[q_idx]
            hi_q = sorted_all_values[len(sorted_all_values) - q_idx]

            bins = np.linspace(lo_q, hi_q, 50)

            ptv_samples = {}
            missense_samples = {}

            for vt_idx, vt in enumerate([ALL_PTV, VCF_MISSENSE_VARIANT, VCF_SYNONYMOUS_VARIANT]):

                ax[vt_idx].set_title('ptv' if vt is ALL_PTV else 'missense' if vt is VCF_MISSENSE_VARIANT else 'syn')

                for gene_name in [None] + list(vt_genes[GENE_NAME]):

                    if gene_name is None:
                        samples_values = all_values[ph]
                        label = 'all, n=' + str(len(all_values))

                    else:

                        gene_variants = filter_variants(all_gene_var_info,
                                                        gene_set=[gene_name],
                                                        consequence=vt)

                        sample_ids = get_samples(gene_variants)

                        if vt is ALL_PTV:
                            ptv_samples[gene_name] = [(s_id, pos) for s_ids, pos in zip(gene_variants.info[ALL_SAMPLES],
                                                                                        gene_variants.info[VCF_POS])
                                                      for s_id in s_ids.split(',')]
                        elif vt is VCF_MISSENSE_VARIANT:

                            missense_samples[gene_name] = {}
                            gv_full = gene_variants.full()
                            #                             display(gv_full)
                            for s_ids, pAI in zip(gv_full[ALL_SAMPLES], gv_full[PRIMATEAI_SCORE]):

                                if np.isnan(pAI):
                                    continue

                                for s_id in s_ids.split(','):

                                    if s_id not in missense_samples[gene_name]:
                                        missense_samples[gene_name][s_id] = pAI
                                    else:
                                        missense_samples[gene_name][s_id] = max(missense_samples[gene_name][s_id], pAI)

                        MIN_SAMPLES_TO_PLOT = 5

                        if len(sample_ids) < MIN_SAMPLES_TO_PLOT:
                            continue

                        samples_phenotypes = pd.merge(all_values, pd.DataFrame({SAMPLE_ID: sample_ids}))
                        if len(samples_phenotypes) < MIN_SAMPLES_TO_PLOT:
                            continue

                        samples_values = samples_phenotypes[ph]

                        non_samples_values = all_values[~all_values[SAMPLE_ID].isin(sample_ids)][ph]

                        mean_diff = (np.mean(samples_values) - np.mean(non_samples_values)) / np.std(sorted_all_values)

                        rs_pval = scipy.stats.ranksums(samples_values, non_samples_values)[1]

                        label = (('* ' if rs_pval < 1e-2 else '') +
                                 gene_name +
                                 ', n=' + str(len(samples_values)) +
                                 ', log_p=%d' % math.log(rs_pval, 10) +
                                 ', norm_mdiff=%.3lf' % mean_diff)

                    samples_values = [sv for sv in samples_values if lo_q <= sv <= hi_q]
                    sns.distplot(samples_values, bins=bins, label=label, ax=ax[vt_idx])

                ax[vt_idx].legend(loc='upper right')

            max_ph_rank = 1

            for gene_name in list(vt_genes[GENE_NAME]):
                ptv_sample_ids = ptv_samples[gene_name]

                ptv_vs_pos = pd.merge(all_values,
                                      pd.DataFrame({SAMPLE_ID: [s[0] for s in ptv_sample_ids],
                                                    VCF_POS: [s[1] for s in ptv_sample_ids]}),
                                      on=SAMPLE_ID)
                if len(ptv_vs_pos) > 0:
                    if len(set(ptv_vs_pos[VCF_POS])) > 1 and len(set(ptv_vs_pos[ph])) > 1:
                        sp_r, sp_pval = scipy.stats.spearmanr(ptv_vs_pos[VCF_POS], ptv_vs_pos[ph])
                    else:
                        sp_r = 0
                        sp_pval = 1

                    if sp_pval == 0:
                        sp_pval = 1e-300

                    ptv_vs_pos['PTV pos rank'] = scipy.stats.rankdata(ptv_vs_pos[VCF_POS])


                    sns.regplot('PTV pos rank', ph, ptv_vs_pos, label=gene_name + ', sp_R= %.2lf, log_sp_p= %lf' % (sp_r, math.log(sp_pval, 10)),
                                ax=ax[3])

                pAI_vs_ph = pd.merge(all_values,
                                     pd.DataFrame({SAMPLE_ID: [s for s in sorted(missense_samples[gene_name])],
                                                   PRIMATEAI_SCORE: [missense_samples[gene_name][s] for s in
                                                                     sorted(missense_samples[gene_name])]})).dropna()

                if len(pAI_vs_ph) > 10:
                    pAI_vs_ph[ph + ' rank'] = scipy.stats.rankdata(pAI_vs_ph[ph])

                    max_ph_rank = max(max_ph_rank, max(pAI_vs_ph[ph + ' rank']))

                    if len(set(pAI_vs_ph[PRIMATEAI_SCORE])) > 1 and len(set(pAI_vs_ph[ph])) > 1:
                        sp_r, sp_pval = scipy.stats.spearmanr(pAI_vs_ph[PRIMATEAI_SCORE], pAI_vs_ph[ph])
                    else:
                        sp_r = -1
                        sp_pval = -1

                    if sp_pval == 0:
                        sp_pval = 1e-300

                    sns.regplot(PRIMATEAI_SCORE, ph + ' rank', pAI_vs_ph,
                                label=gene_name + ', n= ' + str(len(pAI_vs_ph)) + ', sp_R= %.2lf, log_sp_p= %d' % (sp_r, math.log(sp_pval, 10)), ax=ax[4])

            ax[3].set_ylim((lo_q - 1, hi_q + 1))
            ax[4].set_xlim((0, 1))
            ax[4].set_ylim((0, max_ph_rank + 1))

            ax[3].legend(loc='upper right')
            ax[4].legend(loc='upper right')


            # plot top primateAI correlated genes

            rvt_sorted = rvt[rvt['missense|n_fgr_samples'] > 10].sort_values('missense|pAI_spearman_pval')
            pAI_scores_genes = rvt_sorted[rvt_sorted['missense|pAI_spearman_pval'] <= P_VALUE_THRESHOLD]

            if len(pAI_scores_genes) < 3:
                pAI_scores_genes = rvt_sorted.head(3)

            max_ph_rank = 1

            for gene_name in pAI_scores_genes[GENE_NAME]:
                gene_variants = filter_variants(all_gene_var_info,
                                                gene_set=[gene_name],
                                                consequence=VCF_MISSENSE_VARIANT)

                missense_samples = {}
                gv_full = gene_variants.full()

                for s_ids, pAI in zip(gv_full[ALL_SAMPLES], gv_full[PRIMATEAI_SCORE]):

                    if np.isnan(pAI):
                        continue

                    for s_id in s_ids.split(','):

                        if s_id not in missense_samples:
                            missense_samples[s_id] = pAI
                        else:
                            missense_samples[s_id] = max(missense_samples[s_id], pAI)

                pAI_vs_ph = pd.merge(all_values,
                                     pd.DataFrame({SAMPLE_ID: [s for s in sorted(missense_samples)],
                                                   PRIMATEAI_SCORE: [missense_samples[s] for s in
                                                                     sorted(missense_samples)]})).dropna()

                if len(pAI_vs_ph) > 10:
                    pAI_vs_ph[ph + ' rank'] = scipy.stats.rankdata(pAI_vs_ph[ph])

                    max_ph_rank = max(max_ph_rank, max(pAI_vs_ph[ph + ' rank']))

                    if len(set(pAI_vs_ph[PRIMATEAI_SCORE])) > 1 and len(set(pAI_vs_ph[ph])) > 1:
                        sp_r, sp_pval = scipy.stats.spearmanr(pAI_vs_ph[PRIMATEAI_SCORE], pAI_vs_ph[ph])
                    else:
                        sp_r = -1
                        sp_pval = -1

                    if sp_pval == 0:
                        sp_pval = 1e-300

                    sns.regplot(PRIMATEAI_SCORE, ph + ' rank', pAI_vs_ph,
                                label=gene_name + ', n= ' + str(len(pAI_vs_ph)) + ', sp_R= %.2lf, log_sp_p= %d' % (sp_r, math.log(sp_pval, 10)), ax=ax[5])
            ax[5].set_xlim((0, 1))
            ax[5].set_ylim((0, max_ph_rank + 1))

            ax[5].legend(loc='upper right')
            plt.show()


def plot_phenotype_for_cariers(all_gene_var_info,
                               gene_name,
                               phenotype_data,
                               phenotype_names,
                               var_type,
                               var_label,
                               max_AC=None,
                               max_AF=None,
                               find_best_AC_threshold=False,
                               figsize=None,
                               figtitle_suffix=''):

    def get_samples(vcfdata, homozygotes_only=False, heterozygotes_only=False):
        tag = 'all_samples'
        if homozygotes_only:
            tag = 'homozygotes'
        elif heterozygotes_only:
            tag = 'heterozygotes'

        samples = sorted(set(sid for sids in vcfdata.info[tag] for sid in sids.split(',')))
        return samples

    gene_variants = filter_variants(all_gene_var_info,
                                    gene_name=gene_name,
                                    consequence=var_type,
                                    is_canonical=True,
                                    max_AC=max_AC,
                                    max_AF=max_AF,
                                    coverage=20,
                                    min_frac=0.8)

    phenotype_label = phenotype_names[0].split('.')[0].replace('_', ' ')

    if figsize is None:
        figsize = 8

    fig, axs = plt.subplots(1, len(phenotype_names), figsize=(figsize * len(phenotype_names), figsize))
    suptitle_label = ''
    if max_AF is not None:
        suptitle_label += ', AF$\leq$' + str(max_AF)

    if max_AC is not None:
        suptitle_label += ', AC$\leq$' + str(max_AC)

    fig.suptitle(phenotype_label + ', ' + var_label + ' variants in ' + gene_name + suptitle_label + figtitle_suffix)
    if type(axs) is not np.ndarray:
        axs = [axs]

    carriers = []

    for ax, phenotype_name in zip(axs, phenotype_names):
        cur_phenotype_label = phenotype_name.replace('_', ' ').replace('.', ':')

        all_values = phenotype_data[[SAMPLE_ID, phenotype_name]].dropna().copy()

        all_values_sorted = sorted(all_values[phenotype_name])

        bins = np.linspace(all_values_sorted[0], all_values_sorted[-1], 50)

        if find_best_AC_threshold:
            ac_thresholds = gene_variants.info[VCF_AC].unique()
        else:
            ac_thresholds = [max_AC]

        best_set = None
        best_pval = 1

        for max_AC in ac_thresholds:

            gene_variants_above_threshold = filter_variants(gene_variants, max_AC=max_AC)

            fgr_sample_ids = get_samples(gene_variants_above_threshold)

            fgr_samples_phenotypes = pd.merge(all_values, pd.DataFrame({SAMPLE_ID: fgr_sample_ids}))

            fgr_samples_values = fgr_samples_phenotypes[phenotype_name]

            bgr_samples_values = all_values[~all_values[SAMPLE_ID].isin(fgr_sample_ids)][phenotype_name]

            mean_diff = np.mean(fgr_samples_values) - np.mean(bgr_samples_values)

            rs_pval = scipy.stats.ranksums(fgr_samples_values, bgr_samples_values)[1]

            if rs_pval <= best_pval:
                best_set = (fgr_sample_ids, fgr_samples_values, bgr_samples_values, mean_diff, rs_pval, max_AC)
                best_pval = rs_pval

        fgr_sample_ids, fgr_samples_values, bgr_samples_values, mean_diff, rs_pval, max_AC = best_set
        carriers.append(fgr_sample_ids)

        # correct for multiple testing
        best_pval *= len(ac_thresholds)

        label = gene_name + ' ' + var_label + ', n=' + str(len(fgr_samples_values))

        if find_best_AC_threshold:
            label += ', n_tests=%d' % len(ac_thresholds)

        ax.set_title('Log10(RS p-value)=%d' % math.log(best_pval, 10) +
                     ', m_diff=%.3lf' % mean_diff +
                     ', max_AC=' + str(max_AC))

        sns.distplot(fgr_samples_values, bins=bins, label=label + ', m=%.4lf' % np.mean(fgr_samples_values), ax=ax)
        sns.distplot(bgr_samples_values, bins=bins, label='bgr' +
                                                          ', n=' + str(len(bgr_samples_values)) +
                                                          ', m=%.4lf' % np.mean(bgr_samples_values), ax=ax)
        ax.set_xlabel(cur_phenotype_label)

        ax.legend()

    plt.show()

    return carriers


def test_ukb_phenotypes_for_rare_variants_associations(variants,
                                                       phenotype_data,
                                                       phenotype_is_binary=False,
                                                       use_exclusive_variants_only=False,
                                                       phenotype_name=None,
                                                       find_best_AC_threshold=False,
                                                       find_best_pAI_threhold=False,
                                                       variant_types=None,
                                                       ref_phenotype=None,
                                                       ref_group=None):

    MISSENSE_LABEL = 'missense'

    ranksums_test = ranksums_with_bgr_z

    if phenotype_name is None:
        phenotype_name = [c for c in list(phenotype_data) if c != SAMPLE_ID][0]

    echo('Testing for rare variants associations:', phenotype_name,
         ', binary:', phenotype_is_binary,
         ', use_exclusive_variants_only:', use_exclusive_variants_only,
         ', find_best_AC_threshold:', find_best_AC_threshold,
         ', ref_phenotype:', ref_phenotype,
         ', ref_group:', ref_group
         )

    def get_samples(varinfo, homozygotes_only=False, heterozygotes_only=False):
        tag = 'all_samples'
        if homozygotes_only:
            tag = 'homozygotes'
        elif heterozygotes_only:
            tag = 'heterozygotes'

        samples = sorted(set(sid for sids in varinfo[tag] for sid in sids.split(',')))

        return samples

    def best_pAI_threshold_significance_test(x,
                                             n_cases,
                                             n_controls,
                                             on_per_sample_basis=False):

        if on_per_sample_basis:
            fgr_scores = sorted([r[PRIMATEAI_SCORE] for _, r in x[x['in_cases']].iterrows() for _ in range(r['AC_in_cases'])])
            bgr_scores = sorted([r[PRIMATEAI_SCORE] for _, r in x[x['in_controls']].iterrows() for _ in range(r['AC_in_controls'])])

        else:
            fgr_scores = sorted(x[x['in_cases']][PRIMATEAI_SCORE])
            bgr_scores = sorted(x[x['in_controls']][PRIMATEAI_SCORE])

        best_threshold = np.nan
        best_pvalue = 1
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
            min_gene_score = x.iloc[0]['primateAI score_min']
            max_gene_score = x.iloc[0]['primateAI score_max']
            thresholds = np.linspace(min_gene_score, max_gene_score, MAX_NUMBER_OF_THRESHOLDS)

        for score_threshold in thresholds:

            while fgr_idx < len(fgr_scores) and fgr_scores[fgr_idx] <= score_threshold:
                fgr_idx += 1

            fgr_greater = len(fgr_scores) - fgr_idx

            while bgr_idx < len(bgr_scores) and bgr_scores[bgr_idx] <= score_threshold:
                bgr_idx += 1

            bgr_greater = len(bgr_scores) - bgr_idx

            if fgr_greater + bgr_greater > 0:

                pvalue = chi2_or_fisher_test([[fgr_greater, n_cases - fgr_greater], [bgr_greater, n_controls - bgr_greater]])

                if pvalue <= best_pvalue:

                    best_pvalue = pvalue
                    best_threshold = score_threshold

                    best_fgr_greater = fgr_greater
                    best_bgr_greater = bgr_greater

                    best_odds_ratio = odds_ratio(fgr_greater,
                                                 n_cases,
                                                 bgr_greater,
                                                 n_controls)

        return min(best_pvalue * len(thresholds), 1), best_odds_ratio, best_threshold, best_fgr_greater, best_bgr_greater


    def rv_test(x, vlabel, phenotype_data, phenotype_name, n_case_variants, n_control_variants):

        gene_name = x.iloc[0][GENE_NAME]

        if rv_test.cnt % 5000 == 0:
            echo(rv_test.cnt, 'genes tested. current gene:', gene_name)

        rv_test.cnt += 1

        if find_best_AC_threshold:
            ac_thresholds = sorted(x[VCF_AC].unique())
        else:
            ac_thresholds = [max(x[VCF_AC].unique())]

        n_tests = len(ac_thresholds)

        results_at_different_ac_thresholds = []

        for max_AC in ac_thresholds:

            variants_at_AC_threhold = x[x[VCF_AC] <= max_AC]

            samples_with_variant = get_samples(variants_at_AC_threhold)

            phenotype_data_of_samples_with_variant = pd.merge(phenotype_data, pd.DataFrame({SAMPLE_ID: samples_with_variant}))

            ref_group_phenotype_values_of_samples_with_variant = ref_group_phenotype_values_of_samples_without_variant = None
            if ref_group is not None:
                ref_group_phenotype_values_of_samples_with_variant = phenotype_data_of_samples_with_variant[phenotype_data_of_samples_with_variant[ref_group] == 0][phenotype_name]
                phenotype_data_of_samples_with_variant = phenotype_data_of_samples_with_variant[phenotype_data_of_samples_with_variant[ref_group] == 1]

            return_dict = {}

            if len(phenotype_data_of_samples_with_variant) == 0:
                if phenotype_is_binary:
                    return_dict.update({vlabel + '|gene_vars_in_cases': 0,
                                        vlabel + '|gene_vars_in_controls': 0,
                                        vlabel + '|total_vars_in_cases': n_case_variants,
                                        vlabel + '|total_vars_in_controls': n_control_variants,
                                        vlabel + '|per_var_OR': 1,
                                        vlabel + '|per_var_pval': 1,

                                        vlabel + '|cases_with_var': 0,
                                        vlabel + '|controls_with_var': 0,
                                        vlabel + '|cases_without_var': np.sum(phenotype_data[phenotype_name]),
                                        vlabel + '|controls_without_var': len(phenotype_data) - np.sum(phenotype_data[phenotype_name]),

                                        vlabel + '|per_sample_OR': 1,
                                        vlabel + '|per_sample_pval': 1,
                                        vlabel + '|best_AC': max_AC,
                                        vlabel + '|n_tests': n_tests,

                                        })

                else:

                    return_dict.update({vlabel + '|n_fgr_samples': len(phenotype_data_of_samples_with_variant),
                                        vlabel + '|n_bgr_samples': len(phenotype_data),
                                        vlabel + '|fgr_mean': 0,
                                        vlabel + '|bgr_mean': 0,
                                        vlabel + '|mean_diff': 0,
                                        vlabel + '|per_sample_OR': -1,
                                        vlabel + '|per_sample_pval': 1,
                                        vlabel + '|x_test_dec_K': 0,
                                        vlabel + '|x_test_dec_pval': 1,
                                        vlabel + '|x_test_inc_K': 0,
                                        vlabel + '|x_test_inc_pval': 1,
                                        vlabel + '|best_AC': max_AC,
                                        vlabel + '|n_tests': n_tests
                                        })

                    if ref_group is not None or ref_phenotype is not None:
                        return_dict.update({vlabel + '|ref_stat': 0,
                                            vlabel + '|ref_pval': 1,
                                            vlabel + '|ref_info': ''
                                            })

                if vlabel == MISSENSE_LABEL:
                    return_dict[vlabel + '|pAI_spearman_r'] = 0
                    return_dict[vlabel + '|pAI_spearman_pval'] = 1

                    if phenotype_is_binary and find_best_pAI_threhold:
                        for on_per_sample_basis in [True, False]:

                            label = vlabel + '|pAI_best_' + ('per_sample' if on_per_sample_basis else 'per_var') + '_'
                            return_dict[label + 'pval'] = 1
                            return_dict[label + 'OR'] = 1
                            return_dict[label + 't'] = 0
                            return_dict[label + 'greater_cases'] = 0
                            return_dict[label + 'greater_controls'] = 0

                return pd.Series(return_dict)

            phenotype_values_of_samples_with_variant = phenotype_data_of_samples_with_variant[phenotype_name]

            phenotype_data_of_samples_without_variant = phenotype_data[~phenotype_data[SAMPLE_ID].isin(samples_with_variant)]

            if ref_group is not None:
                ref_group_phenotype_values_of_samples_without_variant = phenotype_data_of_samples_without_variant[phenotype_data_of_samples_without_variant[ref_group] == 0][phenotype_name]
                phenotype_values_of_samples_without_variant = phenotype_data_of_samples_without_variant[phenotype_data_of_samples_without_variant[ref_group] == 1][phenotype_name]
            else:
                phenotype_values_of_samples_without_variant = phenotype_data_of_samples_without_variant[phenotype_name]

            if phenotype_is_binary:

                n_case_variants_in_gene = np.sum(variants_at_AC_threhold['in_cases'])
                n_control_variants_in_gene = np.sum(variants_at_AC_threhold['in_controls'])

                per_var_pval = chi2_or_fisher_test([[n_case_variants_in_gene, n_case_variants - n_case_variants_in_gene],
                                                    [n_control_variants_in_gene, n_control_variants - n_control_variants_in_gene]])

                per_var_OR = odds_ratio(n_case_variants_in_gene, n_case_variants, n_control_variants_in_gene, n_control_variants)

                cases_with_variant = int(np.sum(phenotype_values_of_samples_with_variant))
                cases_without_variant = int(np.sum(phenotype_values_of_samples_without_variant))

                per_sample_pval = chi2_or_fisher_test([[cases_with_variant, len(phenotype_values_of_samples_with_variant) - cases_with_variant],
                                                       [cases_without_variant, len(phenotype_values_of_samples_without_variant) - cases_without_variant]])

                per_sample_stat = odds_ratio(cases_with_variant, len(phenotype_values_of_samples_with_variant),
                                             cases_without_variant, len(phenotype_values_of_samples_without_variant))

                return_dict.update({
                    vlabel + '|gene_vars_in_cases': n_case_variants_in_gene,
                    vlabel + '|gene_vars_in_controls': n_control_variants_in_gene,
                    vlabel + '|total_vars_in_cases': n_case_variants,
                    vlabel + '|total_vars_in_controls': n_control_variants,

                    vlabel + '|per_var_OR': per_var_OR,
                    vlabel + '|per_var_pval': per_var_pval,

                    vlabel + '|cases_with_var': cases_with_variant,
                    vlabel + '|controls_with_var': len(phenotype_values_of_samples_with_variant) - cases_with_variant,
                    vlabel + '|cases_without_var': cases_without_variant,
                    vlabel + '|controls_without_var': len(phenotype_values_of_samples_without_variant) - cases_without_variant,

                    vlabel + '|per_sample_OR': per_sample_stat,
                    vlabel + '|per_sample_pval': per_sample_pval,
                    vlabel + '|best_AC': max_AC,
                    vlabel + '|n_tests': n_tests

                })

            else:

                ref_test_stat = None
                ref_test_pval = None
                ref_info = None
                if ref_phenotype is not None:
                    ref_phenotype_values_of_samples_with_variant = phenotype_data_of_samples_with_variant[ref_phenotype]
                    ref_phenotype_values_of_samples_without_variant = phenotype_data[~phenotype_data[SAMPLE_ID].isin(samples_with_variant)][ref_phenotype]

                    ref_test_stat, ref_test_pval = ranksums_test(ref_phenotype_values_of_samples_with_variant,
                                                                 ref_phenotype_values_of_samples_without_variant)

                    m1 = np.mean(ref_phenotype_values_of_samples_with_variant)
                    m2 = np.mean(ref_phenotype_values_of_samples_without_variant)

                    ref_info = 'f=%.2lf;b=%2.lf;d=%.2lf;nf=%d;nb=%d' % (m1, m2, m1 - m2,
                                                                        len(ref_phenotype_values_of_samples_with_variant),
                                                                        len(ref_phenotype_values_of_samples_without_variant))

                if ref_group is not None:
                    ref_test_stat, ref_test_pval = ranksums_test(ref_group_phenotype_values_of_samples_with_variant,
                                                                 ref_group_phenotype_values_of_samples_without_variant)

                    m1 = np.mean(ref_group_phenotype_values_of_samples_with_variant)
                    m2 = np.mean(ref_group_phenotype_values_of_samples_without_variant)

                    ref_info = 'f=%.2lf;b=%2.lf;d=%.2lf;nf=%d;nb=%d' % (m1, m2, m1 - m2,
                                                                        len(ref_group_phenotype_values_of_samples_with_variant),
                                                                        len(ref_group_phenotype_values_of_samples_without_variant))

                fgr_mean = np.mean(phenotype_values_of_samples_with_variant)
                bgr_mean = np.mean(phenotype_values_of_samples_without_variant)

                mean_diff = fgr_mean - bgr_mean

                test_stat, test_pval = ranksums_test(phenotype_values_of_samples_with_variant,
                                                     phenotype_values_of_samples_without_variant,
                                                     bgr_z=ref_test_stat)

                # perform outliers rank test
                n_all_ranks = len(phenotype_values_of_samples_with_variant) + len(phenotype_values_of_samples_without_variant)
                ph_ranks = scipy.stats.rankdata(list(phenotype_values_of_samples_with_variant) +
                                                list(phenotype_values_of_samples_without_variant))[:len(phenotype_values_of_samples_with_variant)]

                x_dec_K, x_dec_pval = test_ranks(ph_ranks, n_all_ranks)

                x_inc_K, x_inc_pval = test_ranks(n_all_ranks + 1 - ph_ranks, n_all_ranks)

                return_dict.update({vlabel + '|n_fgr_samples': len(phenotype_values_of_samples_with_variant),
                                    vlabel + '|n_bgr_samples': len(phenotype_values_of_samples_without_variant),
                                    vlabel + '|fgr_mean': fgr_mean,
                                    vlabel + '|bgr_mean': bgr_mean,
                                    vlabel + '|mean_diff': mean_diff,
                                    vlabel + '|per_sample_OR': test_stat,
                                    vlabel + '|per_sample_pval': test_pval,
                                    vlabel + '|x_test_dec_K': x_dec_K,
                                    vlabel + '|x_test_dec_pval': x_dec_pval,
                                    vlabel + '|x_test_inc_K': x_inc_K,
                                    vlabel + '|x_test_inc_pval': x_inc_pval,
                                    vlabel + '|best_AC': max_AC,
                                    vlabel + '|n_tests': n_tests
                                    })

                if ref_phenotype is not None or ref_group is not None:
                    return_dict.update({vlabel + '|ref_stat': ref_test_stat,
                                        vlabel + '|ref_pval': ref_test_pval,
                                        vlabel + '|ref_info': ref_info
                                        })

            if vlabel == MISSENSE_LABEL:
                pAI_scores = {}

                for row_idx in range(len(variants_at_AC_threhold)):

                    row_samples = variants_at_AC_threhold.iloc[row_idx][ALL_SAMPLES].split(',')
                    pAI = variants_at_AC_threhold.iloc[row_idx][PRIMATEAI_SCORE]

                    if np.isnan(pAI):
                        continue

                    for s_id in row_samples:
                        if s_id in pAI_scores:
                            pAI_scores[s_id] = max(pAI_scores[s_id], pAI)
                        else:
                            pAI_scores[s_id] = pAI

                pAI_scores_df = pd.merge(phenotype_data_of_samples_with_variant,
                                         pd.DataFrame({SAMPLE_ID: sorted(pAI_scores),
                                                       PRIMATEAI_SCORE: [pAI_scores[s_id] for s_id in sorted(pAI_scores)]}),
                                         on=SAMPLE_ID)

                if phenotype_is_binary:
                    phenotype_true_scores = pAI_scores_df[pAI_scores_df[phenotype_name] == 1][PRIMATEAI_SCORE]
                    phenotype_false_scores = pAI_scores_df[pAI_scores_df[phenotype_name] == 0][PRIMATEAI_SCORE]
                    sp_r, sp_pval = ranksums_test(phenotype_true_scores, phenotype_false_scores)

                else:
                    sp_r, sp_pval = scipy.stats.spearmanr(pAI_scores_df[phenotype_name], pAI_scores_df[PRIMATEAI_SCORE])

                return_dict[vlabel + '|pAI_spearman_r'] = sp_r
                return_dict[vlabel + '|pAI_spearman_pval'] = sp_pval

                if phenotype_is_binary and find_best_pAI_threhold:

                    n_cases = np.sum(phenotype_data[phenotype_name])
                    n_controls = len(phenotype_data) - n_cases

                    for on_per_sample_basis in [True, False]:

                        best_pvalue, best_odds_ratio, best_threshold, best_fgr_greater, best_bgr_greater = \
                            best_pAI_threshold_significance_test(variants_at_AC_threhold,
                                                                 n_cases if on_per_sample_basis else n_case_variants,
                                                                 n_controls if on_per_sample_basis else n_control_variants,
                                                                 on_per_sample_basis=on_per_sample_basis)
                        label = vlabel + '|pAI_best_' + ('per_sample' if on_per_sample_basis else 'per_var') + '_'
                        return_dict[label + 'pval'] = best_pvalue
                        return_dict[label + 'OR'] = best_odds_ratio
                        return_dict[label + 't'] = best_threshold
                        return_dict[label + 'greater_cases'] = best_fgr_greater
                        return_dict[label + 'greater_controls'] = best_bgr_greater

            results_at_different_ac_thresholds.append(return_dict)

        best_return_dict = None
        best_pval = 1

        for return_dict in results_at_different_ac_thresholds:
            if phenotype_is_binary:
                c_pval = min(return_dict[vlabel + '|per_var_pval'], return_dict[vlabel + '|per_sample_pval'])
            else:
                c_pval = min(return_dict[vlabel + '|x_test_dec_pval'],
                             return_dict[vlabel + '|x_test_inc_pval'],
                             return_dict[vlabel + '|per_sample_pval'])

            if vlabel == MISSENSE_LABEL and find_best_pAI_threhold and phenotype_is_binary:
                c_pval = min(c_pval,
                             return_dict[MISSENSE_LABEL + '|pAI_best_per_var_pval'],
                             return_dict[MISSENSE_LABEL + '|pAI_best_per_sample_pval'])

            if c_pval <= best_pval:
                best_pval = c_pval
                best_return_dict = return_dict

        if best_return_dict is None:
            echo('ERROR:', gene_name)

        return pd.Series(best_return_dict)

    result = None

    def merge_results(prev_result, new_result):
        if prev_result is None:
            return new_result
        else:
            return pd.merge(prev_result, new_result, on=GENE_NAME, how='outer')

    full = variants.full().copy()

    variant_types_to_test = [(ALL_PTV, 'ptv'),
                             (DELETERIOUS_VARIANT, DELETERIOUS_VARIANT),
                             (DELETERIOUS_MISSENSE, 'del_missense'),
                             (VCF_MISSENSE_VARIANT, MISSENSE_LABEL),
                             (VCF_SYNONYMOUS_VARIANT, 'syn')
                             ]

    if variant_types is not None:
        variant_types_to_test = [(vt, vl) for vt, vl in variant_types_to_test if vt in variant_types]

    for vtype, vlabel in variant_types_to_test:

        echo('Testing:', vlabel)

        if vtype not in [DELETERIOUS_VARIANT, DELETERIOUS_MISSENSE]:
            if type(vtype) is not list:
                vtype = [vtype]
            cur_full = full[full[VCF_CONSEQUENCE].isin(vtype)].copy()
        else:

            cur_full = filter_variants(variants, consequence=vtype).full().copy()

        n_variants = len(cur_full)
        all_sample_ids = get_samples(cur_full)
        echo('n_variants=', n_variants, 'from n_genes=', len(cur_full.drop_duplicates(GENE_NAME)), 'from n_samples=', len(all_sample_ids))

        n_case_variants = None
        n_control_variants = None

        if phenotype_is_binary:
            case_samples = set(phenotype_data[phenotype_data[phenotype_name] == 1][SAMPLE_ID])
            control_samples = set(phenotype_data[phenotype_data[phenotype_name] == 0][SAMPLE_ID])

            cur_full['in_cases'] = [any(sid in case_samples for sid in sids.split(','))
                                                        for sids in cur_full['all_samples']]

            cur_full['AC_in_cases'] = [sum([sid in case_samples for sid in sids.split(',')])
                                                            for sids in cur_full['all_samples']]

            cur_full['in_controls'] = [any(sid in control_samples for sid in sids.split(','))
                                       for sids in cur_full['all_samples']]

            cur_full['AC_in_controls'] = [sum([sid in control_samples for sid in sids.split(',')])
                                       for sids in cur_full['all_samples']]

            if use_exclusive_variants_only:
                cur_full = cur_full[cur_full['in_cases'] ^ cur_full['in_controls']]

            n_case_variants = np.sum(cur_full['in_cases'])
            n_control_variants = np.sum(cur_full['in_controls'])

            n_case_alleles = np.sum(cur_full['AC_in_cases'])
            n_control_alleles = np.sum(cur_full['AC_in_controls'])

            echo('cases: n_variants=', n_case_variants, ', n_alleles=', n_case_alleles, ', n_cases=', len(case_samples), ', per sample=', float(n_case_alleles) / len(case_samples))
            echo('control: n_variants=', n_control_variants, ', n_alleles=', n_control_alleles, ', n_controls=', len(control_samples), ', per sample=', float(n_control_alleles) / len(control_samples))

        rv_test.cnt = 0

        new_result = cur_full.groupby(GENE_NAME).apply(rv_test,
                                                       vlabel,
                                                       phenotype_data,
                                                       phenotype_name,
                                                       n_case_variants,
                                                       n_control_variants)
        sort_by = vlabel + '|per_sample_pval'

        new_result = new_result.sort_values(sort_by)
        new_result = new_result.reset_index()

        result = merge_results(result, new_result)

    return result

def test_quantitative_ukb_phenotype_for_rare_variants_associations(variants,
                                                                   phenotype,
                                                                   phenotype_is_binary=False,
                                                                   use_exclusive_variants_only=False,
                                                                   phenotype_name=None,
                                                                   find_best_AC_threshold=False):

    # echo('New')
    MISSENSE_LABEL = 'missense'

    if phenotype_name is None:
        phenotype_name = [c for c in list(phenotype) if c != SAMPLE_ID][0]

    echo('Testing for rare variants associations:', phenotype_name,
         ', binary:', phenotype_is_binary,
         ', use_exclusive_variants_only:', use_exclusive_variants_only)

    def get_samples(varinfo, homozygotes_only=False, heterozygotes_only=False):
        tag = 'all_samples'
        if homozygotes_only:
            tag = 'homozygotes'
        elif heterozygotes_only:
            tag = 'heterozygotes'

        samples = sorted(set(sid for sids in varinfo[tag] for sid in sids.split(',')))

        return samples

    def ranksum_test(x, vlabel, phenotype, phenotype_name, n_case_variants, n_control_variants):

        gene_name = x.iloc[0][GENE_NAME]

        if ranksum_test.cnt % 5000 == 0:
            echo(ranksum_test.cnt, 'genes tested. current gene:', gene_name)

        ranksum_test.cnt += 1

        if find_best_AC_threshold:
            ac_thresholds = x[VCF_AC].unique()
        else:
            ac_thresholds = [max(x[VCF_AC].unique())]

        n_tests = len(ac_thresholds)

        best_pval = 1
        best_res = None
        best_ac = 0

        for max_AC in ac_thresholds:

            variants_at_AC_threhold = x[x[VCF_AC] <= max_AC]

            sample_ids = get_samples(variants_at_AC_threhold)

            samples_phenotypes = pd.merge(phenotype, pd.DataFrame({SAMPLE_ID: sample_ids}))
            return_dict = {}

            if len(samples_phenotypes) == 0:
                continue

            samples_values = samples_phenotypes[phenotype_name]
            non_samples_values = phenotype[~phenotype[SAMPLE_ID].isin(sample_ids)][phenotype_name]

            fgr_mean = np.mean(samples_values)
            bgr_mean = np.mean(non_samples_values)

            mean_diff = fgr_mean - bgr_mean

            if phenotype_is_binary:

                fgr_with_phenotype = int(np.sum(samples_values))
                bgr_with_phenotype = int(np.sum(non_samples_values))

                pvalue = chi2_or_fisher_test([[fgr_with_phenotype, len(samples_values) - fgr_with_phenotype],
                                               [bgr_with_phenotype, len(non_samples_values) - bgr_with_phenotype]])
                rs_stat = -1

                n_case_variants_in_gene = np.sum(variants_at_AC_threhold['in_cases'])
                n_control_variants_in_gene = np.sum(variants_at_AC_threhold['in_controls'])

                chi2_pvalue = chi2_or_fisher_test([[n_case_variants_in_gene, n_case_variants - n_case_variants_in_gene],
                                                 [n_control_variants_in_gene, n_control_variants - n_control_variants_in_gene]])

                chi2_pvalue *= n_tests

                odds_r = odds_ratio(n_case_variants_in_gene, n_case_variants, n_control_variants_in_gene, n_control_variants)

                return_dict.update({
                    vlabel + '|n_fgr_variants': n_case_variants_in_gene,
                    vlabel + '|n_bgr_variants': n_control_variants_in_gene,
                    vlabel + '|n_total_fgr_variants': n_case_variants,
                    vlabel + '|n_total_bgr_variants': n_control_variants,
                    vlabel + '|chi2_OR': odds_r,
                    vlabel + '|chi2_pval': chi2_pvalue
                })

            else:
                rs_stat, pvalue = scipy.stats.ranksums(samples_values, non_samples_values)

            pvalue *= n_tests

            return_dict.update({vlabel + '|n_fgr_samples': len(samples_values),
                                vlabel + '|n_bgr_samples': len(non_samples_values),
                                vlabel + '|fgr_mean': fgr_mean,
                                vlabel + '|bgr_mean': bgr_mean,
                                vlabel + '|mean_diff': mean_diff,
                                vlabel + '|rs_stat': rs_stat,
                                vlabel + '|rs_pval': pvalue})

            if vlabel == MISSENSE_LABEL:
                pAI_scores = {}

                for row_idx in range(len(variants_at_AC_threhold)):

                    row_samples = variants_at_AC_threhold.iloc[row_idx][ALL_SAMPLES].split(',')
                    pAI = variants_at_AC_threhold.iloc[row_idx][PRIMATEAI_SCORE]

                    if np.isnan(pAI):
                        continue

                    for s_id in row_samples:
                        if s_id in pAI_scores:
                            pAI_scores[s_id] = max(pAI_scores[s_id], pAI)
                        else:
                            pAI_scores[s_id] = pAI

                pAI_scores_df = pd.merge(samples_phenotypes,
                                         pd.DataFrame({SAMPLE_ID: sorted(pAI_scores),
                                                       PRIMATEAI_SCORE: [pAI_scores[s_id] for s_id in sorted(pAI_scores)]}),
                                         on=SAMPLE_ID)

                if phenotype_is_binary:
                    phenotype_true_scores = pAI_scores_df[pAI_scores_df[phenotype_name] == 1][PRIMATEAI_SCORE]
                    phenotype_false_scores = pAI_scores_df[pAI_scores_df[phenotype_name] == 0][PRIMATEAI_SCORE]
                    sp_r, sp_pval = scipy.stats.ranksums(phenotype_true_scores, phenotype_false_scores)

                else:
                    sp_r, sp_pval = scipy.stats.spearmanr(pAI_scores_df[phenotype_name], pAI_scores_df[PRIMATEAI_SCORE])

                return_dict[vlabel + '|pAI_spearman_r'] = sp_r
                return_dict[vlabel + '|pAI_spearman_pval'] = sp_pval

            if pvalue <= best_pval:
                best_res = return_dict
                best_pval = pvalue
                best_ac = max_AC

        if best_res is not None:
            return_dict = best_res

        else:
            return_dict = {}
            if phenotype_is_binary:
                return_dict.update({vlabel + '|n_fgr_variants': 0,
                                    vlabel + '|n_bgr_variants': 0,
                                    vlabel + '|n_total_fgr_variants': n_case_variants,
                                    vlabel + '|n_total_bgr_variants': n_control_variants,
                                    vlabel + '|chi2_OR': 0,
                                    vlabel + '|chi2_pval': 1})

            return_dict.update({vlabel + '|n_fgr_samples': 0,
                                vlabel + '|n_bgr_samples': len(phenotype),
                                vlabel + '|fgr_mean': 0,
                                vlabel + '|bgr_mean': 0,
                                vlabel + '|mean_diff': 0,
                                vlabel + '|rs_stat': -1,
                                vlabel + '|rs_pval': 1})

            if vlabel == MISSENSE_LABEL:
                return_dict[vlabel + '|pAI_spearman_r'] = 0
                return_dict[vlabel + '|pAI_spearman_pval'] = 1

        return_dict[vlabel + '|n_tests'] = n_tests
        return_dict[vlabel + '|max_AC'] = best_ac

        return pd.Series(return_dict)

    result = None

    def merge_results(prev_result, new_result):
        if prev_result is None:
            return new_result
        else:
            return pd.merge(prev_result, new_result, on=GENE_NAME, how='outer')

    full = variants.full().copy()

    for vtype, vlabel in [(DELETERIOUS_VARIANT, DELETERIOUS_VARIANT),
                          (ALL_PTV, 'ptv'),
                          (VCF_MISSENSE_VARIANT, MISSENSE_LABEL)
                          ]:

        echo('Testing:', vlabel)

        if vtype is not DELETERIOUS_VARIANT:
            if type(vtype) is not list:
                vtype = [vtype]
            cur_full = full[full[VCF_CONSEQUENCE].isin(vtype)].copy()
        else:

            cur_full = filter_variants(variants, consequence=DELETERIOUS_VARIANT).full().copy()

        n_variants = len(cur_full)
        all_sample_ids = get_samples(cur_full)
        echo('n_variants=', n_variants, 'from n_genes=', len(cur_full.drop_duplicates(GENE_NAME)), 'from n_samples=', len(all_sample_ids))

        n_case_variants = None
        n_control_variants = None

        if phenotype_is_binary:
            case_samples = set(phenotype[phenotype[phenotype_name] == 1][SAMPLE_ID])
            control_samples = set(phenotype[phenotype[phenotype_name] == 0][SAMPLE_ID])

            cur_full['in_cases'] = [any(sid in case_samples for sid in sids.split(','))
                                                        for sids in cur_full['all_samples']]

            cur_full['in_controls'] = [any(sid in control_samples for sid in sids.split(','))
                                       for sids in cur_full['all_samples']]

            if use_exclusive_variants_only:
                cur_full = cur_full[cur_full['in_cases'] ^ cur_full['in_controls']]

            n_case_variants = np.sum(cur_full['in_cases'])
            n_control_variants = np.sum(cur_full['in_controls'])

            echo('n_case_variants=', n_case_variants, ', per sample=', float(n_case_variants) / len(case_samples))
            echo('n_control_variants=', n_control_variants, ', per sample=', float(n_control_variants) / len(control_samples))

        ranksum_test.cnt = 0
        new_result = cur_full.groupby(GENE_NAME).apply(ranksum_test,
                                                       vlabel,
                                                       phenotype,
                                                       phenotype_name,
                                                       n_case_variants,
                                                       n_control_variants)
        if phenotype_is_binary:
            sort_by = vlabel + '|chi2_pval'
        else:
            sort_by = vlabel + '|rs_pval'

        new_result = new_result.sort_values(sort_by)
        new_result = new_result.reset_index()

        result = merge_results(result, new_result)

    return result


def test_multiple_quantitative_ukb_phenotype_for_rare_variants_associations(variants,
                                                                            phenotype,
                                                                            phenotype_is_binary=False,
                                                                            use_exclusive_variants_only=False,
                                                                            phenotype_names=None,
                                                                            find_best_AC_threshold=False):

    MISSENSE_LABEL = 'missense'

    echo('Testing for rare variants associations:', phenotype_names,
         ', binary:', phenotype_is_binary,
         ', use_exclusive_variants_only:', use_exclusive_variants_only)

    def get_samples(varinfo, homozygotes_only=False, heterozygotes_only=False):
        tag = 'all_samples'
        if homozygotes_only:
            tag = 'homozygotes'
        elif heterozygotes_only:
            tag = 'heterozygotes'

        samples = sorted(set(sid for sids in varinfo[tag] for sid in sids.split(',')))

        return samples

    def ranksum_test(x, vlabel, phenotype, phenotype_names, n_case_variants, n_control_variants):

        gene_name = x.iloc[0][GENE_NAME]

        if ranksum_test.cnt % 5000 == 0:
            echo(ranksum_test.cnt, 'genes tested. current gene:', gene_name)

        ranksum_test.cnt += 1

        if find_best_AC_threshold:
            ac_thresholds = sorted(x[VCF_AC].unique())
        else:
            ac_thresholds = [max(x[VCF_AC].unique())]

        n_tests = len(ac_thresholds)

        phenotype_results = dict((p, []) for p in phenotype_names)

        for max_AC in ac_thresholds:

            variants_at_AC_threhold = x[x[VCF_AC] <= max_AC]

            sample_ids = get_samples(variants_at_AC_threhold)

            samples_phenotypes = pd.merge(phenotype, pd.DataFrame({SAMPLE_ID: sample_ids}))

            if len(samples_phenotypes) == 0:
                continue

            fgr_values = samples_phenotypes[phenotype_names]
            bgr_values = phenotype[~phenotype[SAMPLE_ID].isin(sample_ids)][phenotype_names]

            fgr_mean = np.mean(fgr_values, axis=0)
            bgr_mean = np.mean(bgr_values, axis=0)

            mean_diff = fgr_mean - bgr_mean

            cur_rs_results = [scipy.stats.ranksums(fgr_values[p], bgr_values[p]) for p in phenotype_names]
            cur_rs_results = [(s, p * n_tests) for s, p in cur_rs_results]

            for phenotype_name, (rs_stat, pvalue) in zip(phenotype_names, cur_rs_results):
                phenotype_results[phenotype_name].append(
                                   {vlabel + '|' + phenotype_name + '|n_fgr_samples': len(fgr_values),
                                    vlabel + '|' + phenotype_name + '|n_bgr_samples': len(bgr_values),
                                    vlabel + '|' + phenotype_name + '|fgr_mean': fgr_mean[phenotype_name],
                                    vlabel + '|' + phenotype_name + '|bgr_mean': bgr_mean[phenotype_name],
                                    vlabel + '|' + phenotype_name + '|mean_diff': mean_diff[phenotype_name],
                                    vlabel + '|' + phenotype_name + '|rs_stat': rs_stat,
                                    vlabel + '|' + phenotype_name + '|rs_pval': min(pvalue, 1),
                                    vlabel + '|' + phenotype_name + '|n_tests': n_tests,
                                    vlabel + '|' + phenotype_name + '|max_AC': max_AC
                                    })

        return_dict = {}
        for phenotype_name in phenotype_names:
            if len(phenotype_results[phenotype_name]) == 0:
                best_result = {vlabel + '|' + phenotype_name + '|n_fgr_samples': 0,
                               vlabel + '|' + phenotype_name + '|n_bgr_samples': 0,
                               vlabel + '|' + phenotype_name + '|fgr_mean': 0,
                               vlabel + '|' + phenotype_name + '|bgr_mean': 0,
                               vlabel + '|' + phenotype_name + '|mean_diff': 0,
                               vlabel + '|' + phenotype_name + '|rs_stat': 0,
                               vlabel + '|' + phenotype_name + '|rs_pval': 1,
                               vlabel + '|' + phenotype_name + '|n_tests': 0,
                               vlabel + '|' + phenotype_name + '|max_AC': 0}
            else:
                best_result = min(phenotype_results[phenotype_name],
                                  key=lambda d: d[vlabel + '|' + phenotype_name + '|rs_pval'])

            return_dict.update(best_result)

        return pd.Series(return_dict)

    result = None

    def merge_results(prev_result, new_result):
        if prev_result is None:
            return new_result
        else:
            return pd.merge(prev_result, new_result, on=GENE_NAME, how='outer')

    full = variants.full().copy()

    for vtype, vlabel in [(DELETERIOUS_VARIANT, DELETERIOUS_VARIANT),
                          (ALL_PTV, 'ptv'),
                          (VCF_MISSENSE_VARIANT, MISSENSE_LABEL)
                          ]:

        echo('Testing:', vlabel)

        if vtype is not DELETERIOUS_VARIANT:
            if type(vtype) is not list:
                vtype = [vtype]
            cur_full = full[full[VCF_CONSEQUENCE].isin(vtype)].copy()
        else:

            cur_full = filter_variants(variants, consequence=DELETERIOUS_VARIANT).full().copy()

        n_variants = len(cur_full)
        all_sample_ids = get_samples(cur_full)
        echo('n_variants=', n_variants, 'from n_genes=', len(cur_full.drop_duplicates(GENE_NAME)), 'from n_samples=', len(all_sample_ids))

        n_case_variants = None
        n_control_variants = None

        ranksum_test.cnt = 0
        new_result = cur_full.groupby(GENE_NAME).apply(ranksum_test,
                                                       vlabel,
                                                       phenotype,
                                                       phenotype_names,
                                                       n_case_variants,
                                                       n_control_variants)

        sort_by = vlabel + '|' + phenotype_names[0] + '|rs_pval'

        echo('sort_by:', sort_by)

        new_result = new_result.sort_values(sort_by)
        new_result = new_result.reset_index()

        result = merge_results(result, new_result)

    return result


def get_ukb_medications():
    # return a list of reported medications in the UKB

    meds = pd.read_excel(
        UKB_RARE_VARIANTS_PATH + '/medications_gwas/med_groups.xlsx',
        sheet_name='all_meds')
    med_categories = pd.read_excel(
        UKB_RARE_VARIANTS_PATH + '/medications_gwas/med_groups.xlsx',
        sheet_name='cats')

    def get_med_category(row):

        categories = sorted(med_categories['med_category'])
        cat_names = dict((r['med_category'], r['category_name']) for _, r in med_categories.iterrows())

        row_cats = set()
        atc_codes = list(map(lambda x: x.strip(), row['atc_code'].split('|')))
        for atc_code in atc_codes:
            atc_code = atc_code.strip()
            for c_i, c in enumerate(categories):
                if atc_code.startswith(c):
                    row_cats.add(c)

        if len(row_cats) == 0:
            row['med_category'] = '|'.join(atc_codes)
            row['med_category_name'] = remove_special_chars(row['ukb_name'])
        else:
            row['med_category'] = '|'.join(sorted(row_cats))
            row['med_category_name'] = '|'.join(cat_names[c] for c in sorted(row_cats))

        return row

    res = meds.apply(get_med_category, axis=1)

    counts = pd.read_excel(UKB_DATA_PATH + '/medication_stats.xlsx').rename(columns={'Count': 'n_subjects'})

    res = pd.merge(res, counts[['coding', 'n_subjects']], on='coding')

    return res



def get_all_meds_in_category(ukb_medications, cat_code=None, cat_name=None, return_codes=False):
    # get all medications belonging to specific category of medications

    if cat_code is not None:
        tag = 'med_category'
        to_find = set(cat_code.split('|'))
    else:
        tag = 'med_category_name'
        to_find = set(cat_name.split('|'))

    res = ukb_medications[ukb_medications.apply(lambda r: len(to_find & set(r[tag].split('|'))) > 0, axis=1)]

    if return_codes:
        return sorted(set(res['coding']))
    else:
        return res


def get_per_med_group_subject_statistics(ukb_medications, ukb_phenotypes):
    # count number of UKB samples that have taken each category of medications on
    # their first and second visits

    all_cats = {}

    for _, r in ukb_medications.iterrows():
        c_ids = r['med_category'].split('|')
        c_names = r['med_category_name'].split('|')

        for c_id, c_name in zip(c_ids, c_names):
            if c_id not in all_cats:
                all_cats[c_id] = set()

            all_cats[c_id].add(c_name)

    for c_id in all_cats:
        all_cats[c_id] = '|'.join(sorted(all_cats[c_id]))

    echo('n all_cats=', len(all_cats))

    meds_all_cols = ukb_phenotypes[[SAMPLE_ID] + MEDICATIONS_FIRST_VISIT + MEDICATIONS_SECOND_VISIT]

    came_for_2nd_visit = ((~ukb_phenotypes['21003-0.0'].isnull()) & (~ukb_phenotypes['21003-1.0'].isnull()))

    echo('n second visit:', np.sum(came_for_2nd_visit))

    results = {'med_category': [],
               'med_category_name': [],
               'on_meds_1st_visit': [],
               'on_meds_2nd_visit': [],
               'on_meds_0': [],
               'on_meds_1': [],
               'on_meds_2': [],
               'on_meds_3': []
               }

    for cat_idx, cat in enumerate(sorted(all_cats)):

        if cat_idx % 50 == 0:
            echo('categories processed:', cat_idx)

        meds_in_cat = get_all_meds_in_category(ukb_medications, cat)
        medication_codes = set(meds_in_cat['coding'])

        fst = (meds_all_cols[MEDICATIONS_FIRST_VISIT].isin(medication_codes)).any(axis=1).astype(int)
        sec = (meds_all_cols[MEDICATIONS_SECOND_VISIT].isin(medication_codes)).any(axis=1).astype(int)

        results['on_meds_1st_visit'].append(np.sum(fst))
        results['on_meds_2nd_visit'].append(np.sum(sec))

        fst = fst[came_for_2nd_visit]
        sec = sec[came_for_2nd_visit]

        on_meds_dummies = pd.get_dummies(fst + 2 * sec, prefix='on_meds')

        for i in range(4):
            label = 'on_meds_' + str(i)
            n_subjects = np.sum(on_meds_dummies[label] if label in list(on_meds_dummies) else 0)
            results[label].append(n_subjects)

        results['med_category'].append(cat)
        results['med_category_name'].append(all_cats[cat])

    return pd.DataFrame(results).sort_values('on_meds_2', ascending=False)


def compute_PRS(common_variants_fname, phenotype_data, phenotype_name, exome_sample_ids, model_fname=None, return_model=False, prs_label_prefix=None):

    echo('Phenotype:', phenotype_name)
    echo('Reading common variants:', common_variants_fname)
    with open(common_variants_fname, 'rb') as in_f:
        common_variants_dict = pickle.load(in_f)
        common_variants_dict['genotypes'].rename(columns={'SAMPLE_ID': SAMPLE_ID}, inplace=True)
        common_variants = common_variants_dict['genotypes']

    echo('common variants:', len(list(common_variants)) - 1)

    sample_ids = sorted(list(set(phenotype_data[SAMPLE_ID]) & set(common_variants[SAMPLE_ID])))
    exome_sample_ids = sorted(set(sample_ids) & set(exome_sample_ids))

    some_second_visit_column = [c for c in list(phenotype_data) if '.2nd_visit.original.' in c][0]
    second_visit = phenotype_data[~phenotype_data[some_second_visit_column].isnull()]

    echo('samples from second visit:', len(second_visit))

    second_visit_sample_ids = sorted(second_visit[SAMPLE_ID])
    validation_sample_ids = sorted(set(exome_sample_ids) - set(second_visit_sample_ids))

    training_sample_ids = sorted(set(sample_ids) - set(second_visit_sample_ids) - set(exome_sample_ids))

    echo(len(training_sample_ids), 'training samples')
    echo(len(second_visit_sample_ids), 'excluded samples from second visit')
    echo(len(validation_sample_ids), 'additional samples for validation (exome samples)')

    echo('Casting X to float')
    X = pd.merge(pd.DataFrame({SAMPLE_ID: training_sample_ids}), common_variants)
    X['__CONST__'] = 1

    echo('Reordering Y')
    Y_order = list(X[SAMPLE_ID])
    Y = pd.merge(pd.DataFrame({SAMPLE_ID: Y_order}), phenotype_data)[phenotype_name].astype(float)

    echo(len(Y), 'data points')

    # X = pd.merge(pd.DataFrame({SAMPLE_ID : training_sample_ids}), X)[rsids].astype(float)
    X = X[[c for c in list(X) if c != SAMPLE_ID]]

    echo(len(list(X)), 'predictors')

    echo('Creating model')
    model = sm.OLS(Y, X)

    echo('Fitting model')
    res = model.fit()

    if prs_label_prefix is None:
        prs_label_prefix = phenotype_name

    prs_label = prs_label_prefix + '.PRS'
    prs_label_used_for_training = prs_label_prefix + '.used_for_PRS_training'

    prs_scores = pd.DataFrame({SAMPLE_ID: Y_order,
                               prs_label: res.fittedvalues,
                               prs_label_used_for_training: [True] * len(Y_order)})

    echo('\n' + str(res.summary())[:3000] + '\n...\n' + str(res.summary())[-3000:])

    echo('R^2 =', 1 - np.sum((res.fittedvalues - Y) ** 2) / np.sum((Y - np.mean(Y)) ** 2))
    # echo('R^2 =', np.sum((res - np.mean(Y)) ** 2) / np.sum((Y - np.mean(Y)) ** 2))

    echo('Pearson R=', str(scipy.stats.pearsonr(res.fittedvalues, Y)))
    echo('Spearman R=', str(scipy.stats.spearmanr(res.fittedvalues, Y)))

    echo('Computing PRS on validation set and second visit set')
    for sample_ids_to_test in [validation_sample_ids, second_visit_sample_ids]:
        res_test = test_prs(res, sample_ids_to_test, phenotype_data, common_variants, phenotype_name, prs_label)
        res_test[prs_label_used_for_training] = False
        prs_scores = pd.concat([prs_scores, res_test], ignore_index=True)

    echo('Merging PRS scores with the original phenotype data')
    prs_scores = pd.merge(phenotype_data, prs_scores, on=SAMPLE_ID, how='left')

    echo('PRS scores computed for',
         len(prs_scores[~prs_scores[prs_label].isnull()]),
         'out of',
         len(prs_scores),
         'samples')

    if model_fname is not None:
        echo('Saving the model to:', model_fname)
        res.save(model_fname, remove_data=True)

    if return_model:
        return prs_scores, res
    else:
        return prs_scores


def test_prs(prs, sample_ids, phenotype_data, common_variants, ph_name, prs_label=None):

    echo('Phenotype:', ph_name)
    sample_ids = sorted(set(sample_ids) & set(common_variants[SAMPLE_ID]) & set(phenotype_data[SAMPLE_ID]))
    echo('Samples with common variants and phenotype values:', len(sample_ids))

    echo('Selecting sample_ids from common variants')
    X = pd.merge(pd.DataFrame({SAMPLE_ID: sample_ids}), common_variants)
    X['__CONST__'] = 1

    echo(len(X), 'samples with', len(list(X)) - 1, 'predictors')

    echo('Reordering Y')
    Y_order = list(X[SAMPLE_ID])
    Y = pd.merge(pd.DataFrame({SAMPLE_ID: Y_order}), phenotype_data)[ph_name].astype(float)

    echo(len(Y), 'data points')
    X = X[[c for c in list(X) if c != SAMPLE_ID]]
    echo('Computing prediction')
    res = prs.predict(X)

    echo('R^2 =', 1 - np.sum((res - Y) ** 2) / np.sum((Y - np.mean(Y)) ** 2))

    echo('Pearson R=', str(scipy.stats.pearsonr(res, Y)))
    echo('Spearman R=', str(scipy.stats.spearmanr(res, Y)))

    if prs_label is None:
        prs_label = ph_name

    result = pd.DataFrame({SAMPLE_ID: Y_order,
                           prs_label: res})
    return result

def load_bgen(chrom, load_snp_info=True, bgen_data_chrom_to_fname_mapping=None, verbose=True):

    if bgen_data_chrom_to_fname_mapping is not None:
        bgen_fname, sample_fname, snp_info_fname = bgen_data_chrom_to_fname_mapping[chrom]
    else:
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom

        bgen_fname = BGEN_DIR + f'ukb_imp_{chrom}_v3.bgen'

        if chrom == 'chrX':
            sample_fname = BGEN_DIR + f'ukb33751_imp_chrX_v3_s486669.sample'
        else:
            sample_fname = BGEN_DIR + f'ukb33751_imp_{chrom}_v3_s487320.sample'

        snp_info_fname = BGEN_DIR + f'ukb_mfi_{chrom}_v3.txt.gz'

    if verbose:
        echo('Loading snp_info:', bgen_fname, sample_fname, snp_info_fname)

    bgenf = BgenFile(bgen_fname, sample_fname, delay_parsing=True)

    snp_array = {}
    for i, rs in enumerate(bgenf.rsids()):
        if rs not in snp_array:
            snp_array[rs] = []
        snp_array[rs].append(i)

    snp_info = None
    if load_snp_info:
        if snp_info_fname.endswith('.pickle'):
            snp_info = pd.read_pickle(snp_info_fname)
        else:
            snp_info = pd.read_csv(snp_info_fname,
                                   sep='\t',
                                   names=['varid', VCF_RSID, VCF_POS, VCF_REF, VCF_ALT, 'MAF', 'minor_allele', 'info_score'],
                                   dtype={'MAF': float, 'info_score': float})

            snp_info[VCF_CHROM] = chrom.replace('chr', '')
            snp_info[VCF_AF] = np.where(snp_info[VCF_ALT] == snp_info['minor_allele'], snp_info['MAF'], 1 - snp_info['MAF'])
            snp_info[VARIANT_IDX] = list(range(len(snp_info)))

    return bgenf, snp_array, snp_info


def get_dosages_from_bgen(bgenf, bgenf_snps_array, rsid, samples_idx=None, samples=None, return_samples_order=False, extract_minor_allele_dosage=False):

    variants = {}
    samples_order = None

    if samples is not None:

        samples_set = set(samples) & set(bgenf.samples)

        samples_idx = [s in samples_set for s in bgenf.samples]
        samples_bgen_idx = dict((s, i) for i, s in enumerate(bgenf.samples))

        samples_order = sorted(samples_set, key=lambda s: samples_bgen_idx[s])

    genotypes = np.array([0, 1, 2])

    for var_idx in bgenf_snps_array.get(rsid, []):
        v = bgenf[var_idx]

        varid = re.sub(r'^0*', '', v.chrom) + ':' + str(v.pos) + ':' + ':'.join(v.alleles)

        if extract_minor_allele_dosage:
            variants[varid] = v.minor_allele_dosage
        else:
            variants[varid] = np.dot(v.probabilities, genotypes)

        if samples_idx is not None:
            variants[varid] = variants[varid][samples_idx]

    if return_samples_order:

        if samples_order is None:
            samples_order = list(bgenf.samples)

        return variants, samples_order
    else:
        return variants


def get_latest_covid19_results(all_samples, covid19_fname=None):

    if covid19_fname is None:
        covid19_fname = ROOT_PATH + '/pfiziev/ukbiobank/data/covid19/covid19_result.txt'

    echo('Reading:', covid19_fname)
    covid19_result = pd.read_csv(covid19_fname,
                                 sep='\t', dtype={'eid': str}).rename(columns={'eid': SAMPLE_ID})

    covid19_pos = set(covid19_result[(covid19_result['result'] == 1) & (covid19_result['origin'] == 1)][SAMPLE_ID])
    covid19_neg = set(covid19_result[SAMPLE_ID]) - set(covid19_result[covid19_result['result'] == 1][SAMPLE_ID])
    covid19_pos_non_critical = set(covid19_result[covid19_result['result'] == 1][SAMPLE_ID]) - set(covid19_pos)

    covid19_pos_all = covid19_pos | covid19_pos_non_critical

    covid19_tested = covid19_pos_all | covid19_neg

    echo('covid19_tested:', len(covid19_tested),
         ', covid19_pos_all:', len(covid19_pos_all),
         ', covid19_pos_critical:', len(covid19_pos),
         ', covid19_pos_non_critical:', len(covid19_pos_non_critical),
         ', covid19_neg:', len(covid19_neg),
         ', covid19_unknown:', len(set(all_samples) - set(covid19_tested))
         )


    return pd.DataFrame({SAMPLE_ID: all_samples,

                         'covid19_tested': [int(s in covid19_tested) for s in all_samples],

                         'covid19_pos_all': [int(s in covid19_pos_all) for s in all_samples],
                         'covid19_pos_all_non_neg': [int(s in covid19_pos_all) if s not in covid19_neg else None for s in all_samples],

                         'covid19_pos_critical': [int(s in covid19_pos) for s in all_samples],
                         'covid19_pos_critical_non_neg': [int(s in covid19_pos) if s not in covid19_neg else None for s in all_samples],

                         'covid19_pos_non_critical': [int(s in covid19_pos_non_critical) for s in all_samples],
                         'covid19_pos_non_critical_non_neg': [int(s in covid19_pos_non_critical) if s not in covid19_neg else None for s in all_samples],

                         'covid19_pos_neg': [int(s in covid19_neg) for s in all_samples],
                         'covid19_pos_neg_non_pos': [int(s in covid19_neg) if s not in covid19_pos_all else None for s in all_samples]

                         })


def get_dosages_for_variants(variants,
                             bgen_data=None,
                             sample_ids=None,
                             rsid_label=VCF_RSID,
                             return_bgen_data=False,
                             extract_minor_allele_dosage=False,
                             bgen_data_chrom_to_fname_mapping=None,
                             verbose=True,
                             skip_errors=False):

    if verbose:
        echo('Getting dosages for variants:', len(variants), '!')

    genotypes = {}
    missing = 0

    samples_order_dict = {}
    return_samples_order = (sample_ids is not None)
    for var_idx, (_, var_info) in enumerate(variants.iterrows()):
        if var_idx % 100 == 0 and verbose:
            echo(var_idx, 'variants processed')

        rsid = var_info[rsid_label]
        chrom = var_info[VCF_CHROM]
        pos = var_info[VCF_POS]
        ref = var_info[VCF_REF]
        alt = var_info[VCF_ALT]

        if bgen_data is None:
            bgen_data = {}

        if chrom not in bgen_data:
            if verbose:
                echo('Loading bgen data for:', chrom)
            bgenf, bgenf_snps_array, _ = load_bgen(chrom, load_snp_info=False, bgen_data_chrom_to_fname_mapping=bgen_data_chrom_to_fname_mapping, verbose=verbose)
            bgen_data[chrom] = (bgenf, bgenf_snps_array)

        if len(bgen_data[chrom]) == 3:
            # this is in case bgen_data was passed from outside
            bgenf, bgenf_snps_array, _ = bgen_data[chrom]
        else:
            bgenf, bgenf_snps_array = bgen_data[chrom]

        if chrom not in samples_order_dict:
            if (bgen_data_chrom_to_fname_mapping is not None and
                bgen_data_chrom_to_fname_mapping.get(SPLIT_SAMPLE_IDS_BY_UNDERSCORE, False)):

                real_sample_ids = [s.split('_')[0] for s in bgenf.samples]
            else:
                real_sample_ids = list(bgenf.samples)

            samples_order_dict[chrom] = real_sample_ids

        _res = get_dosages_from_bgen( bgenf,
                                      bgenf_snps_array,
                                      rsid,
                                      samples_idx=None,
                                      samples=sample_ids,
                                      return_samples_order=return_samples_order,
                                      extract_minor_allele_dosage=extract_minor_allele_dosage)

        if return_samples_order:
            gt, samples_order = _res
        else:
            gt = _res
            samples_order = samples_order_dict[chrom]

        if len(gt) == 0:
            missing += 1
            continue

        if SAMPLE_ID not in genotypes:
            genotypes[SAMPLE_ID] = samples_order

        if not all(s1 == s2 for s1, s2 in zip(samples_order, genotypes[SAMPLE_ID])) or len(genotypes[SAMPLE_ID]) != len(
                samples_order):
            echo('ERROR: genotype arrays do not match', var_info)
            break

        if VARID_REF_ALT in list(var_info.keys()):
            key = var_info[VARID_REF_ALT]
        else:
            key = ':'.join([chrom, str(pos), ref, alt])

        if key not in gt:
            echo('ERROR:', key, 'not found in', list(gt.keys()), var_info)

            if not skip_errors:
                break
            else:
                key = list(gt.keys())[0]
                echo('Returning the first found key:', key)

        genotypes[key] = gt[key]

    if verbose:
        echo('Missing variants:', missing)

    common_variants = pd.DataFrame(genotypes)

    if return_bgen_data:
        return common_variants, bgen_data

    return common_variants

def get_missing_rsids(gwas_variants):
    res = None
    for chrom in sorted(set(gwas_variants[VCF_CHROM])):

        dbsnp_fname = ROOT_PATH + f'/pfiziev/dbsnp/hg19/GCF_000001405.25.chr{chrom}.pickle'
        if not os.path.exists(dbsnp_fname):
            echo('File not found, skipping:', dbsnp_fname)
            continue

        echo('Reading dbsnp info:', dbsnp_fname)
        dbsnp = pd.read_pickle(dbsnp_fname)

        gwas_variants_chrom = gwas_variants[gwas_variants[VCF_CHROM] == chrom]

        echo('gwas_variants_chrom:', gwas_variants_chrom.shape)
        gwas_variants_chrom_with_rsid = gwas_variants_chrom[gwas_variants_chrom[VCF_RSID].str.startswith('rs')].copy()

        echo('gwas_variants_chrom_with_rsid:', gwas_variants_chrom_with_rsid.shape)
        gwas_variants_chrom_with_rsid['all_RSIDs'] = gwas_variants_chrom_with_rsid[VCF_RSID]

        gwas_variants_chrom_without_rsid = gwas_variants_chrom[~gwas_variants_chrom[VCF_RSID].str.startswith('rs')]

        echo('gwas_variants_chrom_without_rsid:', len(gwas_variants_chrom_without_rsid))

        d = pd.merge(gwas_variants_chrom_without_rsid,
                     dbsnp,
                     on=[VCF_CHROM, VCF_POS],
                     suffixes=['', '_dbsnp'],
                     how='left')

        def consolidate_rsids(x):
            if consolidate_rsids.cnt % 10000 == 0:
                echo(consolidate_rsids.cnt, 'variants processed')

            snp_info = x.iloc[0].copy()

            exact_matches = x[(x[VCF_REF] == x[VCF_REF + '_dbsnp']) &
                              (x[VCF_ALT] == x[VCF_ALT + '_dbsnp'])]

            ukb_rsid = snp_info[VCF_RSID]

            if ukb_rsid.startswith('rs'):
                all_rsids = [ukb_rsid]
            else:
                if len(exact_matches) > 0:
                    all_rsids = [rsid for rsid in sorted(set(exact_matches['RSID_dbsnp'])) if
                                 not (rsid is None or type(rsid) is float and np.isnan(rsid))]
                else:
                    all_rsids = ['ref_alt_mismatch/' + rsid for rsid in sorted(set(x['RSID_dbsnp'])) if
                                 not (rsid is None or type(rsid) is float and np.isnan(rsid))]

            snp_info['all_RSIDs'] = ','.join(all_rsids)

            consolidate_rsids.cnt += 1

            return snp_info

        consolidate_rsids.cnt = 0

        gwas_variants_chrom_without_rsid = d.groupby('varid_ref_alt').apply(consolidate_rsids)
        gwas_variants_chrom_without_rsid = gwas_variants_chrom_without_rsid[list(gwas_variants_chrom_with_rsid)]
        gwas_variants_chrom_with_rsid = pd.concat([gwas_variants_chrom_with_rsid,
                                                   gwas_variants_chrom_without_rsid],
                                                  ignore_index=True).sort_values(VCF_POS)

        echo('final gwas_variants_chrom_with_rsid:', gwas_variants_chrom_with_rsid.shape)

        if res is None:
            res = gwas_variants_chrom_with_rsid
        else:
            res = pd.concat([res, gwas_variants_chrom_with_rsid], ignore_index=True)

    gwas_variants = res
    echo('gwas_variants:', gwas_variants.shape)

    return gwas_variants


def correct_genotypes_for_genetic_PCs(variants, gPC_data, plot_figures=True, verbose=True, set_sample_ids_as_index=True, batch_size=1000):

    if verbose:
        echo('Correcting genotypes for genetic principal components. Variants shape:', variants.shape,
             ', gPC_data shape:', gPC_data.shape, ', batch_size:', batch_size)

    rsids = [c for c in list(variants) if c != SAMPLE_ID]

    gPC_labels = [c for c in list(gPC_data) if c != SAMPLE_ID]

    final_result = None

    for batch_idx, batch_rsids in enumerate(batchify(rsids, batch_size)):

        d = pd.merge(variants[[SAMPLE_ID] + batch_rsids], gPC_data, on=SAMPLE_ID)
        if verbose:
            echo('batch_idx:', batch_idx, 'after merging with gPC_data:', d.shape, ', nan values:',  d.isnull().any(axis=None))

        d['__CONST__'] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            covariates = gPC_labels + ['__CONST__']

            skm = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(d[covariates], d[batch_rsids])

            predictions = skm.predict(d[covariates])
            resid = d[batch_rsids] - predictions
            res = pd.DataFrame(resid, columns=batch_rsids)

            res[SAMPLE_ID] = d[SAMPLE_ID]

            if plot_figures:
                regr_stats = pd.DataFrame(skm.coef_, columns=covariates)
                regr_stats[VCF_RSID] = batch_rsids
                regr_stats['r2'] = sklearn.metrics.r2_score(d[batch_rsids], predictions, multioutput='raw_values')
                regr_stats = regr_stats[[VCF_RSID, 'r2'] + covariates]

                plt.figure()
                sns.distplot(regr_stats['r2'])
                plt.show()

        if verbose:
            echo(res.shape)

        if final_result is None:
            final_result = res
        else:
            final_result = pd.merge(final_result, res, on=SAMPLE_ID)

        gc.collect()

    if set_sample_ids_as_index:
        final_result = final_result.set_index(SAMPLE_ID)

    echo('final_result:', final_result.shape)

    return final_result


def find_interactions1(snp_genotypes, ph_data, ph_name_label):

    echo('Merging genotypes and ph_data')
    d = pd.merge(ph_data[[SAMPLE_ID, ph_name_label]].set_index(SAMPLE_ID),
                 snp_genotypes,
                 left_index=True,
                 right_index=True).copy()

    variants = [v for v in list(snp_genotypes) if v != SAMPLE_ID]

    echo('Regressing out main effects of', len(variants), 'variants!')

    orthogonized = {ph_name_label + '/corr' : list(get_residuals(d,
                                                                 ph_name_label,
                                                                 variants,
                                                                 interactions=False,
                                                                 make_sample_ids_index=False,
                                                                 verbose=True)[ph_name_label])}

    int_labels = []
    for v_idx, v1 in enumerate(variants):
        for v2_idx in range(v_idx + 1, len(variants)):
            v2 = variants[v2_idx]
            key = v1 + ' X ' + v2
            d2 = pd.DataFrame({v1: d[v1], v2: d[v2], key: d[v1] * d[v2], CONST_LABEL: 1})
            orthogonized[key] = list(get_residuals(d2, [key], [v1, v2, CONST_LABEL], interactions=False, make_sample_ids_index=False, verbose=False)[key])
            int_labels.append(key)

    orthogonized = pd.DataFrame(orthogonized)

    corr, pval = vcorrcoef(orthogonized[[ph_name_label + '/corr']], orthogonized[int_labels], axis='columns')

    return pd.DataFrame({ 'var1': list(map(lambda x: x.split(' X ')[0], corr.index)),
                          'var2': list(map(lambda x: x.split(' X ')[1], corr.index)),
                          'corr': list(corr[ph_name_label + '/corr']),
                          'pvalue': list(pval[ph_name_label + '/corr'])}).sort_values('pvalue')


def find_interactions2(snp_genotypes, ph_data, ph_name_label, perc=10, N_SHUFFLINGS=0):

    echo('Computing interaction statistics')

    variants = [v for v in list(snp_genotypes) if v != SAMPLE_ID]
    echo('Variants:', len(variants))

    if SAMPLE_ID not in list(snp_genotypes):
        snp_genotypes = snp_genotypes.reset_index().rename(columns={'index': SAMPLE_ID})

    regr_data = pd.merge(ph_data[[SAMPLE_ID, ph_name_label]],
                         snp_genotypes,
                         on=SAMPLE_ID).dropna().copy()

    regr_data[CONST_LABEL] = 1

    sorted_values = sorted(regr_data[ph_name_label])

    bottom_perc_value = sorted_values[int(perc * len(regr_data) / 100)]
    top_perc_value = sorted_values[int((100 - perc) * len(regr_data) / 100)]

    case_ctrls = regr_data[(regr_data[ph_name_label] >= top_perc_value) | (regr_data[ph_name_label] <= bottom_perc_value)].copy()
    case_ctrls['high_ph'] = (case_ctrls[ph_name_label] >= top_perc_value).astype(int)

    echo(ph_name_label, ', variants=', len(variants),
         ', bottom/top=', bottom_perc_value, '/', top_perc_value,
         ', case_ctrl=', case_ctrls.shape,
         ', regr_data=', regr_data.shape)

    n_cases = np.sum(case_ctrls['high_ph'])
    n_ctrls = len(case_ctrls) - n_cases

    real_diffs = None
    shuffled_diffs = {}

    for rand_idx in range(1 + N_SHUFFLINGS):

        if rand_idx % 10 == 0:
            echo('Batch:', rand_idx)

        cases = case_ctrls[case_ctrls['high_ph'] == 1]
        ctrls = case_ctrls[case_ctrls['high_ph'] == 0]

        case_corr, _ = vcorrcoef(cases[variants], axis='columns')
        ctrl_corr, _ = vcorrcoef(ctrls[variants], axis='columns')

        c_diffs = np.abs((case_corr - ctrl_corr))

        if real_diffs is None:
            real_diffs = {}

            for v1 in list(c_diffs):

                if v1 not in real_diffs:
                    real_diffs[v1] = {}

                for v2 in list(c_diffs):
                    real_diffs[v1][v2] = c_diffs.loc[v1][v2]

        for v1 in list(c_diffs):

            if v1 not in shuffled_diffs:
                shuffled_diffs[v1] = {}

            for v2 in list(c_diffs):

                if v2 not in shuffled_diffs[v1]:
                    shuffled_diffs[v1][v2] = []

                shuffled_diffs[v1][v2].append(c_diffs.loc[v1][v2])

        cc_status = list(case_ctrls['high_ph'])

        random.shuffle(cc_status)

        case_ctrls['high_ph'] = cc_status

    all_shuffled_diffs = sorted([diff for v1_idx, v1 in enumerate(variants)
                                        for v2 in variants[v1_idx + 1:]
                                            for diff in shuffled_diffs[v1][v2]],
                                 reverse=True)

    echo('shuffled diffs:', len(all_shuffled_diffs))

    n = 0

    real_diffs_overall_fdr = {}
    real_diffs_individual_fdr = {}

    res = {'var1': [], 'var2': [], 'r_diff': [], 'gaussian_pvalue': []}

    if N_SHUFFLINGS > 0:
        res['fdr_overall'] = []
        res['fdr_individual'] = []

    sd_tot = math.sqrt(1 / (n_ctrls - 1) + 1 / (n_cases - 1))

    echo('sd_tot:', sd_tot)
    for v1, v2 in sorted([(v1, v2) for v1_idx, v1 in enumerate(variants) for v2 in variants[v1_idx + 1:]],
                         key=lambda k: real_diffs[k[0]][k[1]],
                         reverse=True):

        diff = real_diffs[v1][v2]

        while n < len(all_shuffled_diffs) and all_shuffled_diffs[n] >= diff:
            n += 1

        if v1 not in real_diffs_overall_fdr:
            real_diffs_overall_fdr[v1] = {}
            real_diffs_individual_fdr[v1] = {}

        res['var1'].append(v1)
        res['var2'].append(v2)

        res['r_diff'].append(diff)

        if N_SHUFFLINGS > 0:
            res['fdr_overall'].append(n / len(all_shuffled_diffs))
            res['fdr_individual'].append(sum([shuff_diff >= diff for shuff_diff in shuffled_diffs[v1][v2]]) / len(shuffled_diffs[v1][v2]))

        test_stat = diff / sd_tot

        gaussian_pvalue = 2 * scipy.stats.norm.sf(test_stat, 0, 1)

        res['gaussian_pvalue'].append(gaussian_pvalue)

    _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(res['gaussian_pvalue'])

    res['gaussian_fdr'] = fdr_corrected_pvalues
    res = pd.DataFrame(res).sort_values('gaussian_pvalue')

    return res


def find_interactions(snp_genotypes, ph_data, ph_name_label, gPC_data=None, perc=10, min_af_in_case_ctrls=0.01):

    variants = [v for v in list(snp_genotypes) if v != SAMPLE_ID]
    echo('Computing interactions statistics, variants:', len(variants))

    original_snp_genotypes = snp_genotypes
    if gPC_data is not None:
        echo('Correcting for genetic PCs')
        snp_genotypes = correct_genotypes_for_genetic_PCs(snp_genotypes, gPC_data, plot_figures=False)

    if SAMPLE_ID not in list(snp_genotypes):
        snp_genotypes = snp_genotypes.reset_index().rename(columns={'index': SAMPLE_ID})

    regr_data = pd.merge(ph_data[[SAMPLE_ID, ph_name_label]],
                         snp_genotypes,
                         on=SAMPLE_ID).dropna().copy()

    sorted_values = sorted(regr_data[ph_name_label])

    bottom_perc_value = sorted_values[int(perc * len(regr_data) / 100)]
    top_perc_value = sorted_values[int((100 - perc) * len(regr_data) / 100)]

    case_ctrls = regr_data[(regr_data[ph_name_label] >= top_perc_value) | (regr_data[ph_name_label] <= bottom_perc_value)].copy()
    case_ctrls['high_ph'] = (case_ctrls[ph_name_label] >= top_perc_value).astype(int)

    echo(ph_name_label, ', variants=', len(variants),
         ', bottom/top=', bottom_perc_value, '/', top_perc_value,
         ', case_ctrl=', case_ctrls.shape,
         ', regr_data=', regr_data.shape)

    n_cases = np.sum(case_ctrls['high_ph'])
    n_ctrls = len(case_ctrls) - n_cases

    cases = case_ctrls[case_ctrls['high_ph'] == 1]
    ctrls = case_ctrls[case_ctrls['high_ph'] == 0]

    cases_original = pd.merge(original_snp_genotypes, case_ctrls[case_ctrls['high_ph'] == 1][[SAMPLE_ID]])
    ctrls_original = pd.merge(original_snp_genotypes, case_ctrls[case_ctrls['high_ph'] == 0][[SAMPLE_ID]])

    echo('cases_original/ctrls_original:', cases_original.shape, ctrls_original.shape)

    all_mafs = original_snp_genotypes[variants].sum(axis=0) / (2 * len(original_snp_genotypes))
    cases_mafs = cases_original[variants].sum(axis=0) / (2 * len(cases_original))
    ctrls_mafs = ctrls_original[variants].sum(axis=0) / (2 * len(ctrls_original))

    mafs = pd.DataFrame({'all': all_mafs,
                         'cases': cases_mafs,
                         'ctrls': ctrls_mafs}).reset_index().rename(columns={'index': 'variant'})
    for k in ['all', 'cases', 'ctrls']:
        mafs[k] = np.where(mafs[k] < 0.5, mafs[k], 1 - mafs[k])

    variants = sorted(mafs[(mafs['all'] >= min_af_in_case_ctrls) &
                           (mafs['cases'] >= min_af_in_case_ctrls) &
                           (mafs['ctrls'] >= min_af_in_case_ctrls)]['variant'])

    echo('Variants with MAF in cases and controls >=', min_af_in_case_ctrls, ':', len(variants))

    echo('Computing correlations in cases')
    case_corr, _ = vcorrcoef(cases[variants], axis='columns')

    echo('Computing correlations in controls')
    ctrl_corr, _ = vcorrcoef(ctrls[variants], axis='columns')

    echo('Computing correlation differences')
    c_diffs = np.abs((case_corr - ctrl_corr))

    sd_tot = math.sqrt(1 / (n_ctrls - 1) + 1 / (n_cases - 1))

    echo('sd_tot=', sd_tot)

    c_diffs = c_diffs.reset_index().melt(id_vars='index').rename(columns={'index': 'var1', 'variable': 'var2', 'value': 'r_diff'})

    echo('Filtering non-redundant pairs')

    to_keep = {'var1': [], 'var2': []}
    for v1_idx, v1 in enumerate(variants):
        for v2 in variants[v1_idx + 1:]:
            to_keep['var1'].append(v1)
            to_keep['var2'].append(v2)

    to_keep = pd.DataFrame(to_keep)

    c_diffs = pd.merge(c_diffs, to_keep, on=['var1', 'var2'])

    echo('Non-redundant pairs:', len(c_diffs))
    echo('Computing Gaussian p-values')
    c_diffs['gaussian_pvalue'] = 2 * scipy.stats.norm.sf(c_diffs['r_diff'] / sd_tot, 0, 1)

    echo('Computing FDR')
    _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(c_diffs['gaussian_pvalue'])

    c_diffs['gaussian_fdr'] = fdr_corrected_pvalues
    c_diffs = c_diffs.sort_values('r_diff', ascending=False)

    return c_diffs


def find_interactions_batches(batch1_idx,
                              batch2_idx,

                              batch1_snp_genotypes,
                              batch2_snp_genotypes,

                              cache,

                              ph_data,
                              ph_name_label,

                              gPC_data=None,

                              perc=10,
                              min_af_in_case_ctrls=0.01,
                              compute_fdr=True):

    echo('Computing interactions statistics')

    case_ctrls_batch = []
    variants_batch = []

    for bidx, bgens in [(batch1_idx, batch1_snp_genotypes),
                        (batch2_idx, batch2_snp_genotypes)]:

        if bidx not in cache:

            original_snp_genotypes = bgens
            bvars = [c for c in list(bgens) if c != SAMPLE_ID]

            bgens_corrected = correct_genotypes_for_genetic_PCs(bgens,
                                                                gPC_data,
                                                                plot_figures=False,
                                                                set_sample_ids_as_index=False)

            regr_data = pd.merge(ph_data[[SAMPLE_ID, ph_name_label]],
                                 bgens_corrected,
                                 on=SAMPLE_ID).dropna()

            sorted_values = sorted(regr_data[ph_name_label])

            bottom_perc_value = sorted_values[int(perc * len(regr_data) / 100)]
            top_perc_value = sorted_values[int((100 - perc) * len(regr_data) / 100)]

            case_ctrls = regr_data[(regr_data[ph_name_label] >= top_perc_value) | (regr_data[ph_name_label] <= bottom_perc_value)].copy()
            case_ctrls['high_ph'] = (case_ctrls[ph_name_label] >= top_perc_value).astype(int)

            echo(ph_name_label, ', variants=', len(bvars),
                 ', bottom/top=', bottom_perc_value, '/', top_perc_value,
                 ', case_ctrl=', case_ctrls.shape,
                 ', regr_data=', regr_data.shape)

            cases_original = pd.merge(original_snp_genotypes, case_ctrls[case_ctrls['high_ph'] == 1][[SAMPLE_ID]])
            ctrls_original = pd.merge(original_snp_genotypes, case_ctrls[case_ctrls['high_ph'] == 0][[SAMPLE_ID]])

            echo('cases_original/ctrls_original:', cases_original.shape, ctrls_original.shape)

            all_mafs = original_snp_genotypes[bvars].sum(axis=0) / (2 * len(original_snp_genotypes))
            cases_mafs = cases_original[bvars].sum(axis=0) / (2 * len(cases_original))
            ctrls_mafs = ctrls_original[bvars].sum(axis=0) / (2 * len(ctrls_original))

            mafs = pd.DataFrame({'all': all_mafs,
                                 'cases': cases_mafs,
                                 'ctrls': ctrls_mafs}).reset_index().rename(columns={'index': 'variant'})

            for k in ['all', 'cases', 'ctrls']:
                mafs[k] = np.where(mafs[k] < 0.5, mafs[k], 1 - mafs[k])

            final_variants = sorted(mafs[(mafs['all'] >= min_af_in_case_ctrls) &
                                         (mafs['cases'] >= min_af_in_case_ctrls) &
                                         (mafs['ctrls'] >= min_af_in_case_ctrls)]['variant'])

            cache[bidx] = (case_ctrls, final_variants)

        cc, v = cache[bidx]
        case_ctrls_batch.append(cc)
        variants_batch.append(v)

    if len(variants_batch[0]) == 0 or len(variants_batch[1]) == 0:
        echo('Empty batch:', len(variants_batch[0]), len(variants_batch[1]))
        return pd.DataFrame({'var1':[], 'var2': [], 'r_diff': [], 'gaussian_pvalue': []})

    cases = [case_ctrls[case_ctrls['high_ph'] == 1] for case_ctrls in case_ctrls_batch]
    ctrls = [case_ctrls[case_ctrls['high_ph'] == 0] for case_ctrls in case_ctrls_batch]

    echo('cases:', len(cases[0]), len(cases[1]))
    echo('ctrls:', len(ctrls[0]), len(ctrls[1]))
    echo('variants:', len(variants_batch[0]), len(variants_batch[1]))

    n_cases = len(cases[0])
    n_ctrls = len(ctrls[0])

    echo('Computing correlations in cases')
    case_corr, _ = vcorrcoef(cases[0][variants_batch[0]], cases[1][variants_batch[1]], axis='columns')
    echo('case_corr:', case_corr.shape)

    echo('Computing correlations in controls')
    ctrl_corr, _ = vcorrcoef(ctrls[0][variants_batch[0]], ctrls[1][variants_batch[1]], axis='columns')
    echo('ctrl_corr:', ctrl_corr.shape)

    echo('Computing correlation differences')
    c_diffs = np.abs((case_corr - ctrl_corr))
    echo('c_diffs:', c_diffs.shape)

    echo('batch 1 variants:\n', list(c_diffs)[:10], '\n', variants_batch[0][:10])
    echo('batch 2 variants:\n', list(c_diffs.index)[:10], '\n', variants_batch[1][:10])

    sd_tot = math.sqrt(1 / (n_ctrls - 1) + 1 / (n_cases - 1))

    echo('sd_tot=', sd_tot)

    c_diffs = c_diffs.T.reset_index().melt(id_vars='index').rename(columns={'index': 'var1', 'variable': 'var2', 'value': 'r_diff'})

    echo('c_diffs:', c_diffs.shape)
    echo('Filtering non-redundant pairs')

    to_keep = {'var1': [], 'var2': []}

    seen = set()

    for v1 in variants_batch[0]:
        for v2 in variants_batch[1]:

            if v1 != v2 and (v2, v1) not in seen and (v1, v2) not in seen:

                to_keep['var1'].append(v1)
                to_keep['var2'].append(v2)

                seen.add((v1, v2))
                seen.add((v2, v1))

    to_keep = pd.DataFrame(to_keep)
    echo('to_keep:', to_keep.shape)

    c_diffs = pd.merge(c_diffs, to_keep, on=['var1', 'var2'])

    echo('kept diffs:', c_diffs.shape)

    if len(c_diffs) == 0:
        c_diffs['gaussian_pvalue'] = []

        if compute_fdr:
            c_diffs['gaussian_fdr'] = []

    else:
        echo('Non-redundant pairs:', len(c_diffs))
        echo('Computing Gaussian p-values')
        c_diffs['gaussian_pvalue'] = 2 * scipy.stats.norm.sf(c_diffs['r_diff'] / sd_tot, 0, 1)

        if compute_fdr:
            echo('Computing FDR')
            _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(c_diffs['gaussian_pvalue'])
            c_diffs['gaussian_fdr'] = fdr_corrected_pvalues

        c_diffs = c_diffs.sort_values('r_diff', ascending=False)

    return c_diffs


def _find_interactions_batches(batch1_idx,
                              batch2_idx,

                              batch1_snp_genotypes,
                              batch2_snp_genotypes,

                              cache,

                              ph_data,
                              ph_name_label,

                              gPC_data=None):

    echo('Computing interactions statistics')

    variants = []

    for bidx, bgens in [(batch1_idx, batch1_snp_genotypes),
                        (batch2_idx, batch2_snp_genotypes)]:

        if bidx not in cache:

            bgens_corrected = correct_genotypes_for_genetic_PCs(bgens,
                                                                gPC_data,
                                                                plot_figures=False,
                                                                set_sample_ids_as_index=False)

            regr_data = pd.merge(ph_data[[SAMPLE_ID, ph_name_label]],
                                 bgens_corrected,
                                 on=SAMPLE_ID).dropna()

            cache[bidx] = regr_data

        regr_data = cache[bidx]
        variants.append(regr_data)

    vars_batch_1 = variants[0]
    vars_batch_2 = variants[1]

    varids_batch_1 = [c for c in list(vars_batch_1) if c not in [SAMPLE_ID, ph_name_label]]
    varids_batch_2 = [c for c in list(vars_batch_2) if c not in [SAMPLE_ID, ph_name_label]]

    interactions = {'var1': [],
                    'var2': [],
                    'r': [],
                    'pvalue': []}

    int_df = {}
    seen = set()
    int_keys = []
    for v1 in varids_batch_1:

        for v2 in varids_batch_2:
            if (v1, v2) in seen or (v2, v1) in seen or v1 == v2:
                continue

            seen.add((v1, v2))
            seen.add((v2, v1))

            int_key = v1 + ' X ' + v2
            int_df[int_key] = vars_batch_1[v1] * vars_batch_2[v2]

            int_keys.append(int_key)

    int_df[ph_name_label] = vars_batch_1[ph_name_label]
    int_df = pd.DataFrame(int_df)

    corr, pval = vcorrcoef(int_df[[ph_name_label]], int_df[int_keys], axis='columns')

    return corr, pval


def get_ukb_exome_variants_for_genes(genes,
                                     ukb_path=None,
                                     fname_template='/exomes/26_OCT_2020/ukb_exomes.26_OCT_2020.chr%s.full_info.with_gnomad_coverage.pickle',
                                     verbose=True):
    if verbose:
        echo('Genes:', len(genes))

    gene_vars = None
    if ukb_path is None:
        ukb_path = UKB_DATA_PATH + fname_template

    for chrom in sorted(set(genes[VCF_CHROM])):

        if verbose:
            echo('Reading chromosome:', chrom, 'from:', ukb_path % chrom)

        chr_vars = pd.read_pickle(ukb_path % chrom)

        chr_gene_vars = chr_vars[chr_vars[GENE_NAME].isin(set(genes[GENE_NAME]))]

        if gene_vars is None:
            gene_vars = chr_gene_vars
        else:
            gene_vars = pd.concat([gene_vars, chr_gene_vars], ignore_index=True)

    if verbose:
        echo('Total variants:', len(gene_vars), ', from', len(set(gene_vars[GENE_NAME])), 'genes')

    return gene_vars


def get_gene_del_variant_stats(gene_names,
                               max_af=None,
                               max_ac=None,
                               ukb200k_db = ROOT_PATH + '/ukbiobank/data/exomes/26_OCT_2020/ukb.exome.db',
                               batch_size=100,
                               samples_to_subset=None):

    gene_stats = {GENE_NAME: [], 'n_variants': [], 'n_carriers': []}

    import sqlite3
    con = sqlite3.connect(ukb200k_db)

    for b_idx, batch_genes in enumerate(batchify(gene_names, batch_size=batch_size)):
        echo(b_idx * batch_size, 'genes processed')

        d = pd.read_sql('select * from variants where symbol IN (%s)' % ', '.join(['"' + g + '"' for g in batch_genes]), con)

        # Be sure to close the connection

        def from_bytes(byte_ids, bit_len=3):
            ''' convert byte sequence to list of ints from 24-bit encoded values

            Decode byte-sequences to int lists e.g. b'\xe8\x03\x00\xd0\x07\x00' -> [1000, 2000]
            \xe8\x03\x00 is 3-byte encoded 1000, and \xd0\x07\x00 is 3-byte encoded 2000
            '''
            if byte_ids is None:
                return []

            length = len(byte_ids)
            return [str(int.from_bytes(byte_ids[i:i + bit_len], byteorder='little', signed=True)) for i in
                    range(0, length, bit_len)]

        gene_vars = d[(d[VCF_CONSEQUENCE].isin([VCF_MISSENSE_VARIANT] + ALL_PTV) | (d['spliceai'] >= 0.2))].copy()
        gene_vars['all_samples'] = gene_vars['all_samples'].apply(from_bytes)

        if max_ac is not None:
            gene_vars = gene_vars[gene_vars['ac'] <= max_ac].copy()

        if max_af is not None:
            gene_vars = gene_vars[gene_vars['af'] <= max_af].copy()

        if samples_to_subset is not None:
            gene_vars['all_samples'] = gene_vars['all_samples'].apply(lambda s: set(s) & samples_to_subset)
            gene_vars = gene_vars[gene_vars['all_samples'].apply(len) > 0].copy()

        for g in batch_genes:
            gene_stats[GENE_NAME].append(g)
            gv = gene_vars[gene_vars['symbol'] == g]
            if len(gv) > 0:
                n_gene_carriers = np.sum(gv['all_samples'].apply(len))
            else:
                n_gene_carriers = 0

            gene_stats['n_variants'].append(len(gv))
            gene_stats['n_carriers'].append(n_gene_carriers)

    con.close()

    gene_stats = pd.DataFrame(gene_stats)

    gene_stats['max_AC'] = max_ac
    gene_stats['max_AF'] = max_af


    return gene_stats


def get_ukb_exome_variants_for_coordinates(chrom, start_pos, end_pos, ukb_path=None):

    echo('Getting variants for positions:', chrom, ':', start_pos, '-', end_pos)

    if ukb_path is None:
        ukb_path = UKB_DATA_PATH + '/exomes/26_OCT_2020/ukb_exomes.26_OCT_2020.chr%s.full_info.with_gnomad_coverage.pickle'

    echo('Reading chromosome:', chrom, 'from:', ukb_path % chrom)

    chr_vars = pd.read_pickle(ukb_path % chrom)

    chr_gene_vars = chr_vars[(chr_vars[VCF_POS] >= start_pos) & (chr_vars[VCF_POS] <= end_pos)].copy()

    echo('Total variants:', len(chr_gene_vars), ', from', len(set(chr_gene_vars[GENE_NAME])), 'genes')

    return chr_gene_vars


def fm_split(r):
    return re.split(r'[|,;]', r)


def gwas_genes_from_finemapped_df(finemapped_snps, verbose=False):

    finemapped_snps = finemapped_snps.copy()

    gwas_genes = {}
    genes_per_index_variant = {}

    gwas_keys = None
    for _, r in finemapped_snps.iterrows():

        for g in sorted(set(fm_split(r[GENE_NAME]))):
            if r['index_variant'] not in genes_per_index_variant:
                genes_per_index_variant[r['index_variant']] = set()

            genes_per_index_variant[r['index_variant']].add(g)

            g_dict = {GENE_NAME: g,
                      'n_variants': 1,
                      'sum_log_p': np.log10(r['pvalue'])}

            for k in r.keys():
                if k != GENE_NAME:
                    g_dict[k] = r[k]

            gwas_keys = list(g_dict)

            if g not in gwas_genes:
                gwas_genes[g] = g_dict
            else:
                prev_g_dict = gwas_genes[g]
                best_g_dict = min([prev_g_dict, g_dict], key=lambda d: d['pvalue'])
                best_g_dict['n_variants'] += 1
                best_g_dict['sum_log_p'] = prev_g_dict['sum_log_p'] + g_dict['sum_log_p']

                gwas_genes[g] = best_g_dict

    gwas_genes = pd.DataFrame(dict((k + ('/gwas' if k not in [GENE_NAME, VCF_CHROM] else ''),
                                    [gwas_genes[g][k] for g in sorted(gwas_genes)]) for k in gwas_keys)).sort_values('pvalue/gwas')

    gwas_genes['log_p_per_variant/gwas'] = gwas_genes['sum_log_p/gwas'] / gwas_genes['n_variants/gwas']

    finemapped_snps['log_p'] = np.log10(finemapped_snps['pvalue'])

    iv_total = finemapped_snps.groupby('index_variant').sum()[['log_p']].reset_index().rename(
        columns={'index_variant': 'index_variant/gwas',
                 'log_p': 'total_log_p/gwas'})

    gwas_genes = pd.merge(gwas_genes, iv_total, on='index_variant/gwas')

    iv_total = gwas_genes.groupby('index_variant/gwas').sum()[['sum_log_p/gwas']].reset_index().rename(
        columns={'sum_log_p/gwas': 'total_sum_log_p/gwas'})

    gwas_genes = pd.merge(gwas_genes, iv_total, on='index_variant/gwas')

    gwas_genes['fm_posterior/gwas'] = gwas_genes['sum_log_p/gwas'] / gwas_genes['total_log_p/gwas']
    gwas_genes['fm_posterior2/gwas'] = gwas_genes['sum_log_p/gwas'] / gwas_genes['total_sum_log_p/gwas']

    index_vars = sorted(genes_per_index_variant)
    genes_per_index_variant = pd.DataFrame({'index_variant/gwas': index_vars,
                                            'n_genes_near_index_variant': [len(genes_per_index_variant[iv]) for iv in index_vars]})

    gwas_genes = pd.merge(gwas_genes, genes_per_index_variant, on='index_variant/gwas')

    return gwas_genes


def read_finemapped_variants(fname, min_AF=0.01, GWAS_PVALUE_THRESHOLD=5e-8, verbose=True, exclude_eqtls=False):

    if verbose:
        echo('Reading finemapped variants from:', fname)
    finemapped_snps = read_finemapped_variants_to_df(fname,
                                                     GWAS_PVALUE_THRESHOLD,
                                                     min_AF,
                                                     verbose=verbose,
                                                     exclude_eqtls=exclude_eqtls)

    finemapped_genes = gwas_genes_from_finemapped_df(finemapped_snps, verbose=verbose)
    if verbose:
        echo('finemapped_snps:', len(finemapped_snps))
        echo('finemapped_genes:', len(finemapped_genes))

    return finemapped_genes, finemapped_snps


def read_finemapped_variants_to_df(fname,
                                   GWAS_PVALUE_THRESHOLD=5e-8,
                                   min_AF=0.0,
                                   verbose=True,
                                   update_index_variant_ids=True,
                                   exclude_eqtls=False,
                                   gencode_path=None):

    if verbose:
        echo('Reading:', fname)

    if fname.endswith('.csv.gz'):
        finemapped_snps = pd.read_csv(fname, sep='\t', dtype={VCF_CHROM: str})
    else:
        finemapped_snps = pd.read_pickle(fname)

    finemapped_snps = finemapped_snps[finemapped_snps['pvalue'] <= GWAS_PVALUE_THRESHOLD].copy()

    if VCF_AF in list(finemapped_snps):
        after = finemapped_snps[(finemapped_snps[VCF_AF] >= min_AF) &
                                (finemapped_snps[VCF_AF] <= 1 - min_AF)].copy()
        fmv_skipped = 0
        rows = []

        for _, r in after.iterrows():
            fmv = r['finemapped_via'].split(',')
            to_skip = False
            for _fmv in fmv:
                af = [k for k in _fmv.split('|') if k.startswith('AF=')]
                if len(af) > 0:
                    af = float(af[0][3:])
                    if af < min_AF or af > 1 - min_AF:
                        to_skip = True
                        fmv_skipped += 1
                        break

            if not to_skip:
                rows.append(r)

        after = pd.DataFrame(rows)
        if verbose:
            echo('Filtering variants below', min_AF, ', before:', len(finemapped_snps), ', after:', len(after),
                 ', fmv_skipped:', fmv_skipped)
        finemapped_snps = after

    if update_index_variant_ids:
        index_variants_to_update = sorted(set(finemapped_snps['index_variant']) - set(finemapped_snps[VARID_REF_ALT]))
        if verbose:
            echo('index_variants_to_update:', len(index_variants_to_update))

        for iv in index_variants_to_update:
            vars_to_update = finemapped_snps[finemapped_snps['index_variant'] == iv].sort_values('pvalue')
            new_index_variant = vars_to_update.iloc[0][VARID_REF_ALT]
            if verbose:
                echo('updating old index_variant:', iv, 'to:', new_index_variant)
            finemapped_snps['index_variant'] = np.where(finemapped_snps['index_variant'] == iv,
                                                        new_index_variant,
                                                        finemapped_snps['index_variant'])

    if exclude_eqtls:

        if verbose:
            echo('Excluding eQTLs')

        if gencode_path is None:
            gencode_path = ROOT_PFIZIEV_PATH + '/rare_variants/data/gencode/gencode.v24lift37.canonical.with_CDS.tsv'

        if verbose:
            echo('Reading gencode:', gencode_path)

        gencode = pd.read_csv(gencode_path, sep='\t').rename(columns={'gene': GENE_NAME, 'chrom': VCF_CHROM})
        gencode[VCF_CHROM] = gencode[VCF_CHROM].str.replace('chr', '')

        to_correct = finemapped_snps[finemapped_snps['variant_type'].str.contains(EQTL)].copy()
        finemapped_snps = finemapped_snps[~finemapped_snps['variant_type'].str.contains(EQTL)].copy()

        _to_correct = pd.merge(to_correct[[VCF_CHROM, VCF_POS, VARID_REF_ALT]].drop_duplicates(),
                               gencode[[VCF_CHROM, 'tss_pos', GENE_NAME]],
                               on=VCF_CHROM)

        _to_correct['nearest_gene_distance'] = np.abs(_to_correct[VCF_POS] - _to_correct['tss_pos'])
        _to_correct = _to_correct.sort_values('nearest_gene_distance').drop_duplicates(VARID_REF_ALT)

        _to_correct = _to_correct.rename(columns={GENE_NAME: GENE_NAME + '/nearest_gene'})

        del _to_correct['tss_pos']
        del _to_correct[VCF_CHROM]
        del _to_correct[VCF_POS]

        to_correct = pd.merge(to_correct, _to_correct, on=VARID_REF_ALT)

        def excl_eqtls(row):
            if row['variant_type'] in [EQTL, 'eQTLs']:
                # find nearest gene
                row['assoc_type'] = 'non_coding'
                row['variant_type'] = 'nearest_gene|distance=' + str(row['nearest_gene_distance'])
                row[GENE_NAME] = row[GENE_NAME + '/nearest_gene']

            elif row['variant_type'] == 'non_coding/eQTLs':
                row[GENE_NAME] = row[GENE_NAME].split(';')[0]
                row['variant_type'] = 'non_coding'

                if len(row[GENE_NAME].split(',')) == 1:
                    row['assoc_type'] = 'non_coding'
                else:
                    row['assoc_type'] = 'non_coding/ambiguous'

            return pd.Series(row)

        to_correct = to_correct.apply(excl_eqtls, axis=1)

        del to_correct[GENE_NAME + '/nearest_gene']
        del to_correct['nearest_gene_distance']

        finemapped_snps = pd.concat([finemapped_snps, to_correct], ignore_index=True)

    finemapped_snps['n_genes'] = finemapped_snps[GENE_NAME].apply(lambda x: len(set(fm_split(x))))

    return finemapped_snps


def evaluate_rv_vs_gwas(rv_phenotypes,
                        gwas_phenotypes=None,
                        RV_DIR=ROOT_PATH + '/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results/',
                        FINEMAPPED_DIR=ROOT_PATH + '/pfiziev/rare_variants/data/finemapping/finemapped_gwas/ukbiobank.v4/',
                        GWAS_THRESHOLD=5e-8,
                        MAX_AMBIGUOUS_GENES=1000,
                        var_type='del',
                        rv_fname_suffix='.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.revel_scores.main_analysis.pickle',
                        finemapped_fname_template='%s.finemapped.csv.gz',
                        assoc_types_to_test=None,
                        TOP_N_GENES_TO_DISPLAY=5,
                        use_GREAT=False,
                        FIG_SIZE=(20, 20),
                        use_scipy_hypergeom=False,
                        verbose=True,
                        plot=True,
                        use_top_k_subindex_variants=None,
                        use_index_variants_only=False,
                        gencode=None,
                        max_N_genes_in_locus=None,
                        locus_width=None,
                        min_AF=0.01,
                        strip_colon_from_gene_names=False,
                        ph_name_mapping=None,
                        gwas_genes_filter=None,
                        rv_pvalue_threshold=None,
                        genes_to_keep=None,
                        gwas_finemapped_tables=None,
                        rv_res_fnames=None,
                        chromosomes=AUTOSOMES,
                        genes_to_collapse=None,
                        rv_metric_template='ALL/%s/carrier/pvalue/fdr_corr'
                        ):

    echo('[evaluate_rv_vs_gwas]')

    if chromosomes is AUTOSOMES:
        chromosomes = [str(i) for i in range(1, 23)]

    echo('GWAS_THRESHOLD:', GWAS_THRESHOLD,
         ', min_AF for GWAS variants:', min_AF,
         ', var_type:', var_type,
         ', MAX_AMBIGUOUS_GENES:', MAX_AMBIGUOUS_GENES,
         ', gwas_genes_filter:', gwas_genes_filter,
         ', genes_to_keep:', len(genes_to_keep) if genes_to_keep else None,
         ', chromosomes:', chromosomes)

    if ph_name_mapping is None:
        ph_name_mapping = {}

    def real_ph_name(ph_name):
        if ph_name in ph_name_mapping:
            return ph_name_mapping[ph_name] + f' / {ph_name}'
        else:
            return ph_name

    to_return = []

    if gwas_phenotypes is None:
        gwas_phenotypes = rv_phenotypes


    CODING_ASSOC_TYPES = ['coding', 'coding/ambiguous']
    NON_CODING_ASSOC_TYPES = ['non_coding', 'non_coding/ambiguous', 'nearest_gene', 'splicing', 'splicing/ambiguous']

    gwas_phenotypes_same_as_rv = (len(gwas_phenotypes) == len(rv_phenotypes) and all(p1 == p2 for p1, p2 in zip(sorted(gwas_phenotypes),
                                                                                                                sorted(rv_phenotypes))))
    if assoc_types_to_test is None:
        assoc_types_to_test = [None]

    rv_cache = {}

    def add_blanks(rv_phenotypes, rv_vs_gwas_pvalues, rv_vs_gwas_n_top_genes):
        for rv_ph_name in sorted(rv_phenotypes):
            rv_vs_gwas_n_top_genes[rv_ph_name].append([0, 0, 0, 0, 0, 0])
            rv_vs_gwas_pvalues[rv_ph_name].append(1)

    # perform the test for high pLI, low pLI and all genes
    rv_metric = None
    for assoc_types in assoc_types_to_test:

        rv_vs_gwas_pvalues = dict((p, []) for p in sorted(rv_phenotypes))

        echo('assoc_types:', assoc_types)
        echo('Rare var. phenotypes:', len(rv_phenotypes), rv_phenotypes)
        echo('GWAS phenotypes:', len(gwas_phenotypes), gwas_phenotypes)

        rv_vs_gwas_pvalues['phenotype'] = []
        rv_vs_gwas_n_top_genes = dict((k, []) for k in rv_vs_gwas_pvalues)

        # iterate over the GWAS results for all phenotypes
        for gwas_ph_name in sorted(gwas_phenotypes):
            if gwas_finemapped_tables is not None:
                finemapped_snps = gwas_finemapped_tables[gwas_ph_name]
            else:
                finemapped_fname = FINEMAPPED_DIR + '/' + finemapped_fname_template % gwas_ph_name

                if not os.path.exists(finemapped_fname):
                    echo('File not found:', finemapped_fname)
                    continue

                finemapped_snps = pd.read_csv(finemapped_fname, sep='\t', dtype={VCF_CHROM: str})

            rv_vs_gwas_pvalues['phenotype'].append(real_ph_name(gwas_ph_name))
            rv_vs_gwas_n_top_genes['phenotype'].append(real_ph_name(gwas_ph_name))

            if chromosomes is not None:
                finemapped_snps = finemapped_snps[finemapped_snps[VCF_CHROM].isin(chromosomes)].copy()

            if strip_colon_from_gene_names:
                finemapped_snps[GENE_NAME] = finemapped_snps[GENE_NAME].apply(lambda x: ','.join([gn.split(':')[0] for gn in fm_split(x)]))

            if genes_to_collapse is not None:
                for g_group in genes_to_collapse:
                    finemapped_snps[GENE_NAME] = finemapped_snps[GENE_NAME].apply(
                        lambda x: ','.join(sorted(set([g_group if gn in genes_to_collapse[g_group] else gn for gn in fm_split(x)]))))

            finemapped_snps = finemapped_snps[finemapped_snps['pvalue'] <= GWAS_THRESHOLD].copy()

            if VCF_AF in list(finemapped_snps):
                after = finemapped_snps[(finemapped_snps[VCF_AF] >= min_AF) &
                                        (finemapped_snps[VCF_AF] <= 1 - min_AF)].copy()
                fmv_skipped = 0
                rows = []

                for _, r in after.iterrows():
                    fmv = r['finemapped_via'].split(',')
                    to_skip = False
                    for _fmv in fmv:
                        af = [k for k in _fmv.split('|') if k.startswith('AF=')]
                        if len(af) > 0:
                            af = float(af[0][3:])
                            if af < min_AF or af > 1 - min_AF:
                                to_skip = True
                                fmv_skipped += 1
                                break

                    if not to_skip:
                        rows.append(r)

                after = pd.DataFrame(rows)
                if verbose:
                    echo(gwas_ph_name, ', Filtering variants below', min_AF, ', before:', len(finemapped_snps), ', after:', len(after), ', fmv_skipped:', fmv_skipped)
                finemapped_snps = after

            if len(finemapped_snps) == 0:
                add_blanks(rv_phenotypes, rv_vs_gwas_pvalues, rv_vs_gwas_n_top_genes)
                continue

            finemapped_snps['n_genes'] = finemapped_snps[GENE_NAME].apply(lambda x: len(set(fm_split(x))))

            if use_index_variants_only:
                finemapped_snps = finemapped_snps[finemapped_snps['index_variant'] == finemapped_snps['varid_ref_alt']].copy()

            if max_N_genes_in_locus is not None:

                d = pd.merge(finemapped_snps,
                             gencode[[VCF_CHROM, 'tss_pos', GENE_NAME]].drop_duplicates(subset=[GENE_NAME]),
                             on=VCF_CHROM)

                d['dist'] = np.abs(d[VCF_POS] - d['tss_pos'])
                d = d[d['dist'] <= locus_width]
                d = d.groupby(VARID_REF_ALT).size().reset_index().rename(columns={'index': VARID_REF_ALT, 0: 'n_genes_in_locus'})
                d = d[d['n_genes_in_locus'] <= max_N_genes_in_locus]
                index_variants_to_keep = set(d[VARID_REF_ALT])

                if verbose:
                    echo('Keeping', len(index_variants_to_keep),
                         'variants that have at most', max_N_genes_in_locus,'genes within', locus_width, 'of the index variant')

                finemapped_snps = finemapped_snps[finemapped_snps[VARID_REF_ALT].isin(index_variants_to_keep)].copy()

                if len(finemapped_snps) == 0:
                    add_blanks(rv_phenotypes, rv_vs_gwas_pvalues, rv_vs_gwas_n_top_genes)
                    continue

            if assoc_types is not None:
                if assoc_types == 'non_ambiguous':
                    gwas_snps = finemapped_snps[~finemapped_snps['assoc_type'].str.contains('ambiguous')].copy()

                else:
                    to_exclude = set()
                    if assoc_types.startswith('non_coding'):
                        assoc_type_labels = NON_CODING_ASSOC_TYPES
                        coding_genes_rows = finemapped_snps[
                            finemapped_snps['assoc_type'].apply(lambda x: any(at in CODING_ASSOC_TYPES for at in x.split(';')))]

                        to_exclude = set(g for r in coding_genes_rows[GENE_NAME] for g in fm_split(r))
                        if verbose:
                            echo(real_ph_name(gwas_ph_name),
                                 'Subtracting coding hits:',
                                 len(to_exclude),
                                 len(to_exclude) / len(set(g for r in finemapped_snps[GENE_NAME] for g in fm_split(r))),
                                 sorted(to_exclude))

                    elif assoc_types.startswith('coding'):
                        assoc_type_labels = CODING_ASSOC_TYPES
                    else:

                        raise Exception('Unknown assoc_type:' + str(assoc_types))

                    if not assoc_types.endswith('/all'):
                        assoc_type_labels = [at for at in assoc_type_labels if not at.endswith('/ambiguous')]

                    gwas_snps = finemapped_snps[
                        finemapped_snps['assoc_type'].apply(lambda x: len(set(x.split(';')) & set(assoc_type_labels)) > 0)].copy()

                    gwas_snps = gwas_snps[
                        ~gwas_snps[GENE_NAME].apply(lambda r: len(set(fm_split(r)) & to_exclude) > 0)].copy()

            else:
                gwas_snps = finemapped_snps

            if len(gwas_snps) > 0 and verbose:
                echo(real_ph_name(gwas_ph_name), ', gwas_snps:',
                     len(gwas_snps),
                     len(set([g for _, r in gwas_snps.iterrows() for g in fm_split(r[GENE_NAME])])),
                     ', <=', MAX_AMBIGUOUS_GENES, 'genes:', np.sum(gwas_snps['n_genes'] <= MAX_AMBIGUOUS_GENES),
                     len(set([g for _, r in gwas_snps[gwas_snps['n_genes'] <= MAX_AMBIGUOUS_GENES].iterrows() for g in
                              fm_split(r[GENE_NAME])])))

            if len(gwas_snps) == 0:
                add_blanks(rv_phenotypes, rv_vs_gwas_pvalues, rv_vs_gwas_n_top_genes)
                continue

            gwas_snps = gwas_snps[gwas_snps['n_genes'] <= MAX_AMBIGUOUS_GENES].copy()

            gene_to_index_variant = dict((g, r['index_variant'])
                                            for _, r in gwas_snps.iterrows()
                                                for g in fm_split(r[GENE_NAME]))

            n_index_variants = len(set(gene_to_index_variant.values()))

            if verbose:
                echo(real_ph_name(gwas_ph_name), 'GWAS variants:', len(gwas_snps))

            if use_top_k_subindex_variants is not None:
                gwas_snps = gwas_snps.sort_values(['index_variant', 'pvalue']).copy()
                gwas_snps['subindex_rank'] = range(len(gwas_snps))
                d = gwas_snps[gwas_snps['index_variant'] == gwas_snps[VARID_REF_ALT]][['index_variant', 'subindex_rank']].rename(columns={'subindex_rank':
                                                                                                                                          'index_rank'})
                gwas_snps = pd.merge(gwas_snps, d, on='index_variant')
                gwas_snps['subindex_rank'] = gwas_snps['subindex_rank'] - gwas_snps['index_rank'] + 1
                gwas_snps = gwas_snps[gwas_snps['subindex_rank'] <= use_top_k_subindex_variants].copy()

                if verbose:
                    echo('Keeping subindex-rank up to', use_top_k_subindex_variants, ',', len(gwas_snps))

            if len(gwas_snps) == 0:
                add_blanks(rv_phenotypes, rv_vs_gwas_pvalues, rv_vs_gwas_n_top_genes)
                continue

            gwas_genes_df = gwas_genes_from_finemapped_df(gwas_snps)

            if gwas_genes_filter is not None:
                for param_l in gwas_genes_filter:
                    param_v, param_leq = gwas_genes_filter[param_l]
                    if param_leq:
                        gwas_genes_df = gwas_genes_df[gwas_genes_df[param_l] <= param_v].copy()
                    else:
                        gwas_genes_df = gwas_genes_df[gwas_genes_df[param_l] >= param_v].copy()

            if len(gwas_genes_df) == 0:
                add_blanks(rv_phenotypes, rv_vs_gwas_pvalues, rv_vs_gwas_n_top_genes)
                continue

            if genes_to_keep is not None:
                gwas_genes_df = gwas_genes_df[gwas_genes_df[GENE_NAME].isin(genes_to_keep)].copy()

            gwas_genes = set(gwas_genes_df[GENE_NAME])

            if verbose:
                echo('Remaining GWAS genes:', len(gwas_genes), len(gwas_genes) / len(set(g for r in finemapped_snps[GENE_NAME] for g in fm_split(r))))

            if use_GREAT:
                great_fname = ROOT_PATH + f'/pfiziev/rare_variants/data/finemapping/finemapped_gwas/ukbiobank.v3.great/{gwas_ph_name}.finemapped.tsv'

                if os.path.exists(great_fname):

                    echo('Reading GREAT:', great_fname)

                    GREAT_THRESHOLD = 1e-3
                    GREAT_METRIC = 'HyperFdrQ'

                    great = pd.read_csv(great_fname,
                                        sep='\t',
                                        skiprows=3,
                                        skipfooter=4,
                                        engine='python')

                    great_genes = set(
                        [g for gg in list(great[great[GREAT_METRIC] <= GREAT_THRESHOLD]['Genes']) for g in gg.split(',')])

                    all_gwas_genes = gwas_genes

                    gwas_genes = gwas_genes & great_genes | set([g for _, r in gwas_snps.iterrows()
                                                                 for g in fm_split(r[GENE_NAME])
                                                                 if (r[VARID_REF_ALT] == r['index_variant'] and
                                                                     len(set(fm_split(r[GENE_NAME]))) == 1)])

                    echo('GWAS genes before and after GREAT filtering:', len(all_gwas_genes), len(gwas_genes))

                else:
                    echo('GREAT results not available for', gwas_ph_name, ', skipping..')

            max_gwas_effects = []
            gwas_index_snps = []

            for g in sorted(gwas_genes):
                gene_gwas_snps = gwas_snps[gwas_snps[GENE_NAME].apply(lambda x: g in fm_split(x))]

                gene_max_gwas_effect, gene_gwas_index_snp = max(zip(gene_gwas_snps['beta'],
                                                                    gene_gwas_snps['index_variant']),
                                                                key=lambda z: abs(z[0]))

                max_gwas_effects.append(gene_max_gwas_effect)
                gwas_index_snps.append(gene_gwas_index_snp)

            gwas_stat = pd.DataFrame({GENE_NAME: sorted(gwas_genes),
                                      'GWAS effect': max_gwas_effects,
                                      'GWAS index SNP': gwas_index_snps})

            gene_gwas_hit_type = pd.DataFrame({GENE_NAME: sorted(gwas_genes),
                                               'assoc_type': [','.join(set(gwas_snps[gwas_snps[GENE_NAME].apply(
                                                   lambda x: g in fm_split(x))]['assoc_type'])) for g in
                                                              sorted(gwas_genes)]})

            if verbose:
                echo(real_ph_name(gwas_ph_name), ', GWAS genes:', len(gwas_genes), ', GWAS SNPs:', len(gwas_snps))

            if len(gwas_genes) == 0:
                add_blanks(rv_phenotypes, rv_vs_gwas_pvalues, rv_vs_gwas_n_top_genes)
                continue

            # iterate over all rare variant analysis results for all phenotypes
            for rv_ph_name in sorted(rv_phenotypes):

                if rv_ph_name not in rv_cache:
                    if rv_res_fnames is not None:
                        rv_fname = rv_res_fnames.get(rv_ph_name, RV_DIR + f'/{rv_ph_name}/{rv_ph_name}' + rv_fname_suffix)
                    else:
                        rv_fname = RV_DIR + f'/{rv_ph_name}/{rv_ph_name}' + rv_fname_suffix

                    rv_test = read_rv_results(rv_fname, recompute_fdr=False)

                    if chromosomes is not None:
                        rv_test = rv_test[rv_test[VCF_CHROM].isin(chromosomes)].copy()

                    if var_type == BOTH:

                        rv_test = rv_test.rename(columns={f'ALL/del/carrier/beta': f'ALL/del/beta',
                                                          f'ALL/ptv/carrier/beta': f'ALL/ptv/beta'})

                        rv_test = rv_test.fillna({f'ALL/del/beta': 0,
                                                  f'ALL/ptv/beta': 0,
                                                  f'ALL/del/carrier/pvalue/fdr_corr': 1,
                                                  f'ALL/ptv/carrier/pvalue/fdr_corr': 1
                                                  })
                        rv_metric = 'rv_p'
                        rv_test[rv_metric] = rv_test.apply(lambda x: min(x[f'ALL/ptv/carrier/pvalue/fdr_corr'],
                                                                      x[f'ALL/del/carrier/pvalue/fdr_corr']
                                                                      ), axis=1)


                    else:
                        rv_metric = rv_metric_template % var_type
                        rv_test = rv_test.rename(columns={f'ALL/{var_type}/carrier/beta': f'ALL/{var_type}/beta',
                                                          f'ALL/ptv/carrier/beta': f'ALL/ptv/beta'})

                        rv_test = rv_test.dropna(subset=[rv_metric])

                    if genes_to_keep is not None:
                        rv_test = rv_test[rv_test[GENE_NAME].isin(genes_to_keep)].copy()

                    if genes_to_collapse:
                        for g_group in genes_to_collapse:
                            rv_test[GENE_NAME] = rv_test[GENE_NAME].apply(lambda gn: g_group if gn in genes_to_collapse[g_group] else gn)
                        rv_test = rv_test.sort_values(rv_metric, ascending=True).drop_duplicates(GENE_NAME)

                    # generate a column to rank by the genes: the absolute value of the rank-sums statistic
                    rv_cols_to_keep = [GENE_NAME, 'ALL/ptv/beta', f'ALL/{var_type}/beta', rv_metric]
                    rv_test = rv_test[rv_cols_to_keep].copy()

                    rv_test['sort'] = rv_test[rv_metric]

                    rv_test = rv_test.sort_values('sort', ascending=True)
                    all_ranks = list(range(1, len(rv_test) + 1))

                    # generate a column for the ranks
                    rv_test['sort_rank'] = all_ranks

                    rv_cache[rv_ph_name] = rv_test

                else:

                    rv_test = rv_cache[rv_ph_name].copy()

                # get the ranks of GWAS genes in the rare variants analysis
                gwas_genes_rv = rv_test[(rv_test[GENE_NAME].isin(gwas_genes))].sort_values('sort_rank')

                gwas_genes_rv_ranks = sorted(gwas_genes_rv['sort_rank'])

                n_gwas_genes = len(gwas_genes_rv_ranks)

                # compute the p-value of the ranks of the GWAS genes
                if n_gwas_genes == 0:
                    best_n_genes, best_pval = 0, 1
                else:
                    if rv_pvalue_threshold is None:
                        best_n_genes, best_pval = test_ranks(gwas_genes_rv_ranks,
                                                             len(rv_test),
                                                             return_empirical=False,
                                                             multiple_testing_correction=True,
                                                             verbose=False,
                                                             use_scipy_hypergeom=use_scipy_hypergeom)
                    else:
                        best_n_genes = np.sum(gwas_genes_rv[rv_metric] <= rv_pvalue_threshold)
                        all_sign_rv_genes = np.sum(rv_test[rv_metric] <= rv_pvalue_threshold)
                        best_pval = scipy.stats.binom_test(best_n_genes, n_gwas_genes, p=all_sign_rv_genes / len(rv_test), alternative='greater')


                rv_vs_gwas_pvalues[rv_ph_name].append(best_pval)

                best_n_index_variants = len(set(gene_to_index_variant[g] for g in gwas_genes_rv[GENE_NAME][:best_n_genes]))

                rv_vs_gwas_n_top_genes[rv_ph_name].append([best_n_genes,
                                                           n_gwas_genes,
                                                           gwas_genes_rv_ranks,
                                                           best_n_index_variants,
                                                           n_index_variants,
                                                           len(rv_test)])

                if gwas_ph_name == rv_ph_name and n_gwas_genes > 0:

                    genes_gwas_info = sorted([(r['sort_rank'],
                                               r[GENE_NAME],
                                               gene_gwas_hit_type[gene_gwas_hit_type[GENE_NAME] == r[GENE_NAME]].iloc[0]['assoc_type'],
                                               r[f'ALL/{var_type}/beta' if var_type != BOTH else f'ALL/del/beta'],
                                               r[f'ALL/ptv/beta'],
                                               r[rv_metric]
                                               )
                                              for _, r in gwas_genes_rv.iterrows()])

                    if verbose:
                        echo(real_ph_name(gwas_ph_name),
                             ', pvalue=', best_pval,
                             ', best_n_genes=', best_n_genes, ', best_rank=', genes_gwas_info[best_n_genes - 1][0],
                             ', best_rv_p_value=', genes_gwas_info[best_n_genes - 1][-1],
                             ', gwas_genes=', len(gwas_genes_rv_ranks),
                             ', total_genes=', len(rv_test),
                             ', top_genes:', genes_gwas_info[:min(best_n_genes, TOP_N_GENES_TO_DISPLAY)],
                             )

                    r = pd.DataFrame(genes_gwas_info).rename(
                        columns={0: 'Rare Var. Test Rank', 1: 'Gene', 2: 'GWAS hit type', 3: 'Rare Var. effect (all)',
                                 4: 'Rare Var. effect (LoF only)',
                                 5: rv_metric})
                    if len(r) > 0:
                        r = pd.merge(r, gwas_stat, left_on='Gene', right_on=GENE_NAME).sort_values(
                            'Rare Var. Test Rank')
                        r['GWAS effect'] = np.where(((r['Rare Var. effect (all)'] > 0) &
                                                     (r['GWAS effect'] > 0)) |
                                                    ((r['Rare Var. effect (all)'] < 0) &
                                                     (r['GWAS effect'] < 0))
                                                    , r['GWAS effect'], -r['GWAS effect'])
                        r = r.fillna(0)

                        r['% effect increase (all)'] = (
                                    100 * (np.abs(r['Rare Var. effect (all)']) - np.abs(r['GWAS effect'])) / np.abs(
                                r['GWAS effect']))  # .astype(int)
                        r['% effect increase (LoF only)'] = (100 * (
                                    np.abs(r['Rare Var. effect (LoF only)']) - np.abs(r['GWAS effect'])) / np.abs(
                            r['GWAS effect']))  # .astype(int)

                        for k in ['GWAS effect', 'Rare Var. effect (all)', 'Rare Var. effect (LoF only)']:
                            r[k] = np.round(r[k], 2)
                        r = r[['Rare Var. Test Rank', 'Gene', 'GWAS hit type', 'GWAS effect', 'Rare Var. effect (all)',
                               'Rare Var. effect (LoF only)', '% effect increase (all)', '% effect increase (LoF only)',
                               'GWAS index SNP']]

                        if verbose:
                            display(r.head(min(best_n_genes, TOP_N_GENES_TO_DISPLAY)))

            echo(real_ph_name(gwas_ph_name), ':',
                 rv_vs_gwas_pvalues[gwas_ph_name][-1] if gwas_ph_name in rv_vs_gwas_pvalues else None,
                 min([(rv_vs_gwas_pvalues[k][-1],
                       rv_vs_gwas_n_top_genes[k][-1][:2],
                       k)
                      for k in rv_phenotypes]))  # , [(k, rv_vs_gwas_pvalues[k][-1]) for k in all_phenotypes])

        if verbose:
            echo('assoc_types:', assoc_types)

        rv_vs_gwas_pvalues_df = pd.DataFrame(rv_vs_gwas_pvalues)

        min_pvalue = min([p for p in rv_vs_gwas_pvalues_df[rv_phenotypes].to_numpy().flatten() if p > 0])

        if verbose:
            echo('min non-zero p-value:', min_pvalue)
        rv_vs_gwas_pvalues_df = rv_vs_gwas_pvalues_df.replace(0, min_pvalue / 10)
        rv_vs_gwas_pvalues_df = rv_vs_gwas_pvalues_df.sort_values('phenotype').rename(columns={'phenotype': 'GWAS'})

        rv_vs_gwas_pvalues_df = rv_vs_gwas_pvalues_df.set_index('GWAS')

        rv_vs_gwas_n_top_genes_df = pd.DataFrame(rv_vs_gwas_n_top_genes)
        rv_vs_gwas_n_top_genes_df = rv_vs_gwas_n_top_genes_df.sort_values('phenotype').rename(columns={'phenotype': 'GWAS'})

        col_order = [c for c in list(rv_vs_gwas_n_top_genes_df) if c != 'GWAS']
        rv_vs_gwas_n_top_genes_df = rv_vs_gwas_n_top_genes_df.set_index('GWAS')[col_order]

        diag_sum = np.sum(np.diag(np.log10(rv_vs_gwas_pvalues_df)))
        off_diag_sum = np.log10(rv_vs_gwas_pvalues_df).to_numpy().sum() - diag_sum

        diag_sum /= len(rv_vs_gwas_pvalues_df)
        off_diag_sum /= len(rv_vs_gwas_pvalues_df) ** 2 - len(rv_vs_gwas_pvalues_df)
        diag_off_diag_ratio = diag_sum / off_diag_sum


        log10p = np.log10(rv_vs_gwas_pvalues_df).to_numpy()

        diag_off_diag_stat, diag_off_diag_pvalue = scipy.stats.ranksums(np.diag(log10p),
                                                                        [log10p[i][j]
                                                                            for i in range(log10p.shape[0])
                                                                                for j in range(log10p.shape[1]) if i != j])

        echo(assoc_types, 'diag_off_diag_ratio:', diag_off_diag_ratio,
             ', diag_sum:', diag_sum,
             ', off_diag_sum:', off_diag_sum,
             ', Ranksums diag/off-diag:', diag_off_diag_pvalue)

        if plot:
            echo('Non-clustered heatmap')
            d = rv_vs_gwas_pvalues_df
            sns.clustermap(-np.log10(d), cmap='Blues', figsize=FIG_SIZE, row_cluster=False, col_cluster=False)
            plt.show()

            echo('Clustered by rows')
            d = -np.log10(rv_vs_gwas_pvalues_df)

            if gwas_phenotypes_same_as_rv:
                sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                               row_linkage=scipy.cluster.hierarchy.linkage(d, optimal_ordering=True),
                               col_linkage=scipy.cluster.hierarchy.linkage(d, optimal_ordering=True))

            else:
                sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                               row_linkage=scipy.cluster.hierarchy.linkage(d, optimal_ordering=True),
                               col_cluster=False)

            plt.show()

            echo('Clustered by columns')

            if gwas_phenotypes_same_as_rv:
                sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                               row_linkage=scipy.cluster.hierarchy.linkage(d.T, optimal_ordering=True),
                               col_linkage=scipy.cluster.hierarchy.linkage(d.T, optimal_ordering=True))
            else:
                sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                               row_cluster=False,
                               col_linkage=scipy.cluster.hierarchy.linkage(d.T, optimal_ordering=True))
            plt.show()


        to_return.append([rv_vs_gwas_pvalues_df, rv_vs_gwas_n_top_genes_df, diag_off_diag_pvalue, diag_sum])

    return to_return


def plot_eval_gwas_vs_rv(rv_vs_gwas_pvalues_df,
                         gwas_phenotypes_same_as_rv=True,
                         FIG_SIZE=(20, 20),
                         plot=True,
                         pvalue_threshold=1,
                         rows_to_col_mapping=None,
                         ph_to_name_mapping=None,
                         cols_to_exclude=None,
                         rows_to_exclude=None,
                         min_pvalue=1e-300,
                         fontsize=12,
                         out_prefix=None):

    echo('[plot_eval_gwas_vs_rv]!!')

    if ph_to_name_mapping is None:
        ph_to_name_mapping = {}

    for c in list(rv_vs_gwas_pvalues_df):
        rv_vs_gwas_pvalues_df[c] = np.where(rv_vs_gwas_pvalues_df[c] < min_pvalue, min_pvalue, rv_vs_gwas_pvalues_df[c])

    if rows_to_col_mapping is not None:
        echo('Reorder rows according to rows_to_col_mapping')

        rv_vs_gwas_pvalues_df = rv_vs_gwas_pvalues_df[sorted(list(rv_vs_gwas_pvalues_df))]
        df_index = list(rv_vs_gwas_pvalues_df.index)
        rv_vs_gwas_pvalues_df = rv_vs_gwas_pvalues_df.loc[sorted(df_index, key=lambda r: rows_to_col_mapping[r])]

    to_show = set()
    for (c, r) in zip(list(rv_vs_gwas_pvalues_df), list(rv_vs_gwas_pvalues_df.index)):
        if rv_vs_gwas_pvalues_df.loc[r][c] <= pvalue_threshold:
            to_show.add((r, c))

    if cols_to_exclude is not None:
        to_show = set([(r, c) for (r, c) in to_show if c not in cols_to_exclude])

    if rows_to_exclude is not None:
        to_show = set([(r, c) for (r, c) in to_show if r not in rows_to_exclude])

    cols_to_keep = [c for r, c in to_show]
    rows_to_keep = [r for r, c in to_show]

    echo('columns that pass p-value threshold of', pvalue_threshold, ':', len(cols_to_keep), ',', cols_to_keep)
    echo('rows that pass p-value threshold of', pvalue_threshold, ':', len(rows_to_keep), ',', rows_to_keep)

    rv_vs_gwas_pvalues_df = rv_vs_gwas_pvalues_df.loc[rows_to_keep][cols_to_keep].copy()

    diag_sum = np.sum(np.diag(np.log10(rv_vs_gwas_pvalues_df)))

    off_diag_sum = np.log10(rv_vs_gwas_pvalues_df).to_numpy().sum() - diag_sum

    diag_sum /= len(rv_vs_gwas_pvalues_df)
    off_diag_sum /= len(rv_vs_gwas_pvalues_df) ** 2 - len(rv_vs_gwas_pvalues_df)
    diag_off_diag_ratio = diag_sum / off_diag_sum

    echo('diag_off_diag_ratio:', diag_off_diag_ratio, ', diag_sum:', diag_sum, ', off_diag_sum:', off_diag_sum)

    log10p = np.log10(rv_vs_gwas_pvalues_df).to_numpy()
    echo('Ranksums diag/off-diag:',
         scipy.stats.ranksums(np.diag(log10p),
                              [log10p[i][j] for i in range(log10p.shape[0]) for j in range(log10p.shape[1]) if
                               i != j]))
    rv_vs_gwas_pvalues_df = rv_vs_gwas_pvalues_df.rename(columns=ph_to_name_mapping, index=ph_to_name_mapping)
    duplicate_columns = list(np.array(list(rv_vs_gwas_pvalues_df))[rv_vs_gwas_pvalues_df.columns.duplicated()])
    duplicate_rows = list(np.array(list(rv_vs_gwas_pvalues_df.index))[rv_vs_gwas_pvalues_df.index.duplicated()])

    echo('duplicate_columns:', duplicate_columns)
    echo('duplicate_rows:', duplicate_rows)

    if len(duplicate_columns) > 0 or len(duplicate_rows) > 0:
        echo('[ERROR] duplicated columns:', duplicate_columns)
        echo('[ERROR] duplicated rows:', duplicate_rows)

    if out_prefix is not None:
        echo('Saving data in:', out_prefix + '.data.csv.gz and pickle')
        rv_vs_gwas_pvalues_df.to_csv(out_prefix + '.data.csv.gz', sep='\t', index=False)
        rv_vs_gwas_pvalues_df.to_pickle(out_prefix + '.data.pickle')

    if plot:
        echo('Non-clustered heatmap')
        d = rv_vs_gwas_pvalues_df

        sns.clustermap(-np.log10(d), cmap='Blues', figsize=FIG_SIZE, row_cluster=False, col_cluster=False)
        if out_prefix is not None:
            plt.savefig(out_prefix + '.non_clustered.svg')
            plt.savefig(out_prefix + '.non_clustered.png', dpi=300)

        plt.show()

        echo('Sorted heatmap')

        plt.figure(figsize=FIG_SIZE)

        col_names = list(d)
        to_plot = -np.log10(d)

        col_linkage = scipy.cluster.hierarchy.linkage(to_plot.T,
                                                      method='single',
                                                      metric='euclidean',
                                                      optimal_ordering=True)

        col_order = [col_names[i] for i in scipy.cluster.hierarchy.leaves_list(col_linkage)]

        row_order = []

        max_col = dict((r, max([(to_plot.loc[r][c], c) for c in col_names])) for r in list(to_plot.index))

        for c in col_order:
            for max_row in list(to_plot[c].reset_index().sort_values(c, ascending=False)['GWAS']):
                if max_row in row_order or max_col[max_row][1] != c:
                    continue

                row_order.append(max_row)

        to_plot = to_plot.loc[row_order][col_order]

        hm = sns.heatmap(to_plot, cmap='Blues')
        hm.set_xticklabels(hm.get_xticklabels(), rotation=45, horizontalalignment='right')

        if out_prefix is not None:
            plt.savefig(out_prefix + '.sorted.svg')
            plt.savefig(out_prefix + '.sorted.png', dpi=300)

        plt.show()

        echo('Clustered by rows')
        d = -np.log10(rv_vs_gwas_pvalues_df)
        if gwas_phenotypes_same_as_rv:
            res = sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                                 row_linkage=scipy.cluster.hierarchy.linkage(d, optimal_ordering=True),
                                 col_linkage=scipy.cluster.hierarchy.linkage(d, optimal_ordering=True))
            res.ax_heatmap.set_xticklabels(res.ax_heatmap.get_xmajorticklabels(), fontsize=fontsize, rotation=45, horizontalalignment='right')
            res.ax_heatmap.set_yticklabels(res.ax_heatmap.get_ymajorticklabels(), fontsize=fontsize)

        else:
            sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                           row_linkage=scipy.cluster.hierarchy.linkage(d, optimal_ordering=True),
                           col_cluster=False)

        if out_prefix is not None:
            plt.savefig(out_prefix + '.clustered_by_row.svg')
            plt.savefig(out_prefix + '.clustered_by_row.png', dpi=300)

        plt.show()

        echo('Clustered by columns')
        if gwas_phenotypes_same_as_rv:
            res = sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                                 row_linkage=scipy.cluster.hierarchy.linkage(d.T, optimal_ordering=True),
                                 col_linkage=scipy.cluster.hierarchy.linkage(d.T, optimal_ordering=True))
            res.ax_heatmap.set_xticklabels(res.ax_heatmap.get_xmajorticklabels(), fontsize=fontsize, rotation=45, horizontalalignment='right')
            res.ax_heatmap.set_yticklabels(res.ax_heatmap.get_ymajorticklabels(), fontsize=fontsize)

        else:
            sns.clustermap(d, cmap='Blues', figsize=FIG_SIZE,
                           row_cluster=False,
                           col_linkage=scipy.cluster.hierarchy.linkage(d.T, optimal_ordering=True))

        if out_prefix is not None:
            plt.savefig(out_prefix + '.clustered_by_column.svg')
            plt.savefig(out_prefix + '.clustered_by_column.png', dpi=300)

        plt.show()

        return cols_to_keep, rows_to_keep, max_col


def _get_nr_quantitative_phenotypes(max_abs_corr=0.5):
    TO_EXCLUDE = ['Mean_time_to_correctly_identify_matches',
                  'Age_at_last_live_birth',
                  'Age_at_first_live_birth',
                  'Number_of_self_reported_cancers',
                  'Sleep_duration',
                  'Age_first_had_sexual_intercourse',
                  'Lifetime_number_of_sexual_partners',
                  'Age_started_wearing_glasses_or_contact_lenses',
                  'Number_of_children_fathered',
                  'Number_of_live_births',
                  'Birth_weight_of_first_child',
                  'Birth_weight',
                  'Rheumatoid_factor',
                  'Testosterone']

    ph_corr = pd.read_pickle(UKB_DATA_PATH + '/processed_tables/quantitative_phenotypes.correlation.pickle')
    qres = pd.read_pickle(UKB_DATA_PATH + '/processed_tables/quantitative_phenotypes.rv_vs_gwas.pickle')

    nr_quant_phenotypes = []

    for ph in sorted(list(qres), key=lambda p: qres.loc[p][p]):
        if ph not in TO_EXCLUDE and (len(nr_quant_phenotypes) == 0 or all(
                [abs(ph_corr.loc[ph][p2]) <= max_abs_corr for p2 in nr_quant_phenotypes])):
            nr_quant_phenotypes.append(ph)

    return nr_quant_phenotypes


def get_nr_quantitative_phenotypes(individuals_per_phenotype,
                                   quant_phenotypes_to_exclude=None,
                                   max_abs_corr=0.5,
                                   verbose=True):

    if quant_phenotypes_to_exclude is None:
        quant_phenotypes_to_exclude = set()

    ph_corr = pd.read_pickle(UKB_DATA_PATH + '/processed_tables/quantitative_phenotypes.correlation.pickle')

    nr_quant_phenotypes = []

    for ph in individuals_per_phenotype.sort_values('n_samples', ascending=False)['phenotype']:

        if ph not in list(ph_corr):
            nr_quant_phenotypes.append(ph)

        elif (len(nr_quant_phenotypes) == 0 or

              all([abs(ph_corr.loc[ph][p2]) <= max_abs_corr for p2 in nr_quant_phenotypes
                   if p2 in list(ph_corr) and
                      not np.isnan(ph_corr.loc[ph][p2])
                   ])):

            nr_quant_phenotypes.append(ph)
        else:
            if verbose:
                echo(ph, 'is redundant:',
                     [(abs(ph_corr.loc[ph][p2]), p2) for p2 in nr_quant_phenotypes if p2 in list(ph_corr) if
                      abs(ph_corr.loc[ph][p2]) >= max_abs_corr])

    return sorted(set(nr_quant_phenotypes) - set(quant_phenotypes_to_exclude))


def open_phenotype(path, pheno, test_meds=False):
    ''' open phenotype data

    This only loads the sample_id column and the phenotype column
    '''
    echo(f'loading {pheno} from {path}')
    samples = []
    values = []
    meds_data = {}

    with gzip.open(path, 'rt') as handle:
        header = handle.readline().strip('\n').split('\t')
        sample_idx = header.index('sample_id')
        pheno_idx = header.index(pheno)
        med_indices = {}
        if test_meds:
            med_indices = {x: i for i, x in enumerate(header) if 'on_med.' in x or '.1st_visit.' in x}
            for k, v in med_indices.items():
                meds_data[k] = []

        for i, line in enumerate(handle):
            line = line.strip('\n').split('\t')
            samples.append(line[sample_idx])
            try:
                values.append(float(line[pheno_idx]))
            except ValueError:
                values.append(float('nan'))
            for k, i in med_indices.items():
                try:
                    meds_data[k].append(float(line[i]))
                except ValueError:
                    meds_data[k].append(float('nan'))

    data = {'sample_id': samples, pheno: values}
    data.update(meds_data)

    return pd.DataFrame(data)


def compute_expected_signal_in_gwas_vs_rv(ph_name, N_SHUFF=100, figsize=(20, 2)):

    def fm_split(r):
        return re.split(r'[|,;]', r)

    RV_DIR = ROOT_PATH + '/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results/'
    FINEMAPPED_DIR = ROOT_PATH + '/pfiziev/rare_variants/data/finemapping/finemapped_gwas/ukbiobank.v4/'
    GWAS_THRESHOLD = 5e-8
    MAX_AMBIGUOUS_GENES = 1000
    finemapped_fname_template = '%s.finemapped.csv.gz'
    strip_colon_from_gene_names = False
    min_AF = 0.01

    rv_fname_suffix = '.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.primateDL_score_3D_newAvg.main_analysis.pickle'

    finemapped_fname = FINEMAPPED_DIR + '/' + finemapped_fname_template % ph_name
    finemapped_snps = pd.read_csv(finemapped_fname, sep='\t', dtype={VCF_CHROM: str})
    if strip_colon_from_gene_names:
        finemapped_snps[GENE_NAME] = finemapped_snps[GENE_NAME].apply(
            lambda x: ','.join([gn.split(':')[0] for gn in fm_split(x)]))

    finemapped_snps = finemapped_snps[finemapped_snps['pvalue'] <= GWAS_THRESHOLD].copy()

    if VCF_AF in list(finemapped_snps):
        after = finemapped_snps[(finemapped_snps[VCF_AF] >= min_AF) &
                                (finemapped_snps[VCF_AF] <= 1 - min_AF)].copy()
        fmv_skipped = 0
        rows = []

        for _, r in after.iterrows():
            fmv = r['finemapped_via'].split(',')
            to_skip = False
            for _fmv in fmv:
                af = [k for k in _fmv.split('|') if k.startswith('AF=')]
                if len(af) > 0:
                    af = float(af[0][3:])
                    if af < min_AF or af > 1 - min_AF:
                        to_skip = True
                        fmv_skipped += 1
                        break

            if not to_skip:
                rows.append(r)

        after = pd.DataFrame(rows)
        echo(ph_name, ', Filtering variants below', min_AF, ', before:', len(finemapped_snps), ', after:', len(after),
             ', fmv_skipped:', fmv_skipped)
        finemapped_snps = after

    finemapped_snps['n_genes'] = finemapped_snps[GENE_NAME].apply(lambda x: len(set(fm_split(x))))

    gwas_genes = gwas_genes_from_finemapped_df(finemapped_snps)

    echo('gwas_genes:', gwas_genes.shape)

    rv_test = pd.read_pickle(RV_DIR + f'/{ph_name}/{ph_name}{rv_fname_suffix}')
    rv_test = rv_test.rename(columns={f'ALL/del/carrier/beta': f'ALL/del/beta',
                                      f'ALL/ptv/carrier/beta': f'ALL/ptv/beta'})

    for var_type in ['del', 'ptv']:
        rv_metric = f'ALL/{var_type}/carrier/pvalue/fdr_corr'

        rv_test = rv_test.fillna({rv_metric: 1})

        # generate a column to rank by the genes: the absolute value of the rank-sums statistic
        rv_test[f'{var_type}/sort'] = rv_test[rv_metric]

        # display(rv_test.head(20))
        rv_test = rv_test.sort_values(f'{var_type}/sort', ascending=True)
        all_ranks = list(range(1, len(rv_test) + 1))

        # generate a column for the ranks
        rv_test[f'{var_type}/rank'] = all_ranks

    rv_test['rv_p'] = rv_test.apply(lambda x: min(x[f'ALL/ptv/carrier/pvalue/fdr_corr'],
                                                  x[f'ALL/del/carrier/pvalue/fdr_corr']
                                                  ), axis=1)

    rv_test = rv_test.sort_values('rv_p', ascending=True)
    rv_test['rv_p/rank'] = list(range(1, len(rv_test) + 1))

    random_ranks = list(range(1, len(rv_test) + 1))
    random.shuffle(random_ranks)
    rv_test['random/rank'] = random_ranks

    rv_test = rv_test[
        [GENE_NAME] + sorted([c for c in list(rv_test) if c != GENE_NAME], key=lambda k: 1 if 'rank' in k else 10)]
    gwas_genes_rv = pd.merge(gwas_genes, rv_test, on=[GENE_NAME, VCF_CHROM]).sort_values('del/rank')
    echo('gwas_genes_rv:', gwas_genes_rv.shape)

    m_stats = []
    t_stats = None
    n_gwas_loci = None

    for s_no in range(N_SHUFF):

        random_ranks = list(range(1, len(rv_test) + 1))
        random.shuffle(random_ranks)

        rv_test['random/rank'] = random_ranks

        gwas_genes_rv = pd.merge(gwas_genes, rv_test, on=[GENE_NAME, VCF_CHROM])

        real = gwas_genes_rv.sort_values('del/rank').drop_duplicates('index_variant/gwas')['del/rank']

        shuffled = gwas_genes_rv.sort_values('random/rank').drop_duplicates('index_variant/gwas')['random/rank']

        if t_stats is None:
            n_gwas_loci = len(real)
            t_stats = [0] * n_gwas_loci

        min_recorded = False
        for i, (r, s) in enumerate(zip(real, shuffled)):

            if s <= r:
                if not min_recorded:
                    m_stats.append(i)
                    min_recorded = True

                t_stats[i] += 1

    sns.distplot(m_stats)
    plt.show()

    t_density = np.array(t_stats) / N_SHUFF

    plt.figure(figsize=figsize)
    plt.plot(range(len(t_stats)), t_density, 'o')

    p_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05]
    i_values = [-1] * len(p_thresholds)

    for i in range(len(t_density)):
        for p_i, p in enumerate(p_thresholds):
            if t_density[i] <= p:
                i_values[p_i] = i

    echo(i_values)
    plt.vlines(i_values, 0, max(t_density), color='red')

    plt.show()

    res = pd.DataFrame({'pvalue': p_thresholds,
                        'n_loci': i_values,
                        'frac': np.array(i_values) / n_gwas_loci,
                        'n_gwas_loci': n_gwas_loci,
                        'phenotype': ph_name})

    return res

