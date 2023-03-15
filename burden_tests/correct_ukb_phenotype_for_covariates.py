import argparse
import os
import pprint
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from jutils import *
from ukb_analysis import *
import re
import pickle
from constants import *


ALL_QUANTITATIVE_PHENOTYPES = [46, 47, 48, 49, 50, 78, 102, 134, 1160, 3062, 3063, 3064, 3143, 3144, 3146, 3147,
                               3148, 4079, 4080, 4100, 4101, 4103, 4104, 4105, 4106, 4119, 4120, 4122, 4123, 4124, 4125,
                               20015, 20022, 20023, 20127, 21001, 21002, 23099, 23100, 23101, 23102, 23104, 23105,
                               23106, 23107, 23108, 23109, 23110, 23111, 23112, 23113, 23114, 23115, 23116, 23117,
                               23118, 23119, 23120, 23121, 23122, 23123, 23124, 23125, 23126, 23127, 23128, 23129,
                               23130, 30000, 30010, 30020, 30030, 30040, 30050, 30060, 30070, 30080, 30090, 30100,
                               30110, 30120, 30130, 30140, 30150, 30160, 30170, 30180, 30190, 30200, 30210, 30220,
                               30230, 30240, 30250, 30260, 30270, 30280, 30290, 30300, 30500, 30510, 30520, 30530,
                               30600, 30610, 30620, 30630, 30640, 30650, 30660, 30670, 30680, 30690, 30700, 30710,
                               30720, 30730, 30740, 30750, 30760, 30770, 30780, 30790, 30800, 30810, 30820, 30830,
                               30840, 30850, 30860, 30870, 30880, 30890, 2744, 2754, 2764, 2405, 2734, 2217, 2149, 2139]

AGE_label = '21003'
SEX_label = '22001-0.0'
ALL_GENDERS = 'all'


def correct_for_medications_effects(phenotypes_table,
                                    phenotype_code,
                                    phenotype_name,
                                    phenotype_kind,
                                    medications,
                                    impute_missing_covariates=False,
                                    out_dir=None,
                                    genders=None,
                                    store_covariates=True,
                                    store_debug_columns=False,
                                    use_only_first_timepoint=False):

    gPCs = [('22009-0.%d' % i, UKB_GPC_PREFIX + '_%d' % i) for i in range(1, 41)]

    baseline_covariates_to_regress = [(AGE_label, 'age'),
                                      (AGE_label, np.square, 'age^2'),

                                      (54, CATEGORICAL_PHENOTYPE, 'UK_Biobank_assessment_centre'),
                                      (1239, CATEGORICAL_PHENOTYPE, 'Current_tobacco_smoking'),
                                      (1249, CATEGORICAL_PHENOTYPE, 'Past_tobacco_smoking'),
                                      (1259, CATEGORICAL_PHENOTYPE, 'Smoking_or_smokers_in_household'),
                                      (1329, CATEGORICAL_PHENOTYPE, 'Oily_fish_intake'),
                                      (1339, CATEGORICAL_PHENOTYPE, 'Non-oily_fish_intake'),
                                      (1349, CATEGORICAL_PHENOTYPE, 'Processed_meat_intake'),
                                      (1359, CATEGORICAL_PHENOTYPE, 'Poultry_intake'),
                                      (1369, CATEGORICAL_PHENOTYPE, 'Beef_intake'),
                                      (1379, CATEGORICAL_PHENOTYPE, 'Lamb_or_mutton_intake'),
                                      (1389, CATEGORICAL_PHENOTYPE, 'Pork_intake'),
                                      (1408, CATEGORICAL_PHENOTYPE, 'Cheese_intake'),
                                      (1418, CATEGORICAL_PHENOTYPE, 'Milk_type_used'),
                                      (1428, CATEGORICAL_PHENOTYPE, 'Spread_type'),
                                      (1448, CATEGORICAL_PHENOTYPE, 'Bread_type'),
                                      (1468, CATEGORICAL_PHENOTYPE, 'Cereal_type'),
                                      (1478, CATEGORICAL_PHENOTYPE, 'Salt_added_to_food'),
                                      (1518, CATEGORICAL_PHENOTYPE, 'Hot_drink_temperature'),
                                      (1538, CATEGORICAL_PHENOTYPE, 'Major_dietary_changes_in_the_last_5_years'),
                                      (1548, CATEGORICAL_PHENOTYPE, 'Variation_in_diet'),
                                      (1558, CATEGORICAL_PHENOTYPE, 'Alcohol_intake_frequency'),
                                      (1628, CATEGORICAL_PHENOTYPE, 'Alcohol_intake_versus_10_years_previously'),
                                      (1677, CATEGORICAL_PHENOTYPE, 'Breastfed_as_a_baby'),
                                      (3089, CATEGORICAL_PHENOTYPE, 'Caffeine_drink_within_last_hour'),
                                      ('21000-0.0', CATEGORICAL_PHENOTYPE, 'ethnicity'),
                                      ('22021-0.0', CATEGORICAL_PHENOTYPE, 'Genetic_kinship_to_other_participants'),
                                      ('24014-0.0', CATEGORICAL_PHENOTYPE, 'Close_to_major_road'),
                                      ('189-0.0', 'Townsend_deprivation_index_at_recruitment'),
                                      ('24003-0.0', 'Nitrogen_dioxide_air_pollution_2010'),
                                      ('24004-0.0', 'Nitrogen_oxides_air_pollution_2010'),
                                      ('24010-0.0', 'Inverse_distance_to_the_nearest_road'),
                                      ('24012-0.0', 'Inverse_distance_to_the_nearest_major_road'),
                                      ('24015-0.0', 'Sum_of_road_length_of_major_roads_within_100m'),
                                      ('24016-0.0', 'Nitrogen_dioxide_air_pollution_2005'),
                                      ('24017-0.0', 'Nitrogen_dioxide_air_pollution_2006'),
                                      ('24018-0.0', 'Nitrogen_dioxide_air_pollution_2007'),
                                      ('24019-0.0', 'Particulate_matter_air_pollution_pm10_2007'),
                                      ('24020-0.0', 'Average_daytime_sound_level_of_noise_pollution'),
                                      ('24021-0.0', 'Average_evening_sound_level_of_noise_pollution'),
                                      ('24022-0.0', 'Average_night_time_sound_level_of_noise_pollution'),
                                      ('24023-0.0', 'Average_16_hour_sound_level_of_noise_pollution'),
                                      ('24024-0.0', 'Average_24_hour_sound_level_of_noise_pollution'),
                                      (74, 'Fasting_time'),
                                      (1289, 'Cooked_vegetable_intake'),
                                      (1299, 'Salad_or_raw_vegetable_intake'),
                                      (1309, 'Fresh_fruit_intake'),
                                      (1319, 'Dried_fruit_intake'),
                                      (1438, 'Bread_intake'),
                                      (1458, 'Cereal_intake'),
                                      (1488, 'Tea_intake'),
                                      (1498, 'Coffee_intake'),
                                      (1528, 'Water_intake'),
                                      ('24009-0.0', 'Traffic_intensity_on_the_nearest_road'),
                                      ('24011-0.0', 'Traffic_intensity_on_the_nearest_major_road'),
                                      ('24013-0.0', 'Total_traffic_load_on_major_roads'),
                                      (53, DATE_PHENOTYPE, 'Date_of_attending_assessment_centre')] + gPCs

    all_field_ids = list(phenotypes_table)

    echo('Splitting into male and female samples')

    male_phenotypes = phenotypes_table[phenotypes_table[SEX_label] == 1].dropna(subset=[str(phenotype_code) + '-' + str(first_visit_index) + '.0'])
    female_phenotypes = phenotypes_table[phenotypes_table[SEX_label] == 0].dropna(subset=[str(phenotype_code) + '-' + str(first_visit_index) + '.0'])

    log_dir = out_dir + '/' + phenotype_name + '.covariate_correction_info'
    if os.path.exists(log_dir):
        echo('[WARNING]', log_dir, 'exists. Cleaning up!')
        shutil.rmtree(log_dir)

    os.mkdir(log_dir)

    for sex in genders:

        echo('Processing:', phenotype_kind, ', field_id:', phenotype_code, ', name:', phenotype_name, ', sex:', sex, ', first_visit_index:', first_visit_index)

        if sex == MALE:
            phenotypes_to_use = male_phenotypes

        elif sex == FEMALE:
            phenotypes_to_use = female_phenotypes

        else:
            phenotypes_to_use = phenotypes_table

        if len(phenotypes_to_use) < 1000:
            echo('Skipping:', phenotype_kind, ', field_id:', phenotype_code, ', name:', phenotype_name, ', sex:', sex, ', first_visit_index:', first_visit_index,
                 ', n=', len(phenotypes_to_use))
            continue

        covariates_to_regress = []

        if sex == BOTH:
            covariates_to_regress.extend([(SEX_label, 'sex'),
                                          ([AGE_label, SEX_label], lambda d: d.iloc[:, 0] * d.iloc[:, 1], 'ageXsex'),
                                          ([AGE_label, SEX_label], lambda d: d.iloc[:, 0] * d.iloc[:, 0] * d.iloc[:, 1],
                                           'age^2Xsex'),
                                          ])

        if sex in [FEMALE, BOTH]:
            covariates_to_regress.extend([(2724, CATEGORICAL_PHENOTYPE, 'Had_menopause'),
                                          (2784, CATEGORICAL_PHENOTYPE, 'taken_contraceptive_pill'),
                                          (2814, CATEGORICAL_PHENOTYPE, 'used_hormone_replacement_therapy'),
                                          (2834, CATEGORICAL_PHENOTYPE, 'both_ovaries_removed'),
                                          (3140, CATEGORICAL_PHENOTYPE, 'pregnant'),
                                          (3591, CATEGORICAL_PHENOTYPE, 'had_womb_removed')])

        covariates_to_regress.extend(baseline_covariates_to_regress)

        biomarker_kind_specific_covariates = []

        if phenotype_kind is BLOOD_BIOMARKER:
            biomarker_kind_specific_covariates = [(1, DATE_PHENOTYPE, phenotype_name + '/assay_date'),
                                                  (1, HOUR_PHENOTYPE, phenotype_name + '/assay_hour'),
                                                  (2, CATEGORICAL_PHENOTYPE, phenotype_name + '/aliquot'),
                                                  (3, CATEGORICAL_PHENOTYPE, phenotype_name + '/correction_level')]

            covariates_to_regress.append(('30897', 'sample_dilution_factor'))

        elif phenotype_kind is URINE_BIOMARKER:
            biomarker_kind_specific_covariates = [(2, DATE_PHENOTYPE, phenotype_name + '/acquisition_day'),
                                                  (2, HOUR_PHENOTYPE, phenotype_name + '/acquisition_hour'),
                                                  (3, CATEGORICAL_PHENOTYPE, phenotype_name + '/device_ID')]
            covariates_to_regress.append(('30897', 'sample_dilution_factor'))

        elif phenotype_kind is BLOOD_CELLTYPE_BIOMARKER:
            biomarker_kind_specific_covariates = [(1, lambda v: v, phenotype_name + '/freeze_thaw_cycles'),
                                                  (2, DATE_PHENOTYPE, phenotype_name + '/acquisition_date'),
                                                  (2, HOUR_PHENOTYPE, phenotype_name + '/acquisition_hour'),
                                                  (3, CATEGORICAL_PHENOTYPE, phenotype_name + '/device_ID'),
                                                  (4, CATEGORICAL_PHENOTYPE, phenotype_name + '/acquisition_route'),
                                                  ]
            covariates_to_regress.append(('30897', 'sample_dilution_factor'))

        elif phenotype_kind is GET_ALL_COVARIATES:
            covariates_to_regress.append(('30897', 'sample_dilution_factor'))

        # adding biomarker kind specific covariates
        for field_idx, field_type, field_label in biomarker_kind_specific_covariates:
            for visit in [0, 1]:
                cov_field = str(phenotype_code + field_idx) + '-%d.0' % visit
                if cov_field in all_field_ids:
                    covariates_to_regress.append((cov_field, field_type, field_label + '_%d_visit' % (visit + 1)))

        phenotype_name_sex = phenotype_name + '.' + sex

        corrected_phenotype = correct_phenotype_for_covariates_and_drug_use(phenotypes_to_use,
                                                                            sex=sex,

                                                                            phenotype_code=phenotype_code,
                                                                            phenotype_name=phenotype_name_sex,

                                                                            medications=medications,

                                                                            covariates_to_regress=covariates_to_regress,
                                                                            log_dir=log_dir,
                                                                            min_subjects_per_category=50,
                                                                            phenotype_is_binary=False,

                                                                            impute_missing_covatiates=impute_missing_covariates,
                                                                            first_visit_index=first_visit_index)

        plt.close()

        ph_col_names = [c for c in corrected_phenotype if c != SAMPLE_ID]

        if not store_covariates:
            ph_col_names = [c for c in ph_col_names if c.startswith(phenotype_name_sex)]

        if not store_debug_columns:
            ph_col_names = [c for c in ph_col_names if not (c.endswith('.original.with_med_correction.RAW') or
                                                            c.endswith('.original.with_med_correction.IRNT'))]

        if use_only_first_timepoint:
            ph_col_names = [c for c in ph_col_names if '.2nd_visit.' not in c]

        echo('Storing phenotype columns:', ph_col_names)

        corrected_phenotype = corrected_phenotype[[SAMPLE_ID] + ph_col_names]

        out_fname = out_dir + '/' + phenotype_name_sex + '.med_corrected.phenotype_values.csv.gz'
        echo('Storing corrected phenotype in:', out_fname)
        corrected_phenotype.to_csv(out_fname, sep='\t', index=False)

        out_fname = out_dir + '/' + phenotype_name_sex + '.med_corrected.phenotype_values.pickle'
        echo('Storing corrected phenotype in:', out_fname)
        corrected_phenotype.to_pickle(out_fname)

        echo('Job completed!')


def read_table(fname, sep='\t'):
    echo('Reading:', fname)
    if fname.endswith('.pickle'):
        return pd.read_pickle(fname)
    else:
        return pd.read_csv(fname, sep=sep, dtype={'eid': str, SAMPLE_ID: str}).rename(columns={'eid': SAMPLE_ID})


def process_phenotype(in_fname,
                      phenotype_code,
                      first_visit_index,
                      phenotype_name,
                      phenotype_is_binary,
                      cov_files,
                      cov_names,
                      ignore_medications,
                      use_only_first_timepoint,
                      out_dir,
                      genders=None,
                      phenotype_sep='\t',
                      store_covariates=True,
                      store_debug_columns=False):

    echo('[process_phenotype]')

    raw_phenotypes = read_table(in_fname, sep=phenotype_sep)
    gc.collect()

    # display(raw_phenotypes.head())
    second_visit_index = first_visit_index + 1
    if use_only_first_timepoint:
        second_timepoint_label = str(phenotype_code) + f'-{second_visit_index}.0'
        if second_timepoint_label in raw_phenotypes:
            echo('Ignoring data for the second time point:', second_timepoint_label)
            del raw_phenotypes[second_timepoint_label]

    echo('Reading covariates:', UKB_DATA_PATH + '/ukb31216.covariates.csv')
    covariates = read_table(UKB_DATA_PATH + '/ukb31216.covariates.csv', sep=',')

    if cov_files is not None:
        for cov_fname in cov_files:

            echo('Reading additional covariates:', cov_fname)
            _c = read_table(cov_fname)
            _c = _c[[SAMPLE_ID] + [col for col in _c if any(col.startswith(cn + '-') for cn in cov_names)]]
            covariates = pd.merge(covariates, _c, on=SAMPLE_ID, suffixes=('', '/' + cov_fname))

    raw_phenotypes = pd.merge(raw_phenotypes, covariates, on=SAMPLE_ID, suffixes=('', '/from_covariates'))

    echo('n_samples=', len(raw_phenotypes))

    echo('Filling out missing values with 0 for estrogen and rheumatoid factor')
    # fill missing values as 0 for estrogen and rheumatoid factor and female only covariates in males

    for ph_id in [30800, 30820]:
        for visit in [0, 1]:
            ph_label = str(ph_id + 1) + '-%d.0' % visit
            if ph_label not in list(raw_phenotypes):
                echo('phenotype field missing, skipping:', ph_label)
                continue

            to_fill = ~raw_phenotypes[ph_label].isnull()

            for f_idx in [0, 3, 4]:
                all_values = raw_phenotypes[str(ph_id + f_idx) + '-%d.0' % visit]
                m_value = min(all_values.dropna()) - 1

                echo('Replacing NaNs from', ph_id, f_idx, 'with', m_value if f_idx == 0 else 0)

                new_values = [v if not np.isnan(v) else (m_value if f_idx == 0 else 0) if to_fill[i] else np.nan for
                              i, v in enumerate(all_values)]

                raw_phenotypes[str(ph_id + f_idx) + '-%d.0' % visit] = new_values
    # take the average value from the two measurements for diastolic and systolic blood pressure
    echo('For systolic and diastolic blood pressure, take average of the two consecutive measurements')
    for ph_id in [4079, 4080]:
        for visit_idx in [0, 1]:
            ph_visit_id = str(ph_id) + '-' + str(visit_idx)
            if ph_visit_id + '.0' not in list(raw_phenotypes):
                echo('phenotype field missing, skipping..')
                continue

            raw_phenotypes[ph_visit_id + '.0'] = (raw_phenotypes[ph_visit_id + '.0'] + raw_phenotypes[
                ph_visit_id + '.1']) / 2
    echo('Filling out missing values for female covariates in male subjects')
    for col_id in [2724, 2784, 2794, 2804, 2814, 2834, 3140, 3591]:
        for visit in [0, 1]:
            col_on_visit = str(col_id) + '-%d.0' % visit
            to_fill = (raw_phenotypes[col_on_visit].isnull()) & (raw_phenotypes[SEX_label] == 1)
            raw_phenotypes[col_on_visit] = np.where(to_fill, 0, raw_phenotypes[col_on_visit])

    if phenotype_code is None:
        if args.icd10:

            phenotype_code = ICD10_PHENOTYPE_CODE

            cases = get_cases_for_ICD10_group(raw_phenotypes, phenotype_name, icd10_column_ids=args.icd10_column_ids)
            phenotype_name = [c for c in list(cases) if c != SAMPLE_ID][0]

            echo('Merging cases for', phenotype_name, 'with all phenotypes')
            raw_phenotypes = pd.merge(raw_phenotypes, cases, on=SAMPLE_ID).rename(
                columns={phenotype_name: str(phenotype_code) + '-0.0'})

            if len(phenotype_name) > 100:
                phenotype_name = phenotype_name[:100]

        else:
            ukb_field_names = pd.read_csv(UKB_DATA_PATH + '/Data_Dictionary_Showcase.csv', escapechar='\\')
            matching_phenotypes = ukb_field_names[
                ukb_field_names['Field'].str.replace(r'[^0-9a-zA-Z]+', '_').str.strip('_') == phenotype_name]
            pd.set_option('display.max_columns', None)

            if len(matching_phenotypes) > 1:
                echo('WARNING: cannot find unique phenotype info for:', phenotype_name)

                echo(matching_phenotypes)
                echo('Checking if any of them overlap with predefined quatitative phenotypes..')
                matching_phenotypes = matching_phenotypes[
                    matching_phenotypes['FieldID'].isin(ALL_QUANTITATIVE_PHENOTYPES)]

            if len(matching_phenotypes) == 0:
                echo('ERROR: cannot find phenotype info for:', phenotype_name)
                echo(matching_phenotypes)
                exit(1)

            phenotype_code = matching_phenotypes.iloc[0]['FieldID']

    if phenotype_code == ICD10_PHENOTYPE_CODE:
        phenotype_kind = ICD10

    elif phenotype_is_binary:
        phenotype_kind = BINARY

    elif 30000 <= phenotype_code <= 30310:
        phenotype_kind = BLOOD_CELLTYPE_BIOMARKER

    elif 30500 <= phenotype_code < 30600:
        phenotype_kind = URINE_BIOMARKER

    elif 30600 <= phenotype_code <= 30900:
        phenotype_kind = BLOOD_BIOMARKER

    elif 23400 <= phenotype_code <= 23648:
        phenotype_kind = NIGHTINGALE_HEALTH_METABOLOME

    elif phenotype_code < 0:
        phenotype_kind = GET_ALL_COVARIATES
        echo('Generating random phenotype')
        raw_phenotypes[str(phenotype_code) + '-0.0'] = np.random.rand(len(raw_phenotypes))

        raw_phenotypes[str(phenotype_code) + '-1.0'] = np.where(~raw_phenotypes[AGE_2nd_visit].isnull(),
                                                                np.random.rand(len(raw_phenotypes)),
                                                                [None] * len(raw_phenotypes))
    else:
        phenotype_kind = OTHER

    phenotype_name = phenotype_name + '.' + 'all_ethnicities'
    echo('Processing:', phenotype_name, phenotype_code, phenotype_kind)

    if ignore_medications:
        medications = {}

    else:
        echo('Loading medication information')
        ukb_medications = get_ukb_medications()
        medication_cateogories = get_per_med_group_subject_statistics(ukb_medications, raw_phenotypes)

        medications = dict((cat_name,
                            get_all_meds_in_category(ukb_medications, cat_name=cat_name, return_codes=True))

                           for cat_name in medication_cateogories[
                               (medication_cateogories[['on_meds_%d' % i for i in range(4)]] >= 50).all(axis=1)][
                               'med_category_name'])

    correct_for_medications_effects(raw_phenotypes,
                                    phenotype_code,
                                    phenotype_name,
                                    phenotype_kind,
                                    medications,
                                    impute_missing_covariates=True,
                                    out_dir=out_dir,
                                    genders=genders,
                                    store_covariates=store_covariates,
                                    store_debug_columns=store_debug_columns,
                                    use_only_first_timepoint=use_only_first_timepoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='in_fname', help='input table with raw phenotype values (csv or pickled pandas)',
                        required=True)

    parser.add_argument('--phenotype-code', dest='phenotype_code', help='phenotype_code', type=int, required=True)
    parser.add_argument('-n', dest='phenotype_name', help='phenotype name after correction', required=True)

    parser.add_argument('--covariates-file', dest='cov_files', nargs='+', help='covariates table(s) (csv or pickled pandas)')

    parser.add_argument('-c', dest='covariates', nargs='+', help='covariate columns')

    parser.add_argument('--use-only-first-timepoint', dest='use_only_first_timepoint', action='store_true', help='only use data from the first timepoint [default: False]')
    parser.add_argument('--first-visit-index', dest='first_visit_index', type=int, default=0, help='Index of the first visit [default: 0]')
    parser.add_argument('--is-binary', dest='is_binary', action='store_true', help='Phenotype is binary [default: False]')
    parser.add_argument('--ignore-medications', dest='ignore_medications', action='store_true', help='no correction for medication use [default: False]')
    parser.add_argument('--store-covariates', dest='store_covariates', action='store_true', help='store covariates in the output files [default: False]')
    parser.add_argument('--store-debug-columns', dest='store_debug_columns', action='store_true', help='store additional phenotype columns for debug purposes [default: False]')

    parser.add_argument('--output-dir', dest='output_dir', help='output dir', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()

    in_fname = args.in_fname
    cov_files = args.cov_files
    cov_names = args.covariates
    use_only_first_timepoint = args.use_only_first_timepoint
    ignore_medications = args.ignore_medications
    store_covariates = args.store_covariates
    first_visit_index = args.first_visit_index
    store_debug_columns = args.store_debug_columns

    phenotype_name = args.phenotype_name
    phenotype_code = args.phenotype_code
    phenotype_is_binary = args.is_binary

    clean_ph_name = remove_special_chars(phenotype_name) + '.' + 'all_ethnicities'

    out_dir = args.output_dir

    open_log(out_dir + '/' + clean_ph_name + '.covariate_correction.log')
    echo('CMD:', ' '.join(sys.argv))

    echo('Parameters:\n' + pprint.pformat(args.__dict__))

    process_phenotype(in_fname,
                      phenotype_code,
                      first_visit_index,
                      phenotype_name,
                      phenotype_is_binary,
                      cov_files,
                      cov_names,
                      ignore_medications,
                      use_only_first_timepoint,
                      out_dir,
                      genders=[BOTH, MALE, FEMALE],
                      store_covariates=store_covariates,
                      store_debug_columns=store_debug_columns)

    close_log()
