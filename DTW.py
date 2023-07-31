'''
Working on server with 'thea_env' environment

'''


# %%
import pandas as pd
import numpy as np
from data_store.data_input import load_bri_vitals
import dtaidistance
import itertools
import os

# %% Functions
def fill_mean(df, lst_features):
    for feature in lst_features:
        # give flexibility for MIMIC vs BRI consciousness features
        if feature in df:
            df[feature] = df[feature].fillna(df.groupby('stay_id')[feature].transform('mean'))
    return df


def fill_mode(df, lst_features):
    for feature in lst_features:
        df.loc[df[feature].isnull(), feature] = df['stay_id'].map(
            fast_mode(df, ['stay_id'], feature).set_index('stay_id')[feature])
    return df


def fast_mode(df, key_cols, value_col):
    return (df.groupby(key_cols + [value_col]).size()
              .to_frame('counts').reset_index()
              .sort_values('counts', ascending=False)
              .drop_duplicates(subset=key_cols)).drop(columns='counts')


def depen_dtw(df, patients):
    dictionary = {}
    for stay in patients:
        dictionary["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'MIN(temperature)', 'MIN(respiration)', 'MIN(heart_rate)', 'MIN(sats)', 'MIN(systolic_bp)',
             'MIN(gcs_eye)', 'MIN(gcs_verbal)', 'MIN(gcs_motor)']].sort_values(by='charttime').drop('charttime',
                                                                                                    axis=1).to_numpy()
    variable_names = list(dictionary.keys())
    combs = list(itertools.combinations(variable_names, 2))
    n = len(patients)
    # df_stay_id = pd.DataFrame.from_records(variable_names, columns=['stay_id','results'])
    stay = pd.DataFrame([dictionary]).melt()
    dist = lambda a, b: dtw_ndim.distance(a, b)
    distance_vectors = [dist(dictionary[pair[0]], dictionary[pair[1]]) for pair in combs]
    # DTW_distance_matrix = squareform(np.array(distance_vectors))
    DTW_distance_matrix = pd.DataFrame(squareform(np.array(distance_vectors)), columns=stay.variable.unique(),
                                       index=stay.variable.unique())
    #DTW_distance_matrix.to_csv('out_dtw_distance_matrix', index=False)
    return DTW_distance_matrix


def indep_dtw(df, patients, bri_data = True):
    '''
    
    Args:
        df (dataframe): rows of patient features
        patients (lst): unique patient stays
        bri_data (bool): toggle for BRI or MIMIC, for consciousness variables
    
    '''
    dictionary = {}
    for stay in patients:
        if bri_data:
            dictionary["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'temperature', 'respiration', 'heart_rate', 'sats', 'systolic_bp',
             'AVPU']
             ].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        else:
            dictionary["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'temperature', 'respiration', 'heart_rate', 'sats', 'systolic_bp',
             'gcs_eye', 'gcs_verbal','gcs_motor']
             ].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        variable_names = list(dictionary.keys())
        combs = list(itertools.combinations(variable_names, 2))
        n = len(patients)
        temperature = {}
        respiration = {}
        heart_rate = {}
        sats = {}
        systolic_bp = {}
        if bri_data:
            avpu = {}
        else:
            eye_gcs = {}
            verbal_gcs = {}
            motor_gcs = {}
    
    for stay in unique_patient_stays:
        temperature["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'temperature']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        respiration["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'respiration']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        heart_rate["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'heart_rate']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        sats["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[['charttime', 'sats']].sort_values(
            by='charttime').drop('charttime', axis=1).to_numpy()
        systolic_bp["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'systolic_bp']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        
        if bri_data:
            avpu["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
                ['charttime', 'AVPU']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        
        else:
            eye_gcs["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[['charttime', 'gcs_eye']].sort_values(
                by='charttime').drop('charttime', axis=1).to_numpy()
            verbal_gcs["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
                ['charttime', 'gcs_verbal']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
            motor_gcs["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[['charttime', 'gcs_motor']].sort_values(
                by='charttime').drop('charttime', axis=1).to_numpy()

    if bri_data:
        vitals = [temperature, respiration, heart_rate, sats, systolic_bp,
                  avpu]
    else:
        vitals = [temperature, respiration, heart_rate, sats, systolic_bp,
                  eye_gcs, verbal_gcs, motor_gcs]
    t = list(temperature.values())
    DTW_t = dtaidistance.dtw.distance_matrix_fast(t, parallel=True)
    
    print('done')

    r = list(respiration.values())
    DTW_r = dtaidistance.dtw.distance_matrix_fast(r, parallel=True)
    hr = list(heart_rate.values())
    DTW_hr = dtaidistance.dtw.distance_matrix_fast(hr, parallel=True)
    s = list(sats.values())
    DTW_s = dtaidistance.dtw.distance_matrix_fast(s, parallel=True)
    sb = list(systolic_bp.values())
    DTW_sb = dtaidistance.dtw.distance_matrix_fast(sb, parallel=True)
    if bri_data:
        avpu = list(avpu.values())
        DTW_avpu = dtaidistance.dtw.distance_matrix_fast(avpu, parallel=True)
        total_DTW_matrix = DTW_t + DTW_r + DTW_hr + DTW_s + DTW_sb + DTW_avpu
        add_100 = lambda i: i / 6
    else:
        eg = list(eye_gcs.values())
        DTW_eg = dtaidistance.dtw.distance_matrix_fast(eg, parallel=True)
        vg = list(verbal_gcs.values())
        DTW_vg = dtaidistance.dtw.distance_matrix_fast(vg, parallel=True)
        mg = list(motor_gcs.values())
        DTW_mg = dtaidistance.dtw.distance_matrix_fast(mg, parallel=True)
        total_DTW_matrix = DTW_t + DTW_r + DTW_hr + DTW_s + DTW_sb + DTW_eg + DTW_vg + DTW_mg
        add_100 = lambda i: i / 8
    vectorized_add_100 = np.vectorize(add_100)
    average_DTW_matrix = vectorized_add_100(total_DTW_matrix)
    return average_DTW_matrix


# %%
if __name__ == '__main__':
    
    df = load_bri_vitals('bri_sample')

    #unique_patient_stays = pd.read_csv('/Users/theabarnes/Documents/Masters/Technical Project/6000_unique_patient_ids.csv')
    unique_patient_stays = df['stay_id'].unique()

    vitals = ['temperature', 'sats','systolic_bp','respiration','heart_rate']
    df = fill_mean(df, vitals)

    gcs_vitals = ['gcs_eye','gcs_verbal','gcs_motor']
    df = fill_mean(df, gcs_vitals)

    df = df.dropna()
    df['charttime'] = pd.to_datetime(df['charttime'])
    df = df.sort_values(by='charttime')

    print(df.head(2))

    unique_patient_stays = unique_patient_stays.tolist()
    
    average_dtw_matrix = indep_dtw(df, unique_patient_stays)
    print(average_dtw_matrix)

    # np.savetxt('bri_sample_dtw.csv',  average_dtw_matrix)


# %%
