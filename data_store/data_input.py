'''
For pilot study of Thea's work, extract:
1. small sample of BRI timeseries datapoints

'''

# %% imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
# %% functions
def load_bri_vitals(data_source, mortality_target=False):
    '''
    Load pickle file of patient data

    Arguments
        data_source (str): name of patient data source
        target (bool): optionally include the discharge method column
    Returns
        dataset_df: dataframe of dataset
    '''

    clusterpath = Path('C:\\','Storage', 'Data','Share')

    drop_cols = ['Embedding_1', 'Embedding_2',
                 'ID',  'icd_p', 'icd_1',
       'ward',  'Spell_ID_Idx', 'ClusterEM', 'EWS_SCORE']
    
    if mortality_target:
        drop_cols.append['discharge_method']

    if data_source == 'hello':
        dataset_df = 'hello'

    elif data_source == "timeseries_2018":
        vts_clusters_pickle = pickle.load(open(clusterpath.joinpath("xumapSsid.pkl"), "rb")) # All time series data
        dataset_df = vts_clusters_pickle["xumapSsid"]
        dataset_df.drop(columns=drop_cols, inplace=True)
        dataset_df.sort_values(by='DateTime', inplace=True)
    
    elif data_source == 'bri_sample':
        dataset_df = pickle.load(open(clusterpath.joinpath("timeseries_smallsample.pkl"), "rb")) # All time series data
        dataset_df.sort_values(by='DateTime', inplace=True)

    # amend headings to fit Thea's existing setup
    
    dataset_df.rename(columns={
        'spell_id':'stay_id',
        'Temperature':'temperature',
        'BP_Systolic': 'systolic_bp',
        'Heart_Rate':'heart_rate',
        'SATS':'sats',
        'Respiratory_Rate': 'respiration',
        'DateTime': 'charttime',
    }, inplace=True)


    return dataset_df
'''
clusterpath = Path('C:\\','Storage', 'Data','Share')

# %% load BRI vitals data
data = load_bri_vitals('timeseries_2018')

# %% list ids and select random sample
np.random.seed(13)
ids = data["spell_id"].unique()

ids = np.random.choice(ids, size=100, replace=False)
# %% select associated cases
sample_cases = data.loc[data["spell_id"].isin(ids)]
# %% Check that all cases have at least 3 entries
sample_cases['spell_id'].value_counts()

# %% only use rows with more than 20 entries
counts = sample_cases['spell_id'].value_counts()
updated = sample_cases[sample_cases['spell_id'].map(counts)>20]
# %%
updated['spell_id'].value_counts()
# %%
# %% save


with open(clusterpath.joinpath('timeseries_smallsample.pkl'),'wb') as file:
    pickle.dump(updated,file)
# %%
load_bri_vitals('bri_sample')

# %%
load_bri_vitals('hello')
# %%
'''