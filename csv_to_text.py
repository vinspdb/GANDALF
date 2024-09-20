import numpy as np
import pandas as pd
from utility import skeleton
from preprocessing import sentineltotext

dataset_dir = 'CZ_200_Settembre_Tiling-2'
baseline = 'baseline-12-s'

df_train = pd.read_csv(dataset_dir+'/train/bands_12.csv', header=None)
df_test = pd.read_csv(dataset_dir+'/test/bands_12.csv', header=None)

mask_train = pd.read_csv(dataset_dir+'/train/masks_12.csv', header=None)
mask_test = pd.read_csv(dataset_dir+'/test/masks_12.csv', header=None)

df_train.columns = skeleton.skeleton[baseline]['feature']
df_test.columns = skeleton.skeleton[baseline]['feature']

train = sentineltotext.SentinelToText(df_train, baseline, 'train', dataset_dir)
test = sentineltotext.SentinelToText(df_test, baseline, 'test', dataset_dir)