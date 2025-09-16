import os
import pandas as pd
import numpy as np
from extract_features import extract_mfcc

def read_protocol(protocol_file, data_dir):
    protocol = pd.read_csv(protocol_file, sep='\s+', header=None, names=['speaker_id', 'filename', 'system_id', 'env_id', 'label'])
    protocol['path'] = protocol['filename'].apply(lambda x: os.path.join(data_dir, 'flac', x + '.flac'))
    return protocol

def preprocess_data(protocol, max_length=500):
    data = []
    labels = []
    for index, row in protocol.iterrows():
        mfcc = extract_mfcc(row['path'])
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > max_length:
            mfcc = mfcc[:, :max_length]
        data.append(mfcc)
        labels.append(1 if row['label'] == 'spoof' else 0)
    return np.array(data), np.array(labels)

def normalize_data(X_train, X_dev, X_eval):
    mean = np.mean(X_train, axis=(0, 1))
    std = np.std(X_train, axis=(0, 1))
    return (X_train - mean) / std, (X_dev - mean) / std, (X_eval - mean) / std
