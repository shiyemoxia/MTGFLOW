import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from pathlib import Path


WADI_META_COLS = [
    "Row",
    "Date",
    "Time",
    "2_LS_001_AL",
    "2_LS_002_AL",
    "2_P_001_STATUS",
    "2_P_002_STATUS",
]
WADI_TIMESTAMP_ORIGIN = "2017-10-09 00:00:00"


def _read_wadi_csv(path, attack=False):
    csv_path = Path(path)
    read_kwargs = {"sep": ","}
    if "lable" in csv_path.name.lower():
        read_kwargs["skiprows"] = 1
    data = pd.read_csv(csv_path, **read_kwargs)
    data.columns = [str(col).strip() for col in data.columns]
    return data


def _build_wadi_timestamp(length):
    return pd.date_range(WADI_TIMESTAMP_ORIGIN, periods=int(length), freq="s")


def _extract_wadi_attack_labels(data):
    label_col = next((col for col in data.columns if "Attack LABLE" in str(col)), None)
    if label_col is None:
        raise ValueError("WADI attack file is missing the official Attack LABLE column.")

    raw = pd.to_numeric(data[label_col], errors="coerce").fillna(1)
    labels = (raw == -1).astype(int).tolist()
    return data.drop(columns=[label_col]), labels


def _drop_wadi_meta_columns(data):
    drop_cols = [col for col in WADI_META_COLS if col in data.columns]
    return data.drop(columns=drop_cols)


def _drop_wadi_excluded_columns(data, exclude_cols=None):
    if not exclude_cols:
        return data
    normalized = {str(col).strip() for col in exclude_cols if str(col).strip()}
    drop_cols = [col for col in data.columns if str(col).strip() in normalized]
    if drop_cols:
        data = data.drop(columns=drop_cols)
    return data


def _drop_nan_rows(df, labels):
    row_mask = ~df.isna().any(axis=1)
    clean_df = df.loc[row_mask]
    clean_labels = np.asarray(labels, dtype=np.int32)[row_mask.to_numpy()]
    return clean_df, clean_labels.tolist()


def _resolve_wadi_paths(root):
    root_path = Path(root) if root else Path("Data/input")
    if root_path.is_file():
        if "attack" in root_path.name.lower():
            attack_csv = root_path
            base_dir = root_path.parent
            train_candidates = [
                base_dir / "WADI_14days.csv",
                base_dir / "WADI_14days_new.csv",
            ]
        else:
            base_dir = root_path.parent
            attack_candidates = [
                base_dir / "WADI_attackdata.csv",
                base_dir / "WADI_attackdataLABLE.csv",
            ]
            attack_csv = next((p for p in attack_candidates if p.exists()), attack_candidates[0])
            train_candidates = [root_path]
    else:
        base_dir = root_path
        attack_candidates = [
            base_dir / "WADI_attackdata.csv",
            base_dir / "WADI_attackdataLABLE.csv",
        ]
        attack_csv = next((p for p in attack_candidates if p.exists()), attack_candidates[0])
        train_candidates = [
            base_dir / "WADI_14days.csv",
            base_dir / "WADI_14days_new.csv",
        ]

    train_csv = next((p for p in train_candidates if p.exists()), train_candidates[0])
    return {
        "train_csv": str(train_csv),
        "attack_csv": str(attack_csv),
    }

def loader_WADI_OCC(root, batch_size, window_size, stride_size, train_split, label=False, exclude_cols=None):
    paths = _resolve_wadi_paths(root)
    data = _read_wadi_csv(paths["train_csv"])#, nrows=1000)
    Timestamp = _build_wadi_timestamp(len(data))
    data = _drop_wadi_meta_columns(data)
    data = _drop_wadi_excluded_columns(data, exclude_cols)
    labels = [0] * len(data)

    data = data.astype(float)
    n_sensor = len(data.columns)

    print('sensor',n_sensor)
    #%%
    feature = data

    min_scaler = StandardScaler()
    feature = min_scaler.fit_transform(feature)
    

    norm_feature = pd.DataFrame(feature, index = Timestamp, columns=data.columns)
    norm_feature, labels = _drop_nan_rows(norm_feature, labels)


    train_df = norm_feature.iloc[:]
    train_label = labels[:]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))
  
   
    val_df = norm_feature.iloc[int(train_split*len(norm_feature)):]
    val_label = labels[int(train_split*len(norm_feature)):]
    data = _read_wadi_csv(paths["attack_csv"])#, nrows=1000)
    data, labels = _extract_wadi_attack_labels(data)
    Timestamp = _build_wadi_timestamp(len(data))
    data = _drop_wadi_meta_columns(data)
    data = _drop_wadi_excluded_columns(data, exclude_cols)
    data = data.astype(float)
    n_sensor = len(data.columns)

    #%%
    feature = data
    min_scaler = StandardScaler()
    feature = min_scaler.fit_transform(feature)
    

    norm_feature = pd.DataFrame(feature, index = Timestamp, columns=data.columns)
    norm_feature, labels = _drop_nan_rows(norm_feature, labels)

    test_df = norm_feature.iloc[int(train_split*len(norm_feature)):]
    test_label = labels[int(train_split*len(norm_feature)):]
    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))

    if label:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WADI_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WADI_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor



def loader_WADI(root, batch_size, window_size, stride_size,train_split,label=False, exclude_cols=None):
    paths = _resolve_wadi_paths(root)
    data = _read_wadi_csv(paths["attack_csv"])#, nrows=1000)
    data, labels = _extract_wadi_attack_labels(data)
    Timestamp = _build_wadi_timestamp(len(data))
    data = _drop_wadi_meta_columns(data)
    data = _drop_wadi_excluded_columns(data, exclude_cols)

    data = data.astype(float)

    n_sensor = len(data.columns)

    print('sensor',n_sensor)

    feature = data
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, index = Timestamp, columns=data.columns)
    norm_feature, labels = _drop_nan_rows(norm_feature, labels)


    train_df = norm_feature.iloc[:int(train_split*len(norm_feature))]
    train_label = labels[:int(train_split*len(norm_feature))]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))

    val_df = norm_feature.iloc[int(0.6*len(norm_feature)):int(0.8*len(norm_feature))]
    val_label = labels[int(0.6*len(norm_feature)):int(0.8*len(norm_feature))]

    test_df = norm_feature.iloc[int(train_split*len(norm_feature)):]
    test_label = labels[int(train_split*len(norm_feature)):]
    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))

    if label:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WADI_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WADI_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

class WADI_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(WADI_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
        print('label', self.label.shape, sum(self.label)/len(self.label))
        print('idx',self.idx.shape)
        print('data',self.data.shape)

    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        
        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        start_index = start_idx[idx_mask]
        
        label = [0 if sum(label[index:index+self.window_size]) == 0 else 1 for index in start_index ]
        return df.values, start_idx[idx_mask], np.array(label)


    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        """
        """
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1), self.label[index], index
