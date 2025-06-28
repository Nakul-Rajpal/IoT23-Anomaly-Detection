import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

data_dir = 'data'
parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))

combined_datasets = {}

benign_path = os.path.join(data_dir, 'PartOfAHorizontalPortScan-002.parquet')
benign_df = pd.read_parquet(benign_path)
combined_datasets['part'] = benign_df

for filepath in parquet_files:
    filename = os.path.basename(filepath)
    if filename == 'PartOfAHorizontalPortScan-002.parquet':
        continue
    base = filename.split('.')[0].split('_')[0]
    if base not in combined_datasets:
        combined_datasets[base] = []
    combined_datasets[base].append(filepath)

for base, files in combined_datasets.items():
    if base == 'part':
        continue
    dfs = [pd.read_parquet(f) for f in sorted(files)]
    combined_datasets[base] = pd.concat(dfs, ignore_index=True)

final_df = pd.concat(combined_datasets.values(), ignore_index=True)
final_df.to_parquet("final_combined.parquet", index=False)

print(f"\nFinal combined shape: {final_df.shape}")
print("Grouped dataset keys:", combined_datasets.keys())

benign_df = combined_datasets['part']
train_benign, test_benign = train_test_split(benign_df, test_size=0.2, random_state=42)

malicious_df = pd.concat(
    [df for name, df in combined_datasets.items() if name != 'part'],
    ignore_index=True
)

test_malicious = malicious_df.sample(n=len(test_benign), random_state=42)
X_test = pd.concat([test_benign, test_malicious], ignore_index=True)
y_test = [0] * len(test_benign) + [1] * len(test_malicious)

features = ['id.orig_p', 'id.resp_p', 'proto', 'conn_state', 'orig_pkts', 'orig_ip_bytes']

for col in ['proto', 'conn_state']:
    le = LabelEncoder()
    le.fit(pd.concat([train_benign[col], X_test[col]], ignore_index=True))
    train_benign[col] = le.transform(train_benign[col])
    X_test[col] = le.transform(X_test[col])

scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_benign[features])
X_test_scaled = scaler.transform(X_test[features])

X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=42)

print(f"train_X shape: {train_X.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
print(f"y_test length: {len(y_test)}")

np.save("train_X.npy", train_X)
np.save("X_test_scaled.npy", X_test_scaled)
np.save("y_test.npy", y_test)
