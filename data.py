import pandas as pd
import glob
import os
from collections import defaultdict

data_dir = 'data'
parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))
grouped_dfs = defaultdict(list)
for filepath in parquet_files:
    filename = os.path.basename(filepath)
    base = filename.split('.')[0].split('_')[0]
    grouped_dfs[base].append(filepath)

combined_datasets = {}

if 'part' in grouped_dfs:
    part_dfs = [pd.read_parquet(f) for f in sorted(grouped_dfs['part'])]
    combined_datasets['part'] = pd.concat(part_dfs, ignore_index=True)
for base, files in grouped_dfs.items():
    if base == 'part':
        continue
    dfs = [pd.read_parquet(f) for f in sorted(files)]
    combined_datasets[base] = pd.concat(dfs, ignore_index=True)

final_df = pd.concat(combined_datasets.values(), ignore_index=True)
print(f"\nFinal combined shape: {final_df.shape}")


print(combined_datasets)


