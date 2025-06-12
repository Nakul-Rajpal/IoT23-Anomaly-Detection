import pandas as pd

df = pd.read_csv("conn.log.labeled",sep='\t',comment='#',low_memory=False)