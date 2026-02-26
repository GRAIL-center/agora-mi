import pandas as pd
import sys, os
os.chdir(sys.argv[1])

for L in [1,12,16,20,24]:
    path = f'results/polarization/layer{L}_train_delta.csv'
    try:
        df = pd.read_csv(path)
        n_surv = int(df['fdr_reject'].sum())
        n_tot = len(df)
        top_d = df['delta'].iloc[0]
        min_p = df['p_value'].min()
        min_q = df['q_value'].min()
        print(f'Layer {L:2d}: {n_tot:5d} feats | top_delta={top_d:7.4f} | min_p={min_p:.6f} | min_q={min_q:.6f} | FDR_survivors={n_surv}')
    except Exception as e:
        print(f'Layer {L:2d}: ERR {e}')
