# make_smoke_data.py
import pandas as pd, numpy as np, pathlib

rng = np.random.default_rng(42)
peptides = ["ACDEF", "GHILL", "MWQRK", "STYVP", "ACGIL", "DEFMW", "QRKST", "YVPAC"]

rows = []
for serial, pep in enumerate(peptides):
    for step in range(1, len(pep) + 1):
        rows.append({
            "serial": serial,
            "peptide": pep[:step],          # растущий N-мер
            "full_peptide": pep,
            "aggregation": bool(rng.integers(0, 2)),
            "first_diff_clean": float(rng.normal(0, 0.3)),
            "split": serial % 5,
        })

pathlib.Path("../../../data/smoke_data").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("../../../data/smoke_data/reactions.csv", index=False)
print("Done")