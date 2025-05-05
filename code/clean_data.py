import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))


input_path = os.path.join(base_dir, "../data/test_data.csv")
output_path = os.path.join(base_dir, "../data/test_data_clean.csv")


df = pd.read_csv(input_path, sep=";", header=0, names=["comment", "isHate"], encoding="utf-8")
df.to_csv(output_path, index=False)
