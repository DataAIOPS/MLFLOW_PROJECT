import pandas as pd
import os
file_path = os.path.abspath("./../artifacts/data/raw_data/homeprices.csv")
df = pd.read_csv(file_path)

df.to_csv("./../artifacts/data/cleaned_data/homeprices_clean.csv")
