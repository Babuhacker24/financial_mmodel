from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "log_returns_wide.csv")

print(df.head())
print(df.shape)