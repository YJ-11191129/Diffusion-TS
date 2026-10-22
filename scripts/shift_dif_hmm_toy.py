"""Create reproducible synthetic inputs and the exact ShiftDiF file bundle on CPU."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Utils.Data_utils.causal_hmm import generate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', default='Data/datasets')
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(123)
dates = pd.bdate_range('2010-01-01', periods=500)
regime = (np.arange(500) // 50) % 3
market = rng.normal(0, .005 + .005 * regime)
features = pd.DataFrame({'date': dates, 'market_return': market,
                         'volatility': pd.Series(market).rolling(5, min_periods=1).std().fillna(0)})
returns = pd.DataFrame(market[:, None] + rng.normal(0, .003, (500, 6)),
                       columns=['asset_' + str(i) for i in range(6)])
returns.insert(0, 'date', dates)
features.to_csv(out / 'market_features.csv', index=False)
returns.to_csv(out / 'dated_asset_returns.csv', index=False)
generate(out / 'market_features.csv', out / 'dated_asset_returns.csv', out,
         ['market_return', 'volatility'], fit_rows=100, proportion=.9)
print('Wrote synthetic HMM states and aligned returns to', out)
