import sys
sys.path.insert(0, '../..')

import pandas as pd
import numpy as np
from UCB_training.UCB_utils import clean_df

DATA_DIR = '../../russian_river_data'
OUTPUT_DIR = 'extreme_year_analysis/data'


class Data_Utils:
    def __init__(self):
        self.data = None

    def load_data(self):
        Calpella_data = clean_df(pd.read_csv(f'{DATA_DIR}/Calpella_daily.csv')).reset_index().rename(columns={'date': 'Date'})
        Guerneville_data = clean_df(pd.read_csv(f'{DATA_DIR}/Guerneville_daily.csv')).reset_index().rename(columns={'date': 'Date'})
        Hopland_data = clean_df(pd.read_csv(f'{DATA_DIR}/Hopland_daily.csv')).reset_index().rename(columns={'date': 'Date'})
        WarmSprings_data = clean_df(pd.read_csv(f'{DATA_DIR}/WarmSprings_Inflow_daily.csv')).reset_index().rename(columns={'date': 'Date'})

        self.data = {
            'Capella Gage FLOW': Calpella_data,
            'Guerneville Gage FLOW': Guerneville_data,
            'Hopland Gage FLOW': Hopland_data,
            'Warm Springs Dam Inflow FLOW': WarmSprings_data
        }

    def Get_Peak_Flows(self):
        """Returns a dict of peak flow dataframes for all basins, indexed by basin name."""
        for basin in self.data.keys():
            basin_df = self.data[basin][['Date', basin]].copy()
            basin_df['water_year'] = np.where(basin_df['Date'].dt.month >= 10, basin_df['Date'].dt.year + 1, basin_df['Date'].dt.year)

            peak_flows = basin_df.groupby('water_year')[basin].max().reset_index()
            peak_flows = peak_flows[peak_flows['water_year'] <= 2009]
            peak_flows.to_csv(f'{OUTPUT_DIR}/peak_flows/{basin.split()[0]}.csv')

    def Get_N_Day_Avg_Minimums(self, N):
        """Returns a dict of N-day min flow dataframes for all basins, indexed by basin name and climatic year (April 1st to March 31st)."""
        for basin in self.data.keys():
            basin_df = self.data[basin][['Date', basin]].copy()
            basin_df['climatic_year'] = np.where(basin_df['Date'].dt.month >= 4, basin_df['Date'].dt.year + 1, basin_df['Date'].dt.year)
            basin_df['mav'] = basin_df[basin].rolling(window=7, min_periods=N).mean()

            min_flows = basin_df.groupby('climatic_year')["mav"].min().reset_index()
            min_flows.to_csv(f'{OUTPUT_DIR}/low_flows_7d_avg/{basin.split()[0]}.csv')

    def Get_N_Day_Avg_Maximums(self, N):
        """Returns a dict of N-day max flow dataframes for all basins, indexed by basin name and water year (October 1st to September 30th)."""
        for basin in self.data.keys():
            basin_df = self.data[basin][['Date', basin]].copy()
            basin_df['water_year'] = np.where(basin_df['Date'].dt.month >= 10, basin_df['Date'].dt.year + 1, basin_df['Date'].dt.year)
            basin_df['mav'] = basin_df[basin].rolling(window=7, min_periods=7).mean()

            max_flows = basin_df.groupby('water_year')["mav"].max().reset_index()
            max_flows = max_flows[max_flows['water_year'] <= 2009]
            max_flows.to_csv(f'{OUTPUT_DIR}/peak_flows_7d_avg/{basin.split()[0]}.csv')


if __name__ == '__main__':
    x = Data_Utils()
    x.load_data()
    x.Get_Peak_Flows()
    x.Get_N_Day_Avg_Minimums(7)
    x.Get_N_Day_Avg_Maximums(7)
