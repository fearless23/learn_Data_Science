import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data_path = "/home/fearless/devCode/learn/learn_17_data_analysis/src/data/"
fileName = "us_elections_result.csv"
df = pd.read_csv(data_path+fileName)
cond = df['year'] == 2008
cols = ['state', 'vote_dem', 'vote_rep', 'vote_total', 'pct_rep', 'pct_dem']

# Year 2008 Data, all states and counties.
df_year = df.loc[cond][cols]

# Counties data but limited to few states only.
states = ['PA', 'OH', 'FL']
cond2 = df_year['state'].isin(states)
df_swing = df_year.loc[cond2][cols]

# State Wise: All counties combined into state.
df_year_state = df_year.groupby('state', as_index=False).sum()
df_year_state['dem_share'] = df_year['vote_dem'].divide(df_year['vote_total'])
df_year_state['rep_share'] = df_year['vote_rep'].divide(df_year['vote_total'])
dff = df_year_state[['state', 'dem_share', 'rep_share']]

# Compute number of data points and bins
n_data = len(df_year['pct_dem'])
n_bins = int(np.sqrt(n_data))


def drawHistogramDem(t, n):
    """Show Histogram"""
    plt.subplot(t, 1, n)
    n_data = len(df_year['pct_dem'])
    n_bins = int(np.sqrt(n_data))
    _ = plt.hist(df_year['pct_dem'], bins=n_bins)
    _ = plt.xlabel("Percent of vote for Democrats i.e Obama")
    _ = plt.ylabel("Number of Counties")


def drawBeeswarmPlot(t, n):
    """Show Bee Swarn Plot"""
    plt.subplot(t, 1, n)
    _ = sns.swarmplot(x="state", y="pct_dem", data=df_swing)
    _ = plt.xlabel("Percent of vote for Democrats i.e Obama")
    _ = plt.ylabel("Number of Counties")


def ecdf(t, n):
    plt.subplot(t, 1, n)
    x1 = np.sort(df_year['pct_dem'])
    y1 = np.arange(1, len(x1)+1)/len(x1)  # len(x)=10, y = [0.1, 0.2,...,0.9]
    x2 = np.sort(df_year['pct_rep'])
    y2 = np.arange(1, len(x2)+1)/len(x2)  # len(x)=10, y = [0.1, 0.2,...,0.9]

    _ = plt.plot(x1, y1, marker=".", linestyle='none', label="Democrats Share")
    _ = plt.plot(x2, y2, marker=".", linestyle='none',
                 label="Republicans Share")
    _ = plt.xlabel('percent of votes for obama')
    _ = plt.ylabel('ECDF')
    plt.margins(0.02)  # Keeping data off plot edges
    plt.legend(loc='best')
    plt.show()


drawHistogramDem(3, 1)
drawBeeswarmPlot(3, 2)
ecdf(3, 3)
plt.show()
print("DONE")
