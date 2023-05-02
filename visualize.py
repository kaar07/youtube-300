import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Data loading and modifications

# Setting up
full_data = pd.read_csv('top-300-youtube-channels.csv');
# Removing extra index column 
full_data.drop(['Unnamed: 0'], axis=1, inplace=True)
full_data.columns = [column.lower().replace(' ','_') for column in full_data.columns]
# Adding new columns (improv)
relate = dict(full_data['genre'].value_counts())
full_data['genre_weight'] = full_data['genre'].map(relate)
full_data['age'] = [datetime.datetime.now().year - started_in for started_in in full_data['channel_started']]
# Adjusting rank order since subscriber_count was updated.
full_data.sort_values(['subscriber_count'], ascending=False, inplace=True)
full_data['rank'] = range(1,len(full_data)+1)

print(f"General info :\n{full_data.info()}\n")
print(f"Null data info:\n{full_data.isnull().sum()}\n")
print(f"Statistics :\n{full_data.describe()}\n")

# Viz: Genre types in top 300 youtube channels
sns.countplot(full_data, x="genre")
plt.xticks(rotate=90)
plt.show()

# Viz: Histograms of individual columns
full_data.hist(bins=50, figsize=(15,15))

# Pair plot for bivariate analysis
sns.pairplot(full_data)
plt.show()

# Viz: Coorelation
sns.heatmap(full_data.corr(),annot=True)
plt.show()

# Scatterplots
sns.scatterplot(full_data, x='video_views', y='subscriber_count', hue='age')
plt.show()
sns.scatterplot(full_data, x='subscriber_count', y='rank', hue='rank')
plt.show()
