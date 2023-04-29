import pandas as pd
import datetime
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score


def univar_box(column, ylabel):
  plt.boxplot(column)
  plt.ylabel(ylabel)
  plt.show()
  
def univar_hist(column, xlabel):
  plt.hist(column)
  plt.xlabel(xlabel)
  plt.show()

def bivar_scatter(column_x, column_y, xlabel):
  plt.scatter(column_x,column_y)
  plt.xlabel(xlabel)
  plt.ylabel("Number of subscribers")
  plt.show()

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


print(full_data.describe())

univar_box(full_data['subscriber_count'], "Subscribers")
univar_box(full_data['video_views'], "Video Views")
univar_box(full_data['video_count'], "Videos made")
univar_box(full_data['age'], "Years since started")
univar_hist(full_data['subscriber_count'], "Subscribers")
univar_hist(full_data['video_views'], "Video Views")
univar_hist(full_data['video_count'], "Videos made")
univar_hist(full_data['age'], "Years since started")

bivar_scatter(full_data['video_views'], full_data['subscriber_count'], "Video Views")
bivar_scatter(full_data['video_count'], full_data['subscriber_count'], "Videos made")
bivar_scatter(full_data['age'], full_data['subscriber_count'], "Years since started")
bivar_scatter(full_data['genre_weight'], full_data['subscriber_count'], "Occurences in top 300")

# Prediction
# The subscriber count has high propensity to get affected by view count and number of videos released.

X_1 = full_data[['video_views']]
X_2 = full_data[['video_views', 'video_count']]
X_3 = full_data[['video_views', 'video_count', 'age']]
Y = full_data['subscriber_count']

# Linear regression, by taking only view count
model_1 = LinearRegression()
model_1.fit(X_1,Y)
predictions_1 = model_1.predict(X_1)
r_value_1 = r2_score(Y, predictions_1)
print(f"R value for linear regression with video_views: {r_value_1}")

# Linear Regression, for both video_views and video_count included
model_2 = LinearRegression()
model_2.fit(X_2,Y)
predictions_2 = model_2.predict(X_2)
r_value_2 = r2_score(Y, predictions_2)
print(f"R value for linear regression with video_views and video_count: {r_value_2}")

# Linear Regression for both video_views and video_count, along with age
model_3 = LinearRegression()
model_3.fit(X_3, Y)
predictions_3 = model_3.predict(X_3)
r_value_3 = r2_score(Y, predictions_3)
print(f"R value for linear regression with video_views, video_count and age: {r_value_3}")

# Linear Regression with splitting training and testing data
X_train_lin, X_test_lin, Y_train_lin, Y_test_lin = train_test_split(X_3,Y, test_size=0.2,random_state=1)
model_lin = LinearRegression()
model_lin.fit(X_train_lin, Y_train_lin)
predictions_lin = model_lin.predict(X_test_lin)
r_value_lin = r2_score(Y_test_rf, predictions_lin)
print(f"R value for linear regression with test train split : {r_value_lin}")

# Random Forest
X_rf = full_data.drop(['subscriber_count', 'channel_name', 'genre', 'channel_started'], axis=1)
Y_rf = full_data['subscriber_count']
X_train_rf,X_test_rf,Y_train_rf,Y_test_rf = train_test_split(X_rf, Y_rf, test_size=0.2, random_state=1)
forest = RandomForestClassifier()
forest.fit(X_train_rf, Y_train_rf)
predictions_rf = forest.predict(X_test_rf)
r_value_rf = r2_score(Y_test_rf, predictions_rf)
print(f"R value using Random Forest : {r_value_rf}") # Clear case of over-fitting when train_test_split is not used

# Gradient Boosting Regression
gbr = GradientBoostingRegressor()
gbr.fit(X_train_rf, Y_train_rf)
predictions_gbr = gbr.predict(X_test_rf)
r_value_gbr = r2_score(Y_test_rf, predictions_gbr)
print(f"R value when Gradient Boosting Regression is used: {r_value_gbr}")
