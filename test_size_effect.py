import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.mode_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from matplotlib import pyplot as plt

def lin_reg_r2(X_lin, Y_lin, testSize):
  X_train, X_test, Y_train, Y_test = train_test_split(X_lin, Y_lin, test_size = testSize, random_state=1)
  model = LinearRegression()
  model.fit(X_train, Y_train)
  Y_predicted = model.predict(X_test)
  R2_score = r2_score(Y_test, Y_predicted)
  return R2_score

def rf_r2(X_rf, Y_rf, testSize):
  X_train, X_test, Y_train, Y_test = train_test_split(X_rf, Y_rf, test_size = testSize, random_state=1)
  model = RandomForestClassifier()
  model.fit(X_train, Y_train)
  Y_predicted = model.predict(X_test)
  R2_score = r2_score(Y_test, Y_predicted)
  return R2_score

def gbr_r2(X_gbr, Y_gbr, testSize):
  X_train, X_test, Y_train, Y_test = train_test_split(X_gbr, Y_gbr, test_size = testSize, random_state=1)
  model = GradientBoostingRegressor()
  model.fit(X_train, Y_train)
  Y_predicted = model.predict(X_test)
  R2_score = r2_score(Y_test, Y_predicted)
  return R2_score
  
  
  

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

# setting up variables
X_lin = full_data[['video_count', 'video_views', 'age']]
Y_lin = full_data['subscriber_count']
X_rf = full_data.drop(['subscriber_count', 'channel_name', 'genre', 'channel_started'], axis=1)
Y_rf = full_data['subscriber_count']
X_gbr = full_data.drop(['subscriber_count', 'channel_name', 'genre', 'channel_started'], axis=1)
Y_gbr = full_data['subscriber_count']

lin_R2_scores = list()
rf_R2_scores = list()
gbr_R2_scores = list()

for i in range(1,20):
  lin_R2_scores.append(lin_reg_r2(X_lin, Y_lin, i/20))
  rf_R2_scores.append(rf_r2(X_rf,Y_rf, i/20))
  gbr_R2_scores.append(gbr_r2(X_rf, Y_rf, i/20))

# Plotting graph for comparison of R2 scores between Linear Regrsssion, Random Forest and GBR over various test sizes
plt.plot([100*i/20 for i in range(1,20)], lin_R2_scores, label="Lin_Reg")
plt.plot([100*i/20 for i in range(1,20)], rf_R2_scores, label= "RF")
plt.plot([100*i/20 for i in range(1,20)], gbr_R2_scores, label= "GBR")
plt.xlabel("Test size % of total data")
plt.ylabel("R2_score")
plt.legend()
plt.show()
