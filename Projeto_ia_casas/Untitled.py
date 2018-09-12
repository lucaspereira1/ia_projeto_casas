import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingRegressor
np.set_printoptions(threshold=np.nan)




df = pd.read_csv("./precos_casa_california.csv")
df.head()
df.describe()
df.info()

# inland 0
# 1h ocena 1 
# near bay 2
# near ocean 3
# Island 4

df = df.replace('INLAND', 0)
df = df.replace('<1H OCEAN', 1)
df = df.replace('NEAR BAY', 2)
df = df.replace('NEAR OCEAN', 3)
df = df.replace('ISLAND', 4)

cols = df.columns
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

df = df.dropna(axis=0)

df.info()

%matplotlib inline
import matplotlib.pyplot as plt
df.hist(bins=30, figsize=(10,15))
plt.show()

df = df.drop(columns=['latitude', 'longitude'])
df.head()

dfout = df
cleanCols = dfout.columns
for i in cleanCols:
   dfout = dfout.loc[dfout[i] < dfout[i].quantile(0.95)]
dfout.info()


features = list(dfout.columns[0:8])
features
x = dfout[features]

# y_name = dfout['median_house_value']
x = dfout.drop(columns=['median_house_value'])
y = dfout['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
# Create linear regression object
regr = GradientBoostingRegressor()

# Train the model using the training sets
model = regr.fit(x_train, y_train)

# Make predictions using the testing set
# y_pred = regr.predict(x_test)

# score
score = model.score(x_test, y_test)
print round(score*100), '%'



