'''
import time
'''

# load data
import mglearn
citibike = mglearn.datasets.load_citibike()
print("Citi Bike data shape:\n{}".format(citibike.shape))
print("Citi Bike data:\n{}".format(citibike.head()))

# visualize data
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
                        freq='D')
plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Rentals")
plt.show()

# get X and y (Series: index and value)
y = citibike.values
X = citibike.index.astype("int64").values.reshape(-1, 1)

# define train data number
n_train = 184

# function to build and evaluate model
def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]

    regressor.fit(X_train, y_train)

    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    # compare predicted values and real values
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,
                ha="left")
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',
            label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()

### ------ start of building model ------###

## Model 1: random forests (require very little preprocessing of the data)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)

# Improve 1: use only the hour of the day
X_hour = citibike.index.hour.values.reshape(-1, 1)
print('X_hour: \n', X_hour[:5])
eval_on_features(X_hour, y, regressor)

# Improve 2: also add the day of the week
import numpy as np
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1),
						citibike.index.hour.values.reshape(-1, 1)])
print('X_hour_week: \n', X_hour_week[:5])
eval_on_features(X_hour_week, y, regressor)

## Model 2: LinearRegression (simpler model)
from sklearn.linear_model import LinearRegression
eval_on_features(X_hour_week, y, LinearRegression())

# Pre_Processing 1: one-hot-encoder
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
eval_on_features(X_hour_week_onehot, y, Ridge()) # use ridge - with regularization

# Pre_Processing 2: Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True,
										include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot) #如果是normalization不能train和test这么同时fit
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)

## plot coefficients learned by the model (NA for random forest)
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
features = day + hour

# name all the interaction features, and keep only the features with nonzero coefficients:
features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

# visualize the coefficients learned by the linear model
plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature name")
plt.ylabel("Feature magnitude")
plt.show()