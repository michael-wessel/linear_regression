import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
%matplotlib inline

# Convert sklearn boston.data dictionary to Pandas dataframe
boston = load_boston()
bos = pd.DataFrame(boston.data)

# Replace column numbers with feature names
bos.columns = boston.feature_names

# Boston housing prices
boston.target[:5]

# Add target prices to the bos dataframe
bos['PRICE'] = boston.target

# Drop the price column because it isnt a parameter
X = bos.drop('PRICE', axis=1)

# Create the linear regression object
lm = LinearRegression()

# fit the model 
lm.fit(X, bos.PRICE)

# Print the intercept and number of coefficients
print('Estimated intercept coefficient:', lm.intercept_)
print('Number of coefficients:', len(lm.coef_))

# Construct the dataframe containing the features and estimated coefficients
pd.DataFrame(zip(X.columns, lm.coef_), columns=['features', 'estimatedCoefficients'])

# Scatter plot of true housing price and rooms (RM)
plt.scatter(bos.RM, bos.PRICE, color='teal')
plt.xlabel('Average number of rooms per house (RM)')
plt.ylabel('Price')
plt.title('Relationshiop between rooms and price')
plt.show()

# Predict housing price
lm.predict(X)[0:5]

# Scatter plot of true price and predicted price
plt.scatter(bos.PRICE, lm.predict(X), color='red')
plt.xlabel('Prices: $Y_i$')
plt.ylabel('Predicted prices: $\hat{Y}_i$')
plt.title('Actual vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$')
plt.show()

# Calculate the mean squared error
mseFull = np.mean((bos.PRICE - lm.predict(X)) ** 2)
print(mseFull)

# Calculate the mean squared on PTRATIO
lm = LinearRegression()
lm.fit(X[['PTRATIO']], bos.PRICE)
msePTRATIO = np.mean((bos.PRICE - lm.predict(X[['PTRATIO']])) ** 2)
print(msePTRATIO)

# Training test on train-test split data
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, bos.PRICE, test_size=0.33,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Building a linear regression model using train-test data
lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

# Calculate the mean squared error for the training and test data
print('Fit a model X_train, and calculate MSE with Y_train:',
    np.mean((Y_train - lm.predict(X_train)) ** 2))
print('Fit a model X_train, and calculate MSE with X_test, Y_test:', 
    np.mean((Y_test - lm.predict(X_test)) ** 2))

# Print residual plots
plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, color='teal', s=40, alpha=0.5)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, color='red', s=40)
plt.hlines(y=0, xmin=0, xmax=50) 
plt.title('Residual Plot using training (teal) and test (red) data')
plt.ylabel('Residuals')
plt.show()