import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# The data
with open('AAPL.csv','r') as f:
    r = f.readlines()
    r = [r[i][:-1:].split(',') for i in range(len(r))]

    for i in range(len(r)):
        r[i][0],r[i][1] = int(r[i][0]),float(r[i][1])

X = np.array([r[i][0] for i in range(len(r))]).reshape(-1, 1)
Y = np.array([r[i][1] for i in range(len(r))]).reshape(-1, 1)


# Transform X to polynomial features
poly = PolynomialFeatures(degree=38)
X_poly = poly.fit_transform(X)

# Fit the model to the data
reg = LinearRegression()
reg.fit(X_poly, Y)

# Predict the value for an input of 11.5
X_test = poly.transform(np.array([[23.5]]))
prediction = reg.predict(X_test)

print("The prediction for an input of 23.5 is:", prediction)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit the model to the training data
reg.fit(X_train, Y_train)

# Predict the outputs for the test set
Y_pred = reg.predict(X_test)

# Calculate the Mean Absolute Error
mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error:", mae)

# Calculate the Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Calculate the R-squared score
r2 = r2_score(Y_test, Y_pred)
print("R-squared score:", r2)

