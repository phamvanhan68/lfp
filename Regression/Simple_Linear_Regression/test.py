import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


X_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = x_0 + 2*x_1 + 3
y_train = np.dot(X_train, np.array([1, 2])) + 3

X_test = np.array([[3, 5], [4,5]])
y_test = [16., 17.]

reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
print(reg.score(X_train, y_train))
# 1.0
reg.coef_
print("coef_", reg.coef_)
# array([1., 2.])
reg.intercept_
print("intercept_", reg.intercept_)
# 3.0000...
# y = 1*x_0 + 2*x_1 + 3 => intercept is 3

pre = reg.predict(X_test)
print("pre",pre)
# array([16.])

# Visualising the Testing set results
print("X_test", X_test)
print("y_test", y_test)
print(X_train)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'yellow')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()