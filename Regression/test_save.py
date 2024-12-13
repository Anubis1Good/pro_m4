from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression as Model
# from sklearn.neighbors import KNeighborsRegressor as Model
# from sklearn.tree import DecisionTreeRegressor as Model
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error

model = load('Regression.joblib')


df_test = pd.read_csv('test.csv')
X_test = df_test[['x']]
y_pred = model.predict(X_test)

print(mean_absolute_percentage_error(df_test.y,y_pred))
print(mean_squared_error(df_test.y,y_pred))
# print(y_pred)
# print(df_test.y.to_numpy())
# df_test.info()
df_test.plot(kind='scatter',x='x',y='y')
plt.scatter(X_test,y_pred,color="#11aa00")
plt.show()