import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


df = pd.read_csv('train.csv')
# print(df.info())

def has_age(bdate):
    bdate = str(bdate)
    return len(bdate) > 7

def get_year(bdate):
    bdate = str(bdate)
    return int(bdate[-4:])

# df_filter = df[pd.isnull(df['bdate'])==False]
df_filter = df[df['bdate'].apply(has_age)]
df_filter['bdate'] = df_filter['bdate'].apply(get_year)
# df_filter.info()
# print(df_filter.sample(10))
df_filter = df_filter[(df['graduation'] > 1900) & (df['graduation'] < 2025)]
X = df_filter['graduation']
y = df_filter['bdate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


regr = LinearRegression()

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_test)
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
print(accuracy_score(y_test,y_pred))
# df_filter.plot(kind='scatter',x='graduation',y='bdate')
# df_filter.plot(kind='scatter',x='result',y='bdate')
# df_filter.plot(kind='scatter',x='result',y='graduation')

plt.show()

