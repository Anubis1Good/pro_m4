import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("train.csv")

df.drop(
    [
        "id",
        "bdate",
        "last_seen",
        "city",
        "occupation_name",
        "career_start",
        "career_end",
    ],
    axis=1,
    inplace=True,
)

df.drop(["relation", "life_main", "people_main"], axis=1, inplace=True)

df["education_form"].fillna("Full-time", inplace=True)
df["occupation_type"].fillna("university", inplace=True)


# 1.sex
def change_sex(sex):
    if sex == 2:
        return 0
    return 1


df["sex"] = df["sex"].apply(change_sex)
# print(df['sex'].value_counts())

# 2.graduation


def change_graduation(graduation):
    if 1930 < graduation < 2030:
        return graduation
    return 2030


df["graduation"] = df["graduation"].apply(change_graduation)
# print(df['graduation'].value_counts())


# 3. education_form
x = pd.get_dummies(df["education_form"])
df[list(x.columns)] = x
df.drop("education_form", axis=1, inplace=True)

# 4. education_status


def change_es(es):
    if es == "Undergraduate applicant":
        return 1
    if es == "Student (Bachelor's)":
        return 2
    if es == "Alumnus (Bachelor's)":
        return 3
    if es == "Student (Specialist)":
        return 4
    if es == "Alumnus (Specialist)":
        return 5
    if es == "Student (Master’s)":
        return 6
    if es == "Alumnus (Master's)":
        return 7
    if es == "Candidate of Sciences":
        return 8
    if es == "PhD":
        return 9
    return 0


df["education_status"] = df["education_status"].apply(change_es)
# print(df['education_status'].value_counts())


# 5. langs
def change_langs(langs):
    return len(str(langs).split(";"))


df["langs"] = df["langs"].apply(change_langs)
# print(df['langs'].value_counts())


# 6. occupation_type
def change_ot(ot):
    if ot == "work":
        return 1
    return 0


df["occupation_type"] = df["occupation_type"].apply(change_ot)
# print(df['occupation_type'].value_counts())
# df.info()

# ОБУЧЕНИЕ
im = [0, 1, 1.0, 178.0, 2023.0, 5, 2, 1, True, False, False]

X = df.drop("result", axis=1)
y = df["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_test.loc[10000] = im

# print(X_test)
X_test_old = X_test.copy()

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

X_test_old['result'] = y_pred

# print(X_test_old)
headers = list(X_test_old.columns.values)
print(headers)
for i in headers:
    X_test_old.plot(kind='scatter', x='result', y=i)
    plt.show()

# print(y_pred)

# print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
# im = sc.transform(im)

# print(classifier.predict(im))
# print('Confusion matrix:')
# print(confusion_matrix(y_test, y_pred))
