################################
# DIABETES PREDICTION with CART
################################

# pip install pydotplus
# pip install skompiler
# pip install astor

import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz, export_text
from skompiler import skompile

from helpers.eda import *
pd.set_option('display.max_columns', None)

from helpers.data_prep import *



################################
# DATA PREPROCESSING
################################
# Ağaç yöntemlerinde aykırı değer ve eksik değerin önemi kalmamaktadır.
# Ya da çok çok çok azdır.

# Sınıflandırma problemi olduğu için aykırı değerler ya da eksik değerlerin bir hiçbir önemi yok.
# Eğer regresyon problemi olsaydı
# bağımlı değişkendeki aykırılıklar ufak bir miktar önemli olabilir ama göz ardı edilebilirdi.



################################
# CART
################################

# 1. CART ile Modelleme
# 2. Holdout Yöntemi ile Model Doğrulama
# 3. Hiperparametre Optimizasyonu
# 4. Final Modelin Yeniden Tüm Veriye Fit Edilmesi


df = pd.read_csv("diabetes.csv")
def median_target(X):
    temp = df[df[X].notnull()]
    temp = temp[[X, 'Outcome']].groupby(['Outcome'])[[X]].median().reset_index()
    return temp
median_target("Pregnancies")
#Eksik gözlemler için verilecek değerlere, hasta olmayanların medyan değeri ve hasta olanların medyan değerleri verilmiştir.
columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]

num_cols = [col for col in df.columns if len(df[col].unique()) > 20 and df[col].dtypes != 'O']

#Outlier
check_outlier(df,num_cols)
from helpers.data_prep import replace_with_thresholds
for col in num_cols:
    replace_with_thresholds(df, col)

### Feature Engineering
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]
df["NewBMI"].value_counts()
### İnsülin
def set_insulin(df,col = "Insulin"):
    if df[col] >= 16 and df[col] <= 166:
        return "Normal"
    else:
        return "Anormal"

df.columns
df["New_Insulin"] = df.apply(set_insulin, axis=1)

#df = df.assign(New_Insulin2=df.apply(set_insulin, axis=1))
df[["New_Insulin","New_Insulin2"]]
# Label-One Hot Encoding

binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

#df.drop("New_Insulin2",axis=1,inplace=True)
for col in binary_cols:
    df = label_encoder(df, col)
df.head()
df = rare_encoder(df, 0.30)
df.shape
df["NewBMI"].value_counts()
df["NewBMI"] = df["NewBMI"].astype("O")
df.info()
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols,drop_first=True)

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)

################################
# HOLDOUT YÖNTEMİ İLE MODEL DOĞRULAMA
################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

################################
# DEĞİŞKEN ÖNEM DÜZEYLERİNİ İNCELEMEK
################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_model, X_train)


################################
# HİPERPARAMETRE OPTİMİZASYONU
################################

cart_model


cart_model = DecisionTreeClassifier(random_state=17)

# arama yapılacak hiperparametre setleri
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [2, 3, 4]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)

# train hatası
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


