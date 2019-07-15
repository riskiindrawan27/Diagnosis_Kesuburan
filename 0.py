import numpy as np 
import pandas as pd 

dfFer = pd.read_csv('fertility.csv')

# print(dfFer)
# print(dfFer.isnull().sum())

# dfFer = dfFer.drop('Season', axis='columns')
# print(dfFer.columns.values)
# print(dfFer)
# x = dfFer.drop('Diagnosis', axis='columns')
# y = dfFer.drop(['Age', 'Childish diseases', 'Accident or serious trauma', 'Surgical intervention', 'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit', 'Number of hours spent sitting per day'] , axis ='columns')
# print(x)
# print(y)

# Label Encoder
from sklearn.preprocessing import LabelEncoder
label1 = LabelEncoder()
label2 = LabelEncoder()
label3 = LabelEncoder()
label4 = LabelEncoder()
label5 = LabelEncoder()
label6 = LabelEncoder()
label7 = LabelEncoder()
label8 = LabelEncoder()
dfFer.pop('Season')

dfFer['Childish diseases'] = label1.fit_transform(dfFer['Childish diseases'])
dfFer['Accident or serious trauma'] = label2.fit_transform(dfFer['Accident or serious trauma'])
dfFer['Surgical intervention'] = label3.fit_transform(dfFer['Surgical intervention'])
dfFer['High fevers in the last year'] = label4.fit_transform(dfFer['High fevers in the last year'])
dfFer['Frequency of alcohol consumption'] = label5.fit_transform(dfFer['Frequency of alcohol consumption'])
dfFer['Smoking habit'] = label6.fit_transform(dfFer['Smoking habit'])
# dfFer.drop(columns=['Season','Childish diseases','Accident or serious trauma','Surgical intervention', 'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit'])
dfTarget = dfFer.pop('Diagnosis')
dfTarget = label7.fit_transform(dfTarget)
# print(dfTarget) 
# print(label1.classes_)
# print(label2.classes_)
# print(label3.classes_)
# print(label4.classes_)
# print(label5.classes_)
# print(label6.classes_)
# print(label7.classes_)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans =  ColumnTransformer(
    [('OHE', OneHotEncoder(categories='auto'),[4, 5, 6])],
    remainder='passthrough'
)
df= coltrans.fit_transform(dfFer)


from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts =  train_test_split(
    df,
    dfTarget,
    test_size = .1
)
# print(xtr)
# print(ytr)

# ML Decision Tree
from sklearn.tree import DecisionTreeClassifier
modelTree=DecisionTreeClassifier()
modelTree.fit(xtr,ytr)

# ML Extra Tree
from sklearn.ensemble import ExtraTreesClassifier
modelExtra = ExtraTreesClassifier()
modelExtra.fit(xtr, ytr)

# ML Logistic Regression
from sklearn.linear_model import LogisticRegression
modelLog = LogisticRegression(solver='liblinear', multi_class='auto')
modelLog.fit(xtr,ytr)

# ML KMeans
from sklearn.cluster import KMeans
modelKMeans = KMeans(n_clusters=len(label7.classes_))
modelKMeans.fit(xtr, ytr)

print(dfTarget)
print(modelTree.predict(df))
print(modelKMeans.predict(df))
print(modelLog.predict(df))

def target(x):
    if x[0]==0:
        return label7.classes_[x[0]]
    elif x[0]==1:
        return label7.classes_[x[0]]

print('Arin, prediksi kesuburan:',target(modelTree.predict([[0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]])),'(Decision Tree)')
print('Arin, prediksi kesuburan:',target(modelExtra.predict([[0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]])),'(Extra Tree)')
print('Arin, prediksi kesuburan:',target(modelLog.predict([[0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]])),'(Logistic Regression)')
print('---------------------------------------------------------------------')
print('Bebi, prediksi kesuburan:',target(modelTree.predict([[0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]])),'(Decision Tree)')
print('Bebi, prediksi kesuburan:',target(modelExtra.predict([[0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]])),'(Extra Tree)')
print('Bebi, prediksi kesuburan:',target(modelLog.predict([[0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]])),'(Logistic Regression)')
print('---------------------------------------------------------------------')
print('Caca, prediksi kesuburan:',target(modelTree.predict([[1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]])),'(Decision Tree)')
print('Caca, prediksi kesuburan:',target(modelExtra.predict([[1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]])),'(Extra Tree)')
print('Caca, prediksi kesuburan:',target(modelLog.predict([[1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]])),'(Logistic Regression)')
print('---------------------------------------------------------------------')
print('Dini, prediksi kesuburan:',target(modelTree.predict([[0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]])),'(Decision Tree)')
print('Dini, prediksi kesuburan:',target(modelExtra.predict([[0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]])),'(Extra Tree)')
print('Dini, prediksi kesuburan:',target(modelLog.predict([[0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]])),'(Logistic Regression)')
print('---------------------------------------------------------------------')
print('Enno, prediksi kesuburan:',target(modelTree.predict([[0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]])),'(Decision Tree)')
print('Enno, prediksi kesuburan:',target(modelExtra.predict([[0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]])),'(Extra Tree)')
print('Enno, prediksi kesuburan:',target(modelLog.predict([[0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]])),'(Logistic Regression)')
