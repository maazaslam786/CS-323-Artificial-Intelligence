import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report

df=pd.read_csv('car data.csv')
print("No of Data rows :",df.shape[0])

#Normalization
scaler=MinMaxScaler()
df[['Present_Price']]=scaler.fit_transform(df[['Present_Price']])

#Standardization
standard_scaler=StandardScaler()
df[['Year','Kms_Driven']]=standard_scaler.fit_transform(df[['Year','Kms_Driven']])

#One Hot Encoding
one_hot_encoder=OneHotEncoder(sparse_output=False).set_output(transform="pandas")

df=pd.concat([
    df.drop(columns=['Fuel_Type','Seller_Type']),
    one_hot_encoder.fit_transform(df[['Fuel_Type','Seller_Type']])
],axis=1)

# Ordinal Encoding
ordinal_encoder = OrdinalEncoder(categories=[['Manual', 'Automatic']])
df['Transmission'] = ordinal_encoder.fit_transform(df[['Transmission']])

#Bining
df['Price_Category']=pd.qcut(df['Selling_Price'],q=3,labels=False)

#Scaling
df['Selling_Price']=scaler.fit_transform(df[['Selling_Price']])

df.to_csv('processed_data.csv',index=False)

X=df.drop(['Car_Name','Selling_Price','Price_Category'],axis=1)
y=df['Price_Category']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print("No of Training rows :",X_train.shape[0])
print("No of Testing rows :",X_test.shape[0])

mlp_classifier=MLPClassifier(hidden_layer_sizes=(100,),max_iter=1000)

mlp_classifier.fit(X_train,y_train)

y_pred=mlp_classifier.predict(X_test)

print("Hidden Layer Size:",mlp_classifier.hidden_layer_sizes)
print("Number of Layers:",mlp_classifier.n_layers_)
print("Number of Iterations:",mlp_classifier.n_iter_)
print("Classes :",mlp_classifier.classes_)
print("Accuracy :",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
