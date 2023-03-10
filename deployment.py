import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
##########################################
# Load the boston house pricing dataset###
##########################################


from sklearn.datasets import load_boston
boston= load_boston()
boston.keys()

print(boston.DESCR)
print(boston.data)

#####################
# Preparing Dataset##
#####################


dataset= pd.DataFrame(boston.data,columns=boston.feature_names)
dataset['Price']= boston.target
dataset.head()
dataset.info()

## Describe my dataset

dataset.describe()

## The missing value

dataset.isnull().sum()

### Exploratory Data Analysis

# Correlation

dataset.corr() # Chercher les multicolinéarité entre les variables explicatives et la corrélation entre la variable dépendante et les variables indépendantes.
sns.pairplot(dataset)

plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel('Crime rate')
plt.ylabel('Price')

# Regression plot
sns.regplot(x='RM',y='Price',data=dataset)

# Independant and dependant feature

x= dataset.iloc[:,:-1]# Toutes les lignes et toutes les colones sauf la dèrnière
x.head()
y= dataset.iloc[:,-1]# Toutes les lignes et la dernière colone
y.head()

# Train test Split

X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=42)

X_train
X_test
# Standardize the dataset

scaler= StandardScaler()

X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
X_test
pickle.dump(scaler,open('scaling.pkl','wb'))
##################
## Model Training#
##################

## Trainning

reg= LinearRegression()
reg.fit(X_train, y_train)

# Print the coefficient

print(reg.coef_)
print(reg.intercept_)

# on which parameter the model has been trained

reg.get_params()

## Prediction

reg_pred= reg.predict(X_test)
reg_pred

# Scatter plot of the prediction

plt.scatter(y_test,reg_pred)
plt.scatter(reg_pred,y_test)

# Residual

residual= y_test-reg_pred
sns.displot(residual,kind='kde')
plt.scatter(reg_pred,residual)

print(mean_absolute_error(y_test, reg_pred))
print(mean_squared_error(y_test, reg_pred))
print (np.sqrt(mean_squared_error(y_test, reg_pred)))


#######################
## Performance Metrics#
#######################

score= r2_score(y_test, reg_pred)# R carré
print(score)

# R carré ajusté

score_adj= 1 - ((1- score)*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1))

######################
# New data Prediction#
######################

new=boston.data[0].reshape(1,-1)# 

# Transform new data

st=scaler.transform(new)

# Predict
new_pred= reg.predict(st)

##########################################
## Pickling the model file for Deployment#
##########################################

# permet d'enregistrer le modèle

pickle.dump(reg,open('regmodel.pkl','wb')) 

# load the model

pickled_model= pickle.load(open('regmodel.pkl','rb'))

pickled_model.predict(st)













