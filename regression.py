import pandas as pd 
import quandl,math
import numpy as np
from sklearn import preprocessing , cross_validation, svm
from sklearn.linear_model import  LinearRegression 
#lib allows to use arrays etc.
#preprocessing is used for scaling and normalising etc
#cross_validation is used for shuffling data for training and testing
#svm - support vector mechines

df=quandl.get('WIKI/GOOGL')


df=df[['Adj. Open','Adj. Volume','Adj. Close','Adj. Low','Adj. High']]

df['HL_Pct']=(df['Adj. High']-df['Adj. Low'])/df['Adj. High']*100.0
df['Pct_change']=(df['Adj. Open']-df['Adj. Close'])/df['Adj. Open']*100.0

df=df[['Adj. Close','HL_Pct','Pct_change','Adj. Volume']]

forecast_col='Adj. Close'

df.fillna(-9999,inplace=True)
#this fills the columns or boxes which have no value i.e. are null, 
#will be filled with the given value so that while regression over the data set 
#no erraneous results are produced due to missing data


forecast_out=int(math.ceil(0.01*len(df)))
#this is the time after which the prediction is made i.e. the number of days 
#in future the prediction is made for 
print(forecast_out)
#suppose it is 30 then the prediction will be for 30 days in future 

df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())

x=np.array(df.drop(['label'],1))  
#features are everything except the lable column
y=np.array(df['label'])
#its only the label column
x=preprocessing.scale(x)
#its scaling the data, all the data 
y=np.array(df['label'])

print(len(x),len(y))

X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(x,y,test_size=0.2)
#what is does is basically take all the features and labels and shuffle them up slect test_size % for test cases and 
#the remaining 100-test_case % as the training data

clf=LinearRegression() 
#clf=LinearRegression(n_jobs=10)
#n_jobs= something will pass the number of threads we want to use while learning,
#-1 being the argument suggesting to use tha maximum numbers of threads that can be created in the computer 
#this makes the learning process a lot faster 

# clf=svm.SVR(kernel='poly')
#this will change the classifier to a SVM with a polynomial kernel
#the default kernel is linear

clf.fit(X_train,Y_train)
#this step fits the linear plot to the training data provided to it in the arguement 

accuracy=clf.score(X_test,Y_test)
#this is used to check the accuracy of the prediction
#do not use the same data for training and testing, that is because 
#the results will be erroneously good 

print(accuracy)