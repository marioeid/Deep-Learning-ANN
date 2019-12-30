                              # part 1 data pre processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # range 3:13 exculudes 13 if you wrote [3,2] he will take those indecies
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2= LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
  
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



                        #part 2 building the ANN
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# intializing the ANN
classifier=Sequential()

#Adding the input layer and the first hidden layer kernel intializer intialzies the weights intialy
# relu is the reqtifier function

# need to define the input in the first layer only
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

# add the next layer 
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

# adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

# compiling the NN the algorithm u want to use is the optimizer (gradient descent)
# here we will use one of the stochastic gradient algorithms and it's name is Adam
# loss is the lost function in the gradient algo
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting classifier to the Training set
# batch size size when to update weights 
# epochs number of iterations on gradient descenet
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
                          
                        # part 3 making predictions on the test set
# Predicting the Test set results that the customer would leave the bank
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix  to validate our model on the test set 
# the confusion matrix uses true or false so convert he prob before using the matrix
y_pred=(y_pred>0.5)

# make a single prediction not from the dataset (new observation)

"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

new_prediction =classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))) # two D array is horizontal one D is vertical 
new_prediction=(new_prediction>0.5)
# don't forget to scale the new predection


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
                                  

                                    # part 4 using k-cross validation to 
                                   # get better accuracy
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# to use two different libraries
from sklearn.model_selection import cross_val_score


def build_classifier():
          # intializing the ANN
        classifier=Sequential()

                    # copy the whole structure of the neural network structure only
             # need to define the input in the first layer only
        classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
        classifier.add(Dropout(rate=0.1))
        
    # first is the fraction of the neurens you want to drop at every iteration is there's ten neuerens and p=0.1 then we will drop 1 neuron
        # if the 0.1 didn't solve the overfitting problem then try a higher value 
        # if you try with a higer value you might end up with under fitting like 0.5 you will drop too much neurons 
        # the best is to try dropout to all your layers 
       
        # add the next layer 
        classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
        classifier.add(Dropout(rate=0.1)) 
      
        # adding the output layer
        classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
        classifier.add(Dropout(rate=0.1))
        # compiling the NN the algorithm u want to use is the optimizer (gradient descent)
        # here we will use one of the stochastic gradient algorithms and it's name is Adam
        # loss is the lost function in the gradient algo
        classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        return classifier


# first argument is build NN funciton so we but the structure into function
# the only differnece between classifiers here and up is the training part only here we use 
# k cross validation to train our data set 
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
# now we are ready to use the k cross validation funciton from scikit library
# the k cross validation function will return accuracies on our set in our case here 10
# estmiator is the obect to use to fit the data
# x is our training set
# y is the target variable of the same training set
# cv is the number of folds k = what
# n jobs click ctrl+i to see it very important
# n_jobs=-1 won't work if you don't have your gbu transflow n_jobs make your computation faster (parallel computation)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1) 

mean=accuracies.mean()
variance=accuracies.std()

# standard deviation (is the std())

                                           #part5 tuNNing our ANN 
# tunning means het best hypered parameters to choose to get the best accurcy like
# epochs (gradient itrerations), batch size (how many k using k-croos),optimizer (which gradient descent algo),units (the number of neurons in the layer)
# so we will use technique (grid search) it will try a combination of those values and returns the best combination to use
                                           
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# to use two different libraries
from sklearn.model_selection import GridSearchCV



# optimizer a parameter added for tuning different choises
def build_classifier(optimizer):
          # intializing the ANN
        classifier=Sequential()

                    # copy the whole structure of the neural network structure only
             # need to define the input in the first layer only
        classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
        # first is the fraction of the neurens you want to drop at every iteration is there's ten neuerens and p=0.1 then we will drop 1 neuron
        # if the 0.1 didn't solve the overfitting problem then try a higher value 
        # if you try with a higer value you might end up with under fitting like 0.5 you will drop too much neurons 
        # the best is to try dropout to all your layers 
       
       
        # add the next layer 
        classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
       
        # adding the output layer
        classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
        
        # compiling the NN the algorithm u want to use is the optimizer (gradient descent)
        # here we will use one of the stochastic gradient algorithms and it's name is Adam
        # loss is the lost function in the gradient algo
        classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        return classifier

# we will remove batch size and gradient iteration because those are the numbers we want 
# to get the best combination with
classifier=KerasClassifier(build_fn=build_classifier)

# those numbers to try on batch size based on experience 
# and also common practise to try some power of 2 like 32
# names has to be the same as you used to train your classifier
# we tuned two hyper parameters 
# what if we want to tune teh optimizer it's already given a value (how to input a different value)
# create a new parameter in the build classifier function and this argument will give you choice for the optimizer
parameters={'batch_size':[25,32]
           ,'epochs':[100,500]
           ,'optimizer':['adam','rmsprop']
          }

gridsearch=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
gridsearch=gridsearch.fit(X_train,y_train)
best_param=gridsearch.best_params_
best_accuracy=gridsearch.best_score_
 