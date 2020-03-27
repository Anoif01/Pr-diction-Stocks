import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping


sns.set_style('darkgrid')
path = r'C:\Users\lssya\Desktop\Portefeuille\Apprentissage'
os.chdir(path)


#%% Read raw data
def preprocess_Xy(data_path):
    data = pd.read_csv(data_path)[:1250]
    
    # Get stock value augmentation, 1 if increase, else 0
    y = (data['open'] >= data['close']).astype('int')
    # Hide close data to let machine predict
    X = data[['open', 'high', 'low', 
              'volume', 'unadjustedVolume', 'change', 
              'changePercent', 'vwap', 'changeOverTime']]
    # Normalization
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X = X/np.linalg.norm(X, axis=0)
    return X, y


data_path = r'data\AAPL.csv'
X, y = preprocess_Xy(data_path)

cols = list(X)
for col in cols:
    plt.hist(X[col], label=col, alpha=.8)
plt.legend()
plt.show()

X = X.values
y = y.values


#%% Create model
model = Sequential()
n_features = X.shape[1]

X_train, X_test = X[:1000], X[1000:1250]
y_train, y_test = y[:1000], y[1000:1250]

#add model layers
model.add(Dense(10, activation='relu', input_shape=(n_features, )))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=20)
#train model
history = model.fit(X_train, y_train, 
                    validation_split=0.2, epochs=100, 
                    callbacks=[early_stopping_monitor], verbose=0)

y_pred = np.round(model.predict(X_test)).astype('int32').flatten()
acc = np.sum(y_pred == y_test) / len(y_test)
print('-'*16, '\nAPPLE stock data accuracy of prediction = %.4f'%acc)

# plot check
plt.figure(figsize=[12, 8])
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.legend()
plt.show()


#%% KFold Cross-Validation
def net(n_features=9):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(n_features, )))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def CVmodel(X, y):
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    
    acc_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
            
        model = net()
        early_stopping_monitor = EarlyStopping(patience=20)
        model.fit(X_train, y_train, validation_split=0.2, epochs=200, 
                  callbacks=[early_stopping_monitor], verbose=0)
        # model.save_weights("model.h5")
        
        y_pred = np.round(model.predict(X_test)).astype('int32').flatten()
        acc = np.sum(y_pred == y_test) / len(y_test)
        acc_list.append(acc)
    return acc_list, model


acc_list, model = CVmodel(X, y)
print('Train and validation on dataset %s'%data_path.split('\\')[1])
print('10Fold mean accuracy = %.4f'%np.mean(acc_list))


#%% Trying its ability of generalisation
data_path = r'data\IBM.csv'
X, y = preprocess_Xy(data_path)
acc_list, model = CVmodel(X, y)
print('Train and validation on dataset %s'%data_path.split('\\')[1])
print('10Fold mean accuracy = %.4f'%np.mean(acc_list))


#%% Testing with pretrained model
test_acc = []
for csv_name in os.listdir('data'):
    data_path = r'data\%s'%csv_name
    X, y = preprocess_Xy(data_path)
    X, y = X.values, y.values
    y_pred = np.round(model.predict(X)).astype('int32').flatten()
    acc = np.sum(y_pred == y) / len(y)
    print('-'*16)
    print('Test length           : %d'%len(y))
    print('Train/Test on dataset : %s'%data_path.split('\\')[1])
    print('10Fold mean accuracy  = %.4f'%acc)
    test_acc.append(acc)

average = np.mean(test_acc)
print('-'*16)    
print("Average of the scores : %.4f"%average)

