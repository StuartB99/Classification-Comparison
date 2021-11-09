# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 18:25:45 2021

Run and test SVM models 

@author: Stuart
"""

import numpy as np
import pandas as pd
import time
import joblib
import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

# PREPROCESSING
start = time.time()

categories = ["with_sunglass", "no_sunglass"]
flat_data_list=[] #input array
target_list=[] #output array

data_dir="D:\\Sunglass_Dataset_224x224\\all" 

for category in categories:
    print(f'loading... Category : {category}')
    path=os.path.join(data_dir,category)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_list.append(img_resized.flatten())
        target_list.append(categories.index(category))
    print(f'Loaded category:{category} successfully')
    
flat_data=np.array(flat_data_list)
target=np.array(target_list)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data

end = time.time()
print("Duration of preprocessing: " + str(end - start))


# Model Construction
param_grid={'C':[0.1,1,10],'gamma':[0.0001,0.001,0.1],'kernel':['linear','poly','rbf']}

svc=SVC(probability=True)
model=GridSearchCV(svc,param_grid, verbose=4)

# Model Training
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=2,stratify=y)
print('Splitted Successfully')

start = time.time()
model.fit(x_train,y_train)
print('Training complete')

print("CV Results: ")
print(model.cv_results_)
print("Best Params: ")
print(model.best_params_)

end = time.time()
print("Duration of training: " + str(end - start))

# Loading the model
joblib.dump(model, 'svm_saved_model.joblib') 

# Model Testing
start = time.time()
y_pred=model.predict(x_test)

# Getting the results and metrics
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
print(f"The model's F1 score is {f1_score(y_pred,y_test)*100} ")
print(f"The model's precision is {precision_score(y_pred,y_test)} ")
print(f"The model's recall is {recall_score(y_pred,y_test)} ")
print(f"The model's roc auc score is {roc_auc_score(y_pred,y_test)} ")
print("End of Testing")

end = time.time()
print("Duration of testing: " + str(end - start))


