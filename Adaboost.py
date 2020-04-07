
#Importing Libraries

import numpy as np
import pandas as pd


#Data preprocessing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=None)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Training

from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier(base_estimator=None,n_estimators=400, learning_rate=1.0, random_state=None)
               


clf.fit(X, Y)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy: {np.mean(accuracy_score(y_test,y_pred))} ± {np.std(accuracy_score(y_test,y_pred))}')
