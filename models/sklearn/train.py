import pandas as pd
from sklearn import svm
import time
import joblib

#Load Training Data
train = pd.read_csv('../../data/fashion-mnist_train.csv', header=0)
y_train = train['label']
X_train = train.drop(['label'], axis=1)
classifier = svm.SVC(kernel="poly", degree=4, gamma=0.1)

#Train Model
start = time.time()
classifier.fit(X_train.values, y_train.values)
end = time.time()
exec_time = end-start
print(f'Execution time: {exec_time} seconds')

#Save Model
joblib.dump(classifier, "Fashion-MNIST.joblib")