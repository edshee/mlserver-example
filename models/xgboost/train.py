import pandas as pd
import xgboost as xgb
import time

#Load Training Data
train = pd.read_csv('../../data/fashion-mnist_train.csv', header=0)
y_train = train['label']
X_train = train.drop(['label'], axis=1)
dtrain = xgb.DMatrix(X_train.values, label=y_train.values)

#Train Model
params = {
    'max_depth': 5,
    'eta': 0.3,
    'verbosity': 1,
    'objective': 'multi:softmax',
    'num_class' : 10
}
num_round = 50

start = time.time()
bstmodel = xgb.train(params, dtrain, num_round, evals=[(dtrain, 'label')], verbose_eval=10)
end = time.time()
exec_time = end-start
print(f'Execution time: {exec_time} seconds')

#Save Model
bstmodel.save_model('Fashion_MNIST.json')