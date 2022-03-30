# Serving Python Machine Learning Models With Ease

Ever trained a new model and just wanted to use it through an API straight away? Sometimes you don't want to bother writing Flask code or containerizing your model and running it in Docker. If that sounds like you, you definitely want to check out [MLServer](https://github.com/seldonio/mlserver). It's a python based inference server that [recently went GA](https://www.seldon.io/introducing-mlserver) and what's really neat about it is that it's a highly-performant server designed for production environments too. That means that, by serving models locally, you are running in the exact same environment as they will be in when they get to production. 

This blog walks you through how to use MLServer by using a couple of image models as examples...

## Dataset

The dataset we're going to work with is the [Fashion MNIST dataset](https://www.kaggle.com/zalando-research/fashionmnist). It contains 70,000 images of clothing in greyscale 28x28 pixels across 10 different classes (top, dress, coat, trouser etc...). 

*If you want to reproduce the code from this blog, make sure you download the files and extract them in to a folder named `data`. They have been omitted from the github repo because they are quite large.*

## Training the Scikit-learn Model

First up, we're going to train a support vector machine (SVM) model using the [scikit-learn](link) framework. We'll then save the model to a file named `Fashion-MNIST.joblib`.

```python
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
```

Note: The SVM algorithm is not particularly well suited to large datasets because of it's exponential nature. The model in this example will, depending on your hardware, take a couple of minutes to train. 

## Serving the Scikit-learn Model

Ok, so we've now got a saved model file `Fashion-MNIST.joblib`. Let's take a look at how we can serve that using MLServer...

First up, we need to install MLServer.

`pip install mlserver`

/* Add install for scikit-learn server too */

Once we've done that, all we need to do is add two configuration files:
- `settings.json` - This contains the configuration for the server itself.
- `model-settings.json` - As the name suggests, this file contains configuration for the model itself. 

For our `settings.json` file it's enough to just define single parameter:

```json
{
    "debug": "true"
}
```

The `model-settings.json` file requires a few more bits of info as it needs to know about the model we're trying to serve:

```json
{
    "name": "fashion-sklearn",
    "implementation": "mlserver_sklearn.SKLearnModel",
    "parameters": {
        "uri": "./Fashion_MNIST.joblib",
        "version": "v1"
    }
}
```

The `name` parameter should be self-explanatory. It gives MLServer a unique identifier which is particularly useful when serving multiple models (we'll come to that in a bit). The `implementation` defines which pre-built server, if any, to use. It is heavily coupled to the machine learning framework used to train your model. In our case we trained the model using scikit-learn so we're going to use the scikit-learn implementation for MLServer. For model `parameters` we just need to provide the location of our model file as well as a version number.

That's it, two small config files and we're ready to serve our model using the command:

`mlserver start .` 

Boom, we've now got our model running on a production-ready server locally. It's now ready to accept requests over HTTP and gRPC (default ports `8080` and `8081` respectively).

## Testing the Model

Now that our model is up and running. Let's send some requests to see it in action.

To make predictions on our model, we need to send requests to the following URL:

`http://localhost:8080/v2/models/<MODEL_NAME>/versions/<VERSION>/infer`

That means to access our scikit-learn model we need to replace the `MODEL_NAME` with `fashion-sklearn` and `VERSION` with `v1`. 

The code below shows how to import the test data, make a request to the model server and then compare the result with the actual label:

```python
import pandas as pd
import requests

#Import test data, grab the first row and corresponding label
test = pd.read_csv('../../data/fashion-mnist_test.csv', header=0)
y_test = test['label'][0:1]
X_test = test.drop(['label'],axis=1)[0:1]

#Prediction request parameters
inference_request = {
    "inputs": [
        {
          "name": "predict",
          "shape": X_test.shape,
          "datatype": "FP64",
          "data": X_test.values.tolist()
        }
    ]
}
endpoint = "http://localhost:8080/v2/models/fashion-sklearn/versions/v1/infer"

#Make request and print response
response = requests.post(endpoint, json=inference_request)
print(response.text)
print(y_test.values)
```

/* something about the inference request object and v2 serving protocol */

When running the our test request code above we get the following response from MLServer:

```json
{
  "model_name": "fashion-sklearn",
  "model_version": "v1",
  "id": "31c3fa70-2e56-49b1-bcec-294452dbe73c",
  "parameters": null,
  "outputs": [
    {
      "name": "predict",
      "shape": [
        1
      ],
      "datatype": "INT64",
      "parameters": null,
      "data": [
        0
      ]
    }
  ]
}
```

You'll notice that MLServer has generated a request id and automatically added metadata about the model and version that was used to serve our request. Capturing this kind of metadata is super important once our model gets to production; it allows us to log every request for audit and troubleshooting purposes. 

You might also notice that MLServer has returned an array for `outputs`. In our request we only sent one row of data but MLServer will also handle batch requests and return them together. You can even use a technique called [adaptive batching](link) to optimise the way multiple requests are handled in production environments. 

In our example above, the model's prediction can be found in `outputs[0].data` which shows that the model has labeled this sample with the category `0` (The value 0 corresponds to the category `INSERT CATEGORY`). The true label for that sample was a `0` so the model got this prediction correct!

## Training the XGBoost Model

Now that we've seen how to create and serve a single model using MLServer, let's take a look at how we'd handle multiple models trained in different frameworks. 

We'll be using the same Fashion MNIST dataset but, this time, we'll train an [XGBoost](link) instead.

```python
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
```

The code above, used to train the XGBoost model, is similar to the code we used earlier to train the scikit-learn model but this time our model has been saved in an XGBoost-compatible format as `Fashion_MNIST.json`.

## Serving Multiple Models

One of the cool things about MLServer is that it supports [multi-model serving](link). This means that you don't have to create or run a new server for each ML model you want to deploy. Using the models we built above, we'll use this feature to serve them both at once.

When MLServer starts up, it will search the directory (and any subdirectories) for `model-settings.json` files. If you've got multiple `model-settings.json` files then it'll automatically serve them all. 

*Note: you still only need a single `settings.json` file in the root directory (the one that contains server config)*

Here's a breakdown of my directory structure for reference:

```bash
.
├── data
│   ├── fashion-mnist_test.csv
│   └── fashion-mnist_train.csv
├── models
│   ├── sklearn
│   │   ├── Fashion_MNIST.joblib
│   │   ├── model-settings.json
│   │   ├── test.py
│   │   └── train.py
│   └── xgboost
│       ├── Fashion_MNIST.json
│       ├── model-settings.json
│       ├── test.py
│       └── train.py
├── README.md
├── settings.json
└── test_models.py
```

Notice that there are two `model-settings.json` files - one for the scikit-learn model and one for the XGBoost model. 

We can now just run `mlserver start .` and it will start handling requests for both models.

```bash
[mlserver] INFO - Loaded model 'fashion-sklearn' succesfully.
[mlserver] INFO - Loaded model 'fashion-xgboost' succesfully.
```

## Testing Accuracy of Multiple Models

With both models now up and running on MLServer, we can use the samples from our test set to validate how accurate each of our models is. 

The following code sends a batch request (containing the test set) to each of the models and then compares the predictions received to the true labels. Doing this across the whole test set gives us a reasonably good measure for each model's accuracy, which will be printed. 

```python
import pandas as pd
import requests
import json

#Import the test data and split the data from the labels
test = pd.read_csv('./data/fashion-mnist_test.csv', header=0)
y_test = test['label']
X_test = test.drop(['label'],axis=1)

#Build the inference request
inference_request = {
    "inputs": [
        {
          "name": "predict",
          "shape": X_test.shape,
          "datatype": "FP64",
          "data": X_test.values.tolist()
        }
    ]
}

#Send the prediction request to the relevant model, compare responses to training labels and calculate accuracy
def infer(model_name, version):
    endpoint = f"http://localhost:8080/v2/models/{model_name}/versions/{version}/infer"
    response = requests.post(endpoint, json=inference_request)

    #calculate accuracy
    correct = 0
    for i, prediction in enumerate(json.loads(response.text)['outputs'][0]['data']):
        if y_test[i] == prediction:
            correct += 1
    accuracy = correct / len(y_test)
    print(f'Model Accuracy for {model_name}: {accuracy}')

infer("fashion-xgboost", "v1")
infer("fashion-sklearn", "v1")
```

The results show that the XGBoost model slightly outperforms the SVM scikit-learn one:

```
Model Accuracy for fashion-xgboost: 0.8953
Model Accuracy for fashion-sklearn: 0.864
```

## Summary

Hopefully by now you've gained an understanding of how easy it is to serve models using [MLServer](link). For further info it's worth reading the [docs](link) and taking a look at the [examples for different frameworks](link). 

For [MLFlow](link) users you can now serve [models directly in MLFlow using MLServer](link) and if you're a [Kubernetes](link) user you should definitely check out [Seldon Core](link) - an open source tool that deploys models to Kubernetes (it uses MLServer under the covers). 

All of the code from this example can be found [here](link).