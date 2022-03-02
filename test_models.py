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