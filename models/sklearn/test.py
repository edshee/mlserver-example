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