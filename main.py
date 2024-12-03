from __future__ import annotations
import os
from os import environ
from flask import Flask
import boto3

import os
import numpy as np
import matplotlib.pyplot as plt
from aiohttp import ClientError
import requests
import threading
import time

def store_file_in_s3(path, key):
    """
    stores a file in the given path in an s3 bucket
    :param path: path in our filesystem
    :param key: key to where to store the file in s3
    :return: returns ContentLength if successfully uploaded, 0 otherwise
    """
    try:
        bucket = os.getenv('AWS_BUCKET')
        client = boto3.client('s3', endpoint_url=os.getenv('AWS_ENDPOINT'),
                              aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                              aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))
        print("Uploading file:: Bucket. key. path ", bucket, key, path)
        client.upload_file(path, bucket, key)
        response = client.head_object(Bucket=bucket, Key=key)
        print("Response from the upload: ", response)
        return response['ContentLength']
    except ClientError as e:
        print(e)
    return 0

app = Flask(__name__)
# endpoint = os.getenv('ENDPOINT')
# access_key = os.getenv('ACCESS_KEY')
# secret_key = os.getenv(
#     'SECRET_KEY')
# modelname = os.getenv('MODEL_NAME')
# database_uri = os.getenv(
#     'DATABASE_URI')
# bucket = os.getenv('BUCKET')





# s3 = boto3.client('s3', endpoint_url=endpoint,
#                   aws_access_key_id=access_key, aws_secret_access_key=secret_key)
# print("starting download")
# s3.download_file(bucket, modelname, modelname)
# print("Download finished, loading model")
# clf = joblib.load(modelname)
# print("Model loaded, ready to dispose")
# dispose(modelname)

# db = MongoClient(database_uri).get_default_database()





def send_request():
    """
    Sends a request to the /benchmark route of the Flask app.
    This function will be run in a separate thread to ensure that it runs after Flask has started.
    """
    url = 'http://0.0.0.0:9090/benchmark'
    time.sleep(20)  # Wait a bit for the Flask app to start and be ready
    try:
        response = requests.post(url)
        print("Response status code:", response.status_code)
        print("Response body:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error while sending request:", e)

@app.route("/benchmark")
def benchmark():
    upload_id = os.getenv('UPLOAD_ID')
    key_path = os.getenv('PATH')
    file_type = os.getenv('FILETYPE')

    print(f"Upload ID: {upload_id}")
    print(f"Key Path: {key_path}")
    print(f"File Type: {file_type}")

    x = np.linspace(0, 2 * np.pi, 100)  # Generate 100 points between 0 and 2*pi
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(x, y, label='Sine Wave', color='blue')
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()

    
    # Save the plot locally
    local_file = '/tmp/plot.png'
    plt.savefig(local_file)

    # Upload to Google Cloud Storage
    store_file_in_s3(local_file, key_path)




if __name__ == "__main__":

    
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # Start a background thread to send the request after Flask starts
    send_request()
