import os
import numpy as np
import matplotlib.pyplot as plt
import boto3
from aiohttp import ClientError

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

def main():
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
    main()
