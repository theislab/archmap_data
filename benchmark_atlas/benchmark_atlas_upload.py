from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from google.cloud import storage
import os

app = Flask(__name__)

# Endpoint to generate and save the plot
@app.route('/trigger_cloud_run_job', methods=['POST'])
def generate_plot():
    data = request.get_json()
    data_points = data.get('dataPoints')
    bucket_name = data.get('bucketName')
    file_path = data.get('filePath')

    if not data_points or not bucket_name or not file_path:
        return jsonify({'error': 'Missing parameters'}), 400

    # Generate the plot
    x = data_points['x']
    y = data_points['y']
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Generated Plot')

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
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(local_file)

    return jsonify({'message': f'Plot saved to gs://{bucket_name}/{file_path}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
