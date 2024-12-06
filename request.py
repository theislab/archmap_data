from flask import Flask, request, jsonify
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest
import requests
import os

app = Flask(__name__)

# Endpoint to trigger the Cloud Run job
@app.route('/trigger_cloud_run_job', methods=['POST'])
def trigger_cloud_run_job():
    try:
        # Extract variables from the request body
        data = request.get_json()
        upload_id = data.get('uploadId')
        key_path = data.get('keyPath')
        upload_file_type = data.get('uploadFileType')

        if not upload_id or not key_path or not upload_file_type:
            return jsonify({"error": "Missing uploadId, keyPath, or uploadFileType in request body."}), 400

        # Get the Cloud Run Job URL from environment variables
        url = os.getenv('CLOUD_RUN_JOB')

        # Authenticate with Google Cloud using service account credentials
        credentials = service_account.Credentials.from_service_account_file(
            'path/to/service_account.json',  # Replace with the path to your service account JSON
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Refresh the credentials to get an access token
        credentials.refresh(GoogleRequest())
        access_token = credentials.token

        # Trigger the job asynchronously with environment variables passed in the request
        payload = {
            "environmentVariables": {
                "UPLOAD_ID": upload_id,
                "PATH": key_path,
                "FILETYPE": upload_file_type  # Pass additional variables as needed
            }
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        # Handle response and send back to the client
        if response.status_code == 200:
            return f"Job triggered successfully: {response.json().get('name')}", 200
        else:
            return f"Failed to trigger job: {response.text}", response.status_code

    except Exception as e:
        print(f"Error triggering job: {str(e)}")
        return "Failed to trigger job", 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
