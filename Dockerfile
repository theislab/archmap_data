# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Install system dependencies
RUN set -e; \
    apt-get update -y && apt-get install -y \
        tini \
        lsb-release \
        curl \
        gnupg; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-bullseye main" \
        | tee /etc/apt/sources.list.d/gcsfuse.list; \
    apt-get update; \
    apt-get install -y gcsfuse



# Install required packages
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean
    #pip install git+https://github.com/theislab/pertpy


RUN apt-get update && \
    apt-get install -y git && \
    pip install git+https://github.com/theislab/scarches.git@speed_improvement_merge 


# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

ENV DATABASE_URI=

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY benchmark_atlas/ ./


# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT 9090

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

CMD ["python", "test.py"]