# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Install system dependencies
RUN set -e; \
    apt-get update -y && apt-get install -y \
    tini \
    lsb-release curl gnupg2; \
    GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s); \
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -; \
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