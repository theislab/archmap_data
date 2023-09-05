FROM python:3.9-slim


# Install system dependencies
RUN set -e; \
    apt-get update -y && apt-get install -y \
    tini \
    lsb-release curl gnupg2; \
    gcsFuseRepo=gcsfuse-`lsb_release -c -s`; \
    echo "deb http://packages.cloud.google.com/apt $gcsFuseRepo main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && apt-get clean 

# Install required packages
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean

ENV APP_HOME /app
WORKDIR $APP_HOME


ENV MNT_DIR /mnt/gcs

ADD ./scarches_api/requirements.txt .

RUN pip install -r requirements.txt 

COPY . .

RUN chmod +x ${APP_HOME}/gcsfuse_run.sh


ENV WORKERS 1
ENV THREADS 1
ENV PORT 8080
ENV PYTHONPATH $APP_HOME

# Use tini to manage zombie processes and signal forwarding
# https://github.com/krallin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["./gcsfuse_run.sh"]