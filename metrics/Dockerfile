#FROM rocker/verse:latest
FROM rocker/verse:4.1.3

# Install python and R essentials
#RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base python3.9 python3-pip python3-setuptools python3-dev
RUN apt-get update && apt-get install -y python3.9 python3-pip

#Needed to get "annoy" running
RUN apt-get -y install python3-dev
RUN apt-get -y install libpython3.9-dev
RUN apt-get -y install gcc

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.9 /usr/bin/python

RUN python3.9 -m pip install --upgrade pip

# Create workdir "R" and copy "metrics" folder to the workdir
ENV APP_HOME /app/R
WORKDIR $APP_HOME
COPY . ./

# Install python packages
#RUN pip3 install --no-cache-dir -r requirements.txt

RUN python3.9 -m pip install -r requirements.txt --no-cache

# Install R packages (with rocker)
RUN install2.r --error \
    tibble \
    RColorBrewer \
    dynutils \
    stringr \
    Hmisc \
    plyr \
    dplyr \
    scales \
    ggimage \
    cowplot

#CMD Rscript test.R