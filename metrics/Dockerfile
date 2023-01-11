FROM rocker/verse:latest

# Install python and R essentials
RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base python3.9 python3-pip python3-setuptools python3-dev

# Create workdir "R" and copy "metrics" folder to the workdir
ENV APP_HOME /app/R
WORKDIR $APP_HOME
COPY . ./

# Install python packages
RUN pip3 install --no-cache-dir -r requirements.txt

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