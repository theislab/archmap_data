
#!/bin/bash

# # Function to log memory and CPU usage
# log_resources() {
#     echo "====== Resource Usage Before Termination ======"
    
#     # Memory usage
#     echo "---- Memory Usage ----"
#     free -m
    
#     # Top 5 CPU consuming processes
#     echo "---- Top 5 CPU Consuming Processes ----"
#     top -bn1 | head -n 8 | tail -n 5 | awk '{ printf "%-8s %-5s %-5s %-10s %-5s %s\n", $1, $9, $10, $12, $6, $NF }'
# }

# # trap to log resources when the script receives a termination signal
# trap log_resources SIGTERM SIGINT SIGHUP




# set -eo pipefail;

# Create mount directory for service
mkdir -p $MNT_DIR;

pwd;
echo "Mounting GCS Fuse for bucket $AWS_BUCKET to $MNT_DIR"
gcsfuse  --implicit-dirs  $AWS_BUCKET $MNT_DIR 
echo "Mounting completed."
echo "Listing the mount directory."
ls -l $MNT_DIR;





# cd into ${MNT_DIR}/${FULL_PATH}/ 
# echo "changing directory to ${MNT_DIR}/${FULL_PATH}/"
# cd ${MNT_DIR}/${FULL_PATH}/

echo "the pwd is now: "
pwd;


#run the api
gunicorn --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 --chdir ./scarches_api/ api:app