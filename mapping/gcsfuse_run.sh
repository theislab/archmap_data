
#!/bin/bash
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