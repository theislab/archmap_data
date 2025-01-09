#!/bin/bash

# Exit the script on any error
set -e

# Variables
ENV_NAME="archmap_env"  # Name of the Conda environment
REQ_FILE="scarches_api/requirements_local.txt"  # Path to the requirements file

# Function to create a Conda environment
create_conda_env() {
    echo "Creating Conda environment: $ENV_NAME..."
    conda create -y -n $ENV_NAME python=3.10
    echo "Conda environment '$ENV_NAME' created successfully."
}

# Function to install packages from the requirements file
install_requirements() {
    echo "Activating Conda environment: $ENV_NAME..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME

    echo "Installing scarches..."
    pip install git+https://github.com/theislab/scarches.git@speed_improvement_merge

    echo "Installing packages from $REQ_FILE..."
    pip install -r $REQ_FILE
    pip install ipykernel
    python -m ipykernel install --user --name $ENV_NAME
    echo "Packages installed successfully."

    echo "Deactivating environment..."
    conda deactivate
}

# Main script
if [ -f "$REQ_FILE" ]; then
    create_conda_env
    install_requirements
else
    echo "Error: Requirements file '$REQ_FILE' not found!"
    exit 1
fi

echo "Setup complete. Use 'conda activate $ENV_NAME' to activate the environment."
