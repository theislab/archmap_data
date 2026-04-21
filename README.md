# archmap_data
This repo contains the machine learning code for the models used by archmap. 

## Running your mapping locally

To run the mapping process locally, please follow these steps:

### 1. Clone the repository

```bash
git clone -b mappingjob --single-branch https://github.com/theislab/archmap_data.git
cd mapping
```

### 2. Create the environment

```bash
bash create_env.sh
```

### 3. Download the HLCA reference atlas

Please follow this link:
https://drive.google.com/drive/folders/1-LUEad1iy5DNDZmKjQoPTz8ehMYDj91l?usp=drive_link

to access all the files needed to run the tutorial.

Download the folder "hlca_tutorial" and ensure that this folder is saved locally as "hlca_tutorial" in the path:

archmap_data/mapping/

If the link does not work, please copy and paste the URL into your browser.

The data_only_count.h5ad file is only needed if you would like to include the reference count data in your final mapped output for downstream analysis. The gene conversion files are only needed if you are using your own mapping and not the provided query data.

### 4. Run the notebook

You can now run the tutorial located at:

archmap_data/mapping/tutorial.ipynb

Please make sure to read the instructions in tutorial.ipynb carefully.

If you encounter any issues, please do not hesitate to contact us via:
https://www.archmap.bio/#/







