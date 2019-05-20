# 8thEGOMeeting-notebooks
This is a collection of Jupyter Notebooks and other tools to demonstrate the process of data review and evaluation of science data parameters from OOI gliders. This is part of the OOI Synthesis and Education project conducted by Rutgers University and led by the Consortium for Ocean Leadership, tasked with reviewing and identifying datasets for educational use from OOI 1.0 (defined as instrument deployments recovered before Oct 1, 2018). See the [RU Data Review Portal](https://datareview.marine.rutgers.edu/) for progress and results.

**Contributors**: Leila Belabbassi, Lori Garzio, and Sage Lichtenwalner 

Rutgers University Center for Ocean Observing Leadership

## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the 8thEGOMeeting-notebooks repository

`git clone https://github.com/ooi-data-lab/8thEGOMeeting-notebooks.git`

Change your current working directory to the location that you downloaded 8thEGOMeeting-notebooks. 

`cd /Users/lgarzio/Documents/repo/8thEGOMeeting-notebooks/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate EGOMeeting`

Install the toolbox to the conda environment from the root directory of the 8thEGOMeeting-notebooks toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.

To activate the jupyter notebook:

`jupyter notebook`

This will start a Jupyter notebook server in a browser window. You can then run and modify the notebooks.

## Steps for Data Review
- Data request: Use the [1.0_quickstart_request_data](https://github.com/ooi-data-lab/8thEGOMeeting-notebooks/blob/master/1.0_quickstart_request_data.ipynb) or [1.1_data_request_using_file](https://github.com/ooi-data-lab/8thEGOMeeting-notebooks/blob/master/1.1_data_request_using_file.ipynb) notebook to learn how to download NetCDF data files using the OOI API.

- Create a list of OOI 1.0 datasets to review: Use the [2.0_data_review_list](https://github.com/ooi-data-lab/8thEGOMeeting-notebooks/blob/master/2.0_data_review_list.ipynb) notebook to learn how to select the correct OOI 1.0 datasets for each deployment of an instrument for review.

- Test the glider location and file coordinates: Use the [3.0_location_test](https://github.com/ooi-data-lab/8thEGOMeeting-notebooks/blob/master/3.0_location_test.ipynb) notebook to check the glider depth and track, and make sure the files have the correct coordinates.

- Test the data coverage and timestamps: Use the [4.0_time&gap_test](https://github.com/ooi-data-lab/8thEGOMeeting-notebooks/blob/master/4.0_time&gap_test.ipynb) notebook to review the data coverage (compared to the deployment depths) and check if there are any data gaps in the deployments.

- Evaluate science data: Use the [5.0_valid_data_test](https://github.com/ooi-data-lab/8thEGOMeeting-notebooks/blob/master/5.0_valid_data_test.ipynb) notebook to learn how the science data variables are reviewed.

For more a complete list of automated tools used for OOI 1.0 data reviews, check out our [Data Review github repo](https://github.com/ooi-data-lab/data-review-tools).
