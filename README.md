# 8thEGOMeeting-notebooks
This is a collection of Jupyter Notebooks and other tools to demonstrate the process of data review and evaluation of science data parameters from OOI gliders. This is part of the OOI Synthesis and Education project conducted by Rutgers University and led by the Consortium for Ocean Leadership. See the [RU Data Review Portal](https://datareview.marine.rutgers.edu/) for progress and results.

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

Steps:
1 - Data request: 
Use data_request_interactive or data_request_using_file python notebooks to learn how to get datafiles from the OOI system

2 - 