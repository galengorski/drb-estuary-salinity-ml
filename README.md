# drb-estuary-salinity-ml

Code repo for Delaware River Basin information theory code and machine learning models that predict estuary salinity

This information is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The information has not received final approval by the U.S. Geological Survey (USGS) and is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the information.

Steps for running the model:

1) clone this repo using `git clone git@github.com:USGS-R/drb-estuary-salinity-ml.git --recurse-submodules`, the `--recurse-submodules` command initiates and updates the `river-dl` submodule housed in `03b_model/src/` directory 
1) add the file `953860.zip` into the `01_fetch/in folder`, the file can be found on S3 in drb_estuary_salinity/01_fetch/in
2) from within the github cloned directory create the environment using `conda env create -f environment.yaml` or `conda env update --file environment.yaml –prune` if you have already created the environment and just need to update it
3) run `snakemake -s Snakefile_fetch_munge -j` (-j runs the job on the available number of cpus cores, use -j 2 for fewer)
4) you might have to rerun the same command if there is an error that pops up, this is because snakemake doesn't run rules in order and some directories need to be created
5) now open the file `03b_model/model_config.yaml` and change n_epochs to a small number say 5, and change the run_id to whatever you want to name the test run, say Test_Run
6) run `snakemake -s Snakefile_b_ml_model_baseline -j`, you should see the training progress in the command window and you should have model results written to `03b_model/out/Test_Run/`
