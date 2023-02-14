# drb-estuary-salinity-ml

Code repo for Delaware River Basin information theory code and machine learning models that predict estuary salinity

The steps for reproducing the results from the manuscript are detailed below. To retrain a new model use all steps. To analyze results from the manuscript run steps 1-2, then use steps 7-8 for analysis. Model ouptut used for the manuscript is located in `03_model/out/Manuscript_Results` 

#### Steps for reproducibility:

1) clone this repo using `git clone git@github.com:USGS-R/drb-estuary-salinity-ml.git --recurse-submodules`, the `--recurse-submodules` command initiates and updates the `river-dl` submodule housed in `03_model/src/` directory 
2) from within the github cloned directory create the environment using `conda env create -f environment.yaml` or `conda env update --file environment.yaml –prune` if you have already created the environment and just need to update it

#### To (re)train the model:  

3) run `snakemake -s Snakefile_fetch_munge -j` (-j runs the job on the available number of cpus cores, use -j 2 for fewer)
4) you might have to rerun the same command if there is an error that pops up, this is because snakemake doesn't run rules in order and some directories need to be created
5) now open the file `03_model/model_config.yaml` and adjust modeling parameters, and change the run_id to whatever you want to name the test run, for example `Test_Run`
6) run `snakemake -s Snakefile_run_ml_model run_replicates -j`, you should see the training progress in the command window and you should have model results written to `03_model/out/Test_Run/`

#### To analyze model output and (re)produce figures:

7) To calculate functional performance for model output:
    - ensure that COAWST model output are in `03_model/in/COAWST_model_runs/processed`
    - make sure that the parameters in `Snakefile_model_analysis` lines 9-13 describe the desired model run, sources, sinks, and years for analysis. Note: functional performance is calculated on an annual basis
    - run `snakemake -s Snakefile_model_analysis calc_functional_performance_wrapper -j` and the results should be written to `04_analysis/out/"run_id"`
8) To reproduce manuscript figures:
    - make sure that the `run_id` parameter in `Snakefile_model_analysis` and `04_analysis/src/results.R` describe the correct model run
    - run `snakemake -s Snakefile_model_analysis generate_manuscript_figures -j`
    - run `snakemake -s generate_expected_gradient_figure -j`
