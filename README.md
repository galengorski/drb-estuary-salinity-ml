# drb-estuary-salinity-ml

This model archive contains code used for deep learning and hydrodynamic modeling aimed at understanding and simulating salinity dynamics in the Delaware Bay. The steps for reproducing the results from the manuscript are detailed below.

You will need to have [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), [R](https://cran.r-project.org/), and [mamba](https://mamba.readthedocs.io/en/latest/installation.html) installed on your computer to run this pipeline.

To retrain a new model run all steps 1-6 below.

If you want to reproduce the analysis of the results from the manuscript, run steps 1-4 to prepare the environment and input datasets, then use steps 7-8 to analyze the results contained in `03_model\out\Run_Manuscript_Results`.

If you would like to analyze the results of your own model run that you produced by running steps 1-6, you may need to manually reconfigure some of the scripts used in steps 7-8 of the pipeline, as there are elements that are hard-coded for the manuscript run.

## Prepare your environment

1) Download the model archive and unzip its contents.
2) Navigate into the root directory of the archive and use [mamba](https://mamba.readthedocs.io/en/latest/installation.html) to create the environment using `mamba env create -f environment.yaml`.
3) Activate your conda environment with `conda activate drb_estuary_salinity`.

## To train a new model:  

4) Run `snakemake -s Snakefile_fetch_munge -j` (-j runs the job on the available number of cpus cores, use -j 2 for fewer) to fetch and munge the data inputs used in the model. You might have to rerun the same command if an error that pops up, this is because snakemake doesn't run rules in the necessary order sometimes.
5) Open the file `03_model/model_config.yaml` and adjust modeling parameters. Change the run_id to whatever you want to name your run of the model (e.g. Test_Run).
6) Run `snakemake -s Snakefile_run_ml_model run_replicates -j` to train your model. Your model results will be written to `03_model/out/{run_id}/` where `{run_id}` is the run_id specified in `03_model/model_config.yaml`.

## To analyze model output and produce figures for the modeling run used in the associated manuscript (contained in 03_model\out\Run_Manuscript_Results):

7) To calculate functional performance for model output:
    - run `snakemake -s Snakefile_model_analysis calc_functional_performance_wrapper -j` and the results should be written to two csv files starting with `04_analysis/out/{run_id}_` where `{run_id}` is the run_id specified in `03_model/model_config.yaml`.
8) To reproduce manuscript figures:
    - check that the correct filepath for your RScript.exe is inserted in `rule generate_manuscript_figures` in `Snakefile_model_analysis`
    - run `snakemake -s Snakefile_model_analysis generate_manuscript_figures -j`
    - run `snakemake -s Snakefile_model_analysis generate_expected_gradient_figure -j`; note that this job may fail due to filesystem latency, but if you check in the `04_analysis/fig` folder you will see the output `fig_7_2019.pdf` has been generated
