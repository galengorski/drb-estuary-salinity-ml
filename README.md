# drb-estuary-salinity-ml

This model archive contains code used for deep learning and hydrodynamic modeling aimed at understanding and simulating salinity dynamics in the Delaware Bay. The steps for reproducing the results from the manuscript are detailed below.

You will need to have [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), [R](https://cran.r-project.org/), and [mamba](https://mamba.readthedocs.io/en/latest/installation.html) installed on your computer to run this pipeline.

To retrain a new model run all steps below (1-8). If you only want to analyze results from the manuscript, run steps 1-3, then use steps 7-8 for analysis. Model ouptut used for the manuscript is located in `03_model/out/Manuscript_Results`.

## Prepare your environment

1) Download the model archive and unzip its contents.
2) Navigate into the root directory of the archive and use [mamba](https://mamba.readthedocs.io/en/latest/installation.html) to create the environment using `mamba env create -f environment.yaml`.
3) Activate your conda environment with `conda activate drb_estuary_salinity`.

## To train the model:  

4) Run `snakemake -s Snakefile_fetch_munge -j` (-j runs the job on the available number of cpus cores, use -j 2 for fewer) to fetch and munge the data inputs used in the model. You might have to rerun the same command if an error that pops up, this is because snakemake doesn't run rules in the necessary order sometimes.
5) Open the file `03_model/model_config.yaml` and adjust modeling parameters. Change the run_id to whatever you want to name your run of the model (e.g. Test_Run). **Note**: If you do not change the `run_id` to something other than 'Run_Manuscript_Results', the manuscript results will be overwritten in the next step.
6) Run `snakemake -s Snakefile_run_ml_model run_replicates -j` to train your model. Your model results will be written to `03_model/out/{run_id}/` where `{run_id}` is the run_id specified in `03_model/model_config.yaml`.

## To analyze model output and produce figures:

7) To calculate functional performance for model output:
    - make sure that the parameters in `Snakefile_model_analysis` lines 9-13 describe the desired model run, sources, sinks, and years for analysis (note: functional performance is calculated on an annual basis).
    - run `snakemake -s Snakefile_model_analysis calc_functional_performance_wrapper -j` and the results should be written to two csv files starting with `04_analysis/out/{run_id}_` where `{run_id}` is the run_id specified in `03_model/model_config.yaml`.
8) To reproduce manuscript figures:
    - make sure that the `run_id` parameter in `04_analysis/src/results.R` describes the correct model run (it should probably match the run_id you specifed in `03_model/model_config.yaml`)
    - check that the correct filepath for your RScript.exe is inserted in `rule generate_manuscript_figures` in `Snakefile_model_analysis`
    - run `snakemake -s Snakefile_model_analysis generate_manuscript_figures -j`
    - run `snakemake -s Snakefile_model_analysis generate_expected_gradient_figure -j`; note that this job may fail due to filsystem latency, but if you check in the `04_analysis/fig` folder you will see the output `fig_7_201.pdf` has been generated
