configfile: "01_fetch/fetch_config.yaml"
configfile: "02_munge/munge_config.yaml"
configfile: "03_it_analysis/it_analysis_data_prep_config.yaml"
configfile: "03_it_analysis/it_analysis_preprocess_config.yaml"

rule all:
    input:
         "logfile.log"

rule fetch_usgs_nwis:
    input:
        "01_fetch/fetch_config.yaml"
    output:
        expand("01_fetch/out/usgs_nwis_{site}.txt", site=config["fetch_usgs.py"]["site_ids"]),
        "01_fetch/out/usgs_nwis_params.txt"
    shell:
        "python -m 01_fetch.src.fetch_usgs"

rule munge_usgs_nwis:
    input:
        expand("01_fetch/out/usgs_nwis_{site}.txt", site=config["fetch_usgs.py"]["site_ids"]),
        "01_fetch/out/usgs_nwis_params.txt"
    output:
        expand("02_munge/out/usgs_nwis_{site}.csv", site=config["fetch_usgs.py"]["site_ids"]),
        "02_munge/out/usgs_nwis_params.csv"
    shell:
        "python -m 02_munge.src.munge_usgs"

rule it_analysis_data_prep:
    input:
        expand("02_munge/out/usgs_nwis_{site}.csv", site=config["fetch_usgs.py"]["site_ids"]),
    output:
        "03_it_analysis/it_analysis_preprocess_config.yaml",
        "03_it_analysis/out/srcs",
        "03_it_analysis/out/snks"
    shell:
        "python -m 03_it_analysis.src.it_analysis_data_prep"

rule it_analysis_preprocess:
    input:
        "03_it_analysis/out/srcs",
        "03_it_analysis/out/snks"
    output:
        "logfile.log"
    shell:
        "python -m 03_it_analysis.src.it_analysis_preprocess.py"
