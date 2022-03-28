configfile: "01_fetch/fetch_config.yaml"
configfile: "02_munge/munge_config.yaml"
configfile: "03_it_analysis/it_analysis_data_prep_config.yaml"

rule all:
    input:
        "03_it_analysis/out/srcs_proc_lagged",
        "03_it_analysis/out/snks_proc",
        expand("02_munge/out/H/noaa_nos_{site}.csv", site=config["fetch_noaa_nos.py"]["station_ids"])

rule fetch_usgs_nwis:
    input:
        "01_fetch/fetch_config.yaml"
    output:
        expand("01_fetch/out/usgs_nwis_{usgs_site}.txt", usgs_site=config["fetch_usgs.py"]["site_ids"]),
    shell:
        "python -m 01_fetch.src.fetch_usgs"
        
rule fetch_noaa_nos:
    input:
        "01_fetch/fetch_config.yaml"
    output:
        expand("01_fetch/out/noaa_nos_{noaa_site}_{noaa_product}.txt", noaa_site=config["fetch_noaa_nos.py"]["station_ids"], noaa_product = config["fetch_noaa_nos.py"]["products"]),
    shell:
        "python -m 01_fetch.src.fetch_noaa_nos"


rule munge_usgs_nwis:
    input:
        expand("01_fetch/out/usgs_nwis_{usgs_site}.txt", usgs_site=config["fetch_usgs.py"]["site_ids"]),
    output:
        expand("02_munge/out/D/usgs_nwis_{usgs_site}.csv", usgs_site=config["fetch_usgs.py"]["site_ids"]),
    shell:
        "python -m 02_munge.src.munge_usgs"
        
rule munge_noaa_nos:
    input:
        expand("01_fetch/out/noaa_noss_{noaa_site}_{noaa_product}.txt", noaa_site=config["fetch_noaa_nos.py"]["station_ids"], noaa_product = config["fetch_noaa_nos.py"]["products"])
    output:
        expand("02_munge/out/H/noaa_nos_{site}.csv", site=config["fetch_noaa_nos.py"]["station_ids"]),
        expand("02_munge/out/daily_summaries/noaa_nos_{site}.csv", site=config["fetch_noaa_nos.py"]["station_ids"])
    shell:
        "python -m 02_munge.src.munge_noaa_nos"

rule it_analysis_data_prep:
    input:
        expand("02_munge/out/D/usgs_nwis_{usgs_site}.csv", usgs_site=config["fetch_usgs.py"]["site_ids"]),
        expand("02_munge/out/daily_summaries/noaa_nos_{noaa_site}.csv", noaa_site=config["fetch_noaa_nos.py"]["station_ids"]),
    output:
        "03_it_analysis/out/srcs_proc_lagged",
        "03_it_analysis/out/snks_proc"
    shell:
        "python -m 03_it_analysis.src.it_analysis_data_prep"


