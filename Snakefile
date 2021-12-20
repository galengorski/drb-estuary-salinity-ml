configfile: "01_fetch/fetch_config.yaml"
configfile: "02_munge/munge_config.yaml"

rule all:
    input:
        expand("02_munge/out/usgs_nwis_{site}.csv", site=config["fetch_usgs.py"]["site_ids"]),
        "02_munge/out/usgs_nwis_params.csv"

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
