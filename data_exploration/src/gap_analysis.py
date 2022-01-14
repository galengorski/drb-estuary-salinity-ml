import os
import yaml
import utils
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def compile_data(nwis_var_names, source):
    # select munged files we want to fetch data for
    files_to_process = [f for f in os.listdir(os.path.join('.', '02_munge', 'out')) if f.startswith(f'{source}_')]

    # create dataframe per variable
    var_dfs = {}
    for var in nwis_var_names:
        var_dfs[var] =  pd.DataFrame()
    for f in files_to_process:
        print(f'processing {f}')
        # get site id
        site_id = os.path.splitext(f)[0].split('_')[-1]
        # read in data per site and append it to the appropriate dataframe
        site_data_csv = os.path.join('.', '02_munge', 'out', f)
        site_data_df = pd.read_csv(site_data_csv, index_col='datetime')
        # make index into datetime type
        site_data_df.index = pd.to_datetime(site_data_df.index)
        for var in site_data_df.columns:
            var_dfs[var][site_id] = site_data_df[var]
    return var_dfs

def gap_analysis_calc(source, var_dfs):
    metrics = ['p_coverage', 'n_gaps', 'gap_median_days', 'gap_max_days']
    gap_template_df = pd.DataFrame(columns=metrics)
    metric_dfs = {}
    for var, df in var_dfs.items():
        df.dropna(axis=0, how='all', inplace=True)
        if df.empty:
            continue
        metric_dfs[var] = {}
        # get list of years available for this variable
        # # include all years with any measurements at any of our sites
        years = df.index.year.unique()
        for site in df.columns:
            var_site_gap_df = gap_template_df.copy()
            for year in years:
                year_df = df[df.index.year==year][site]
                # year_df.dropna(inplace=True)
                var_site_gap_df.loc[year, 'p_coverage'] = year_df.count()/365                
                deltas = year_df.dropna().index.to_series().diff()[1:]
                gaps = deltas[deltas > dt.timedelta(days=1)]
                var_site_gap_df.loc[year, 'n_gaps'] = len(gaps)
                var_site_gap_df.loc[year, 'gap_median_days'] = gaps.median().days if pd.notna(gaps.median().days) else 0
                var_site_gap_df.loc[year, 'gap_max_days'] = gaps.max().days if pd.notna(gaps.max().days) else 0
                var_site_gap_df.to_csv(os.path.join('data_exploration', 'out', f'{source}_{var}_{site}_gap_analysis.csv'))
            metric_dfs[var][site]= var_site_gap_df
    return metric_dfs

def plot_gap_analysis(source, metric_dfs, site_colors):
    for var, data_by_site in metric_dfs.items():
        plot_df = pd.DataFrame()
        fig, axs = plt.subplots(4, sharex=True, figsize=(8,8))
        i=0
        for metric in metrics:
            for site, df in data_by_site.items():
                #df.plot.line(y='p_coverage', label=site)
                plot_df[site] = df[metric]
            #years_to_keep
            lines = plot_df.plot.line(ylabel=f'{metric}', ax=axs[i], legend=False, color=[site_colors.get(x) for x in plot_df.columns])
            i+=1
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.15,0.9), loc='upper right')
        fig.suptitle(var)
        save_path = os.path.join('data_exploration', 'out', f'{source}_{var}_gap_analysis_plot.png')
        fig.savefig(save_path, bbox_inches = 'tight')

def main():
    # import config
    with open("02_munge/munge_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['munge_usgs.py']
    # read in list of variables of interest
    var_codes = config['params_to_process']
    # change to machine+human-readable names
    var_names = [utils.usgs_nwis_param_code_to_name(code) for code in var_codes]

    # import site:color mapping from config
    with open("data_exploration/data_exploration_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['gap_analysis.py']
    # read in data source we want to do gap analysis for
    source = config['source']
    os.makdirs('data_exploration/out/', exist_ok  = True)
    # fetch site data and compile into nested dictionary of dataframes
    var_dfs = compile_data(nwis_var_names, source)

    # calculate gap metrics per site variable combo
    metric_dfs = gap_analysis_calc(source, var_dfs)

    # plot gap metrics per variable
    site_colors = config[f'{source}_site_colors']
    plot_gap_analysis(source, metric_dfs, site_colors)

if __name__ == '__main__':
    main()