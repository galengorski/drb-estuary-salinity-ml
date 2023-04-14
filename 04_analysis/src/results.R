#===================================================================================#
# NOTES: script for making results figures for manuscript
#-----------------------------------------------------------------------------------#
# Galen Gorski                                                                      #
# ggorski@usgs.gov                                                                  #
# 08-10-2022                                                                        #  
#-----------------------------------------------------------------------------------#
#===================================================================================#

#===================================================================================#
#####INSTALL PACKAGES#####
pkgLoad <- function( packages = "favourites" ) {

    if( length( packages ) == 1L && packages == "favourites" ) {
        packages <- c( "cowplot", "ggplot2", "gridExtra", "hexbin", "hydroGOF",
                       "lubridate", "ModelMetrics", "scales", "tidyverse", 
                       "patchwork", "viridis"
        )
    }

    packagecheck <- match( packages, utils::installed.packages()[,1] )

    packagestoinstall <- packages[ is.na( packagecheck ) ]

    if( length( packagestoinstall ) > 0L ) {
        utils::install.packages( packagestoinstall,
                             repos = "http://cran.csiro.au"
        )
    } else {
        print( "All requested packages already installed" )
    }

    for( package in packages ) {
        suppressPackageStartupMessages(
            library( package, character.only = TRUE, quietly = TRUE )
        )
    }

}

pkgLoad()


# install.packages('cowplot')
library(cowplot)
# install.packages('ggplot2')
library(ggplot2)
# install.packages('gridExtra')
library(gridExtra)
# install.packages('hexbin')
library(hexbin)
# install.packages('hydroGOF')
library(hydroGOF)
# intstall.packages('lubridate')
library(lubridate)
# install.packages('ModelMetrics')
library(ModelMetrics)
#install.packages("scales")
library(scales)
# install.packages('tidyverse')
library(tidyverse)
# install.packages('patchwork')
library(patchwork)
#install.packages("viridis")
library(viridis)

#load plotting functions
source('04_analysis/src/results_functions.R')
dir.create('04_analysis/fig', showWarnings = FALSE)
#####
#===================================================================================#

#===================================================================================#

#run id
run <- 'Run_Manuscript_Results'
#number and ids of reps
reps <- c( '00', '01', '02', '03', '04', '05', '06', '07', '08', '09')
#aggregate results from replicates
model_results_df <- aggr_reps(run, reps)
#subset results into training and validation
train_set <-  model_results_df[model_results_df$train_val == 'Training',] 
val_set <- model_results_df[model_results_df$train_val == 'Validation',]
#caculate rmse by interval for validation
model_results_in_val <- rmse_by_int(model_results_df %>% filter(train_val == 'Validation'),"saltfront_obs","saltfront_pred_mean", "2016-2020")

#read in COAWST results
c_16 <- read_csv('03_model/in/COAWST_model_runs/processed/COAWST_2016_7day.csv', show_col_types = FALSE)
c_18 <- read_csv('03_model/in/COAWST_model_runs/processed/COAWST_2018_7day.csv', show_col_types = FALSE)
c_19 <- read_csv('03_model/in/COAWST_model_runs/processed/COAWST_2019_7day.csv', show_col_types = FALSE)

c_all <- rbind(c_16, c_18, c_19)


####-----------Figure 2---------------####
print("Creating Figure 2...")

trenton <- read_csv('02_munge/out/D/usgs_nwis_01463500.csv', show_col_types = FALSE) %>% 
  dplyr::select(datetime, discharge) %>%
  rename_with(.fn = ~paste0(.,'_del'), .cols = everything())

schuylkill <- read_csv('02_munge/out/D/usgs_nwis_01474500.csv', show_col_types = FALSE) %>%
  dplyr::select(datetime, discharge) %>%
  rename_with(.fn = ~paste0(.,'_sch'), .cols = everything())

sf_q <- model_results_df %>%
  dplyr::select(date, saltfront_obs, saltfront_pred_mean) %>%
  left_join(trenton, by = c('date' = 'datetime_del')) %>%
  left_join(schuylkill, by = c('date' = 'datetime_sch')) %>%
  left_join(c_all, by = c('date' = 'date'))

sf_del_corr <- lagged_correlation(sf_q, 'discharge_del', seq(0,50))
sf_sch_corr <- lagged_correlation(sf_q, 'discharge_sch', seq(0,50))

ts_flow <- sf_q %>%
  rename('Delaware' = 'discharge_del', 'Schuylkill' = 'discharge_sch') %>%
  pivot_longer(cols = c(Delaware, Schuylkill)) %>%
  dplyr::select(date, name, value) %>%
  rename('Site' = 'name', 'Discharge' = 'value') %>%
  mutate(year = as.factor(year(date)), month = as.factor(month(date)))

ts_flow %>%
  filter(Site == 'Schuylkill') %>%
  group_by(year) %>%
  summarise(sum_q = sum(Discharge)) %>%
  arrange(sum_q)


ts_flow_plot_l <- ts_flow %>%
  ggplot(aes(x = month, y = Discharge, fill = Site))+
  geom_boxplot()+
  scale_fill_manual(values = c('#c1121f','#669bbc'))+
  theme_bw()+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        strip.text.x = element_text(size = 14),
        legend.text = element_text(size=14, color = 'black'),
        legend.key.size = unit(1, 'cm'), #change legend key size
        legend.key.height = unit(1, 'cm'), #change legend key height
        legend.key.width = unit(1, 'cm'),
        legend.title = element_blank(),
        axis.text.x=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  scale_y_log10()+
  ylab('Discharge (cfs)')+
  xlab('')

flow_leg <- get_legend(ts_flow_plot_l)
ts_flow_plot <- ts_flow_plot_l + theme(legend.position = 'none')

sf_obs <- model_results_df %>%
  mutate(site = 'salt_front') %>%
  dplyr::select(date, saltfront_obs, site) %>%
  filter(year(date) > 2000 & year(date) < 2021) %>%
  mutate(month = as.factor(month(date))) %>%
  ggplot(aes(x = month, y = saltfront_obs, fill = site))+
  geom_boxplot(fill = '#57cc99')+
  theme_bw()+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        strip.text.x = element_text(size = 14),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  scale_y_log10()+
  ylab('Salfront location (RM)')+
  scale_x_discrete(name = '', breaks = seq(1,12), labels = c('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

corr_plot_df <- rbind(sf_sch_corr %>%
                        mutate(site = 'Schuylkill'),
                      sf_del_corr %>%
                        mutate(site = 'Delaware'))

corr_plot <- corr_plot_df %>%
  ggplot(aes(x = lags, y = corr_list, color = site))+
  geom_line(size = 1.5)+
  scale_color_manual(values = c('#c1121f','#669bbc'))+
  theme_bw()+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        legend.position = 'none')+
  labs(x = 'Lag (days)', y = 'Pearson r')+
  geom_vline(xintercept = 7, linetype = 2, color = '#c1121f', size = 1)+
  geom_vline(xintercept = 8, linetype = 2, color = '#669bbc', size = 1)




#pdf('04_analysis/fig/fig_2.pdf', height = 6, width = 10)
fig_2 <- plot_grid(ts_flow_plot,flow_leg, sf_obs, corr_plot, ncol = 2, align = 'v', axis = 'rl', rel_widths = c(2,1),
          labels = c('a)','','b)','c)'))
ggsave('04_analysis/fig/fig_2.pdf', fig_2, height = 6, width = 10)
#dev.off()

####-----------FIGURE 3---------------####
#validation time series
print("Creating Figure 3...")
val_ts <-model_results_df %>%
  filter(train_val == 'Validation') %>%
  ggplot()+
  geom_point(aes(x = date, y = saltfront_obs), fill = 'gray30', shape = 21,
             show.legend = FALSE, size = 2.3, color = 'black', alpha = 0.3)+
  theme_classic()+
  ylim(54,90)+
  xlab('')+
  ylab('7-day average\nsalt front location\n(River mile)')+
  geom_line(data = model_results_df %>% filter(train_val == 'Validation'),
            aes(x = date, y = saltfront_pred_mean), size = 0.9,
            color = '#00b4d8')+
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))

#training time series
trn_ts <-model_results_df %>%
  filter(train_val == 'Training') %>%
  ggplot()+
  geom_point(aes(x = date, y = saltfront_obs), fill = 'gray30', shape = 21,
             show.legend = FALSE, size = 2.3, color = 'black', alpha = 0.3)+
  theme_classic()+
  ylim(54,90)+
  xlab('')+
  ylab('7-day average\nsalt front location\n(River mile)')+
  geom_line(data = model_results_df %>% filter(train_val == 'Training'),
            aes(x = date, y = saltfront_pred_mean), size = 0.7,
            color = 'indianred2')+
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))

#validation cross plot
val_xplot <- val_set %>%
  ggplot(aes(x = saltfront_obs, y = saltfront_pred_mean))+
  geom_abline(slope = 1, intercept = 0, linetype = 2)+
  geom_hex(bins = 50)+
  scale_fill_gradient(low = "#00b4d8",high = "midnightblue")+
  theme_classic() +
  theme(legend.position = 'none')+
  ylim(55,90)+
  xlim(55,90)+
  ylab('Predicted Salt Front\n(River mile)')+
  xlab('Observed Salt Front\n(River mile)')+
  ggtitle('Testing')+
  theme(plot.title = element_text(color = "#00b4d8"))+
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))

#training cross plot
train_xplot <- train_set %>%
  ggplot(aes(x = saltfront_obs, y = saltfront_pred_mean))+
  geom_abline(slope = 1, intercept = 0, linetype = 2)+
  geom_hex(bins = 50)+
  scale_fill_gradient(low = "indianred2",high = "orangered4")+
  theme_classic() +
  theme(legend.position = 'none')+
  ylim(55,90)+
  xlim(55,90)+
  ylab('Predicted Salt Front\n(River mile)')+
  xlab('Observed Salt Front\n(River mile)')+
  ggtitle('Training')+
  theme(plot.title = element_text(color = "indianred2"))+
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))

#validation rmse by interval
val_int <- model_results_in_val %>%
  mutate(interval = factor(interval,
                           levels = c("< 58", "58-68", "68-70",
                                      "70-78", "78-82","> 82","All River Miles"))) %>%
  ggplot(aes(x = interval, y = rmse, fill = interval))+
  geom_bar(stat = 'identity', color = 'black', position = position_dodge())+
  geom_errorbar(aes(ymin = rmse-std_rmse , ymax = rmse+std_rmse ), width = .2)+
  labs(title="RMSE by river mile interval", x="River mile interval", y = "RMSE (River mile)")+
  theme_classic() +
  scale_fill_manual(values = c('#00b4d8','#00b4d8','#00b4d8','#00b4d8','#00b4d8','#00b4d8','#0049b6'))+
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 12),
        legend.position = 'None')

lay <- rbind(c(1,1,2,2,2,2,2,2), c(3,3,4,4,4,4,5,5))
#pdf('04_analysis/fig/fig_3.pdf', height = 5, width = 10)
fig_3 <- grid.arrange(train_xplot, trn_ts, val_xplot, val_ts,val_int, layout_matrix = lay)
ggsave('04_analysis/fig/fig_3.pdf', fig_3, height = 5, width = 10)
dev.off()
####---------------------------------####


####-----------FIGURE 4---------------####
print("Creating Figure 4...")


#Bar plots comparing ML and COAWST
#run <- 'Run_00'
reps <- c( '00', '01', '02', '03', '04', '05', '06', '07', '08', '09')
#aggregate results from replicates
model_results_df <- aggr_reps(run, reps)
#subset results into training and validation
train_set <-  model_results_df[model_results_df$train_val == 'Training',]
val_set <- model_results_df[model_results_df$train_val == 'Validation',]

#merge coawst with observations
ml_c <- merge(model_results_df[,c('date','saltfront_obs')], c_all, by = 'date', all.x = TRUE, all.y = TRUE) %>%
 as_tibble()

#calculate rmse by interval for all coawst years
model_results_in_val_coawst <- rmse_by_int(ml_c,"saltfront_obs","coawst_pred","coawst")


dummy_df <- rbind(
  tibble(interval = c("< 58", "58-68", "68-70", "70-78", "78-82","> 82","All River Miles"),
         std_rmse = rep(NA,7),
         rmse = rep(NA,7),
         nse = rep(NA,7),
         r = rep(NA,7),
         nobs_yr = rep(NA,7),
         nobs = rep(NA,7),
         run = rep('coawst_2017',7)),
  tibble(interval = c("< 58", "58-68", "68-70", "70-78", "78-82","> 82","All River Miles"),
         std_rmse = rep(NA,7),
         rmse = rep(NA,7),
         nse = rep(NA,7),
         r = rep(NA,7),
         nobs_yr = rep(NA,7),
         nobs = rep(NA,7),
         run = rep('coawst_2020',7)))

rmse_year <- rbind(
  rmse_by_int(model_results_df %>% filter(year(date) == '2016'),"saltfront_obs","saltfront_pred_mean","ml_2016"),
  rmse_by_int(model_results_df %>% filter(year(date) == '2017'),"saltfront_obs","saltfront_pred_mean","ml_2017"),
  rmse_by_int(model_results_df %>% filter(year(date) == '2018'),"saltfront_obs","saltfront_pred_mean","ml_2018"),
  rmse_by_int(model_results_df %>% filter(year(date) == '2019'),"saltfront_obs","saltfront_pred_mean","ml_2019"),
  rmse_by_int(model_results_df %>% filter(year(date) == '2020'),"saltfront_obs","saltfront_pred_mean","ml_2020"),

  rmse_by_int(ml_c %>% filter(year(date) == '2016'),"saltfront_obs","coawst_pred","coawst_2016"),
  rmse_by_int(ml_c %>% filter(year(date) == '2018'),"saltfront_obs","coawst_pred","coawst_2018"),
  rmse_by_int(ml_c %>% filter(year(date) == '2019'),"saltfront_obs","coawst_pred","coawst_2019"),
  dummy_df)

rmse_year$model <- str_split_fixed(rmse_year$run, '_', 2)[,1]
rmse_year$year <- str_split_fixed(rmse_year$run, '_', 2)[,2]

rmse_plot <- rmse_year %>%
  filter(interval == 'All River Miles') %>%
  mutate(model = factor(model, levels = c('ml','coawst'))) %>%
  ggplot(aes(x = year, y = as.numeric(rmse), fill = model))+
  geom_bar(stat = 'identity', color = 'black', position = position_dodge())+
  geom_errorbar(aes(ymin = rmse-std_rmse, ymax = rmse+std_rmse), width = .2, position = position_dodge(0.9))+
  theme_bw() +
  scale_fill_manual(values = c('#00b4d8','#feb24c'))+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        strip.text.x = element_text(size = 14),
        axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        strip.background =element_rect(fill="white"),
        legend.position = 'none')+
  ylab('RMSE (River mile)')+
  xlab('Year')

comp_by_int <- rbind(model_results_in_val,
                     model_results_in_val_coawst) %>%
  mutate(run = factor(run,
                      levels = c('coawst','2016-2020'))) %>%
  mutate(interval = factor(interval,
                           levels = c("< 58", "58-68", "68-70",
                                      "70-78", "78-82","> 82","All River Miles"))) %>%
  mutate(run = factor(run, levels = c('2016-2020','coawst'))) %>%
  ggplot(aes(x = interval, y = rmse, fill = run))+
  geom_bar(stat = 'identity', color = 'black', position = position_dodge())+
  geom_errorbar(aes(ymin = rmse-std_rmse, ymax = rmse+std_rmse), width = .2, position = position_dodge(0.9))+
  labs(x="River mile interval", y = "RMSE (River mile)")+
  theme_bw() +
  scale_fill_manual(values = c('#00b4d8','#feb24c'))+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        strip.text.x = element_text(size = 14),
        axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        strip.background =element_rect(fill="white"),
        legend.position = 'none')


#pdf('04_analysis/fig/fig_4.pdf', height = 3, width = 10)
fig_4 <- plot_grid(rmse_plot, comp_by_int, nrow = 1, align = 'l', labels = c('a)','b)'))
ggsave('04_analysis/fig/fig_4.pdf', fig_4, height = 3, width = 10)
#dev.off()
####---------------------------------####


####-----------FIGURE 5---------------####
print("Creating Figure 5...")

#merge coawst with observations
ml_c <- merge(model_results_df[,c('date','saltfront_obs','saltfront_pred_mean')], c_all, by = 'date', all.x = TRUE, all.y = TRUE) %>%
  as_tibble() %>% 
  rename('ml_pred' = 'saltfront_pred_mean')

#Time series comparison of ML ML Lagged and COAWST
comp_years_ml <- ml_c %>%
  mutate(year = year(date)) %>%
  filter(year %in% c('2016','2018','2019'))


ts_comp <- ggplot(data = comp_years_ml)+
  geom_point(aes(x = date, y = saltfront_obs), fill = 'gray30', shape = 21,
             show.legend = FALSE, size = 2.3, color = 'black', alpha = 0.3)+
  theme_bw()+
  xlab('')+
  ylab('7-day average\nsalt front location\n(River mile)')+
  geom_line(data = comp_years_ml,
            aes(x = date, y = ml_pred), size = 1.2,
            color = '#00b4d8')+
  geom_line(data = comp_years_ml,
            aes(x = date, y = coawst_pred), size = 1.2,
            color = '#feb24c')+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        strip.text.x = element_text(size = 14),
        axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        strip.background =element_rect(fill="white"))+
  facet_wrap(~year, ncol = 1, scales = 'free_x')#+


int <- rmse_year %>%
  filter(year %in% c('2016','2018','2019')) %>%
  filter(!grepl('mll', model)) %>%
  mutate(model = factor(model, levels = c('ml','coawst'))) %>%
  mutate(interval = factor(interval,
                           levels = c("< 58", "58-68", "68-70",
                                      "70-78", "78-82","> 82","All River Miles"))) %>%
  filter(interval != 'All River Miles') %>%
  ggplot(aes(x = interval, y = rmse, fill = model))+
  geom_bar(stat = 'identity', color = 'black', position = position_dodge())+
  theme_bw() +
  scale_fill_manual(values = c('#00b4d8','#feb24c'))+
  facet_wrap(~year, ncol = 1, scales='free_x')+
  scale_x_discrete(limits=c("< 58", "58-68", "68-70",
                            "70-78", "78-82","> 82"))+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        strip.text.x = element_text(size = 14),
        axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        strip.background =element_rect(fill="white"),
        legend.position = 'none')+
  ylab('RMSE (River mile)')+
  xlab('River mile interval')


##pdf('04_analysis/fig/fig_5.pdf', height = 8.64, width = 10.51)
fig_5 <- plot_grid(ts_comp, int, ncol = 2)
ggsave('04_analysis/fig/fig_5.pdf', fig_5, height = 8.64, width = 10.51)
#dev.off()
####---------------------------------####


####-----------FIGURE 6---------------####
print("Creating Figure 6...")
#Functional performance
fp_ann <- read_csv(paste0('04_analysis/out/',run,'_functional_performance_df.csv'), show_col_types = FALSE) %>%
  separate(source, c('source', 'site')) %>%
  mutate(site = replace(site, site == '01463500', 'Schuylkill')) %>%
  mutate(site = replace(site, site == '01474500', 'Delaware')) %>%
  mutate(train_set = if_else(year %in% c('2016','2017','2018','2019','2020'), 'Testing','Training')) %>%
  filter(train_set == 'Testing') %>%
  filter((lag == 7 & site == 'Delaware')|(lag == 8 & site == 'Schuylkill'))

shape <- c(24,23,22,21,20)

fp <-
  fp_ann %>%
  filter(year %in% c('2016','2018','2019')) %>%
  mutate(year = as.character(year)) %>%
  left_join(rmse_year %>% filter(interval == 'All River Miles'), by = c('year', 'model')) %>%
  ggplot(aes(x = Func_Perf, y = rmse, fill = model, shape = year)) +
  geom_point(size = 8)+
  scale_fill_manual(values = c('#feb24c','#00b4d8','#80ed99'))+
  scale_shape_manual(values = shape)+
  theme_bw()+
  theme(axis.text = element_text(size = 14, color = 'black'),
        axis.title = element_text(size = 14),
        strip.text.x = element_text(size = 14),
        strip.background =element_rect(fill="white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()#,
        #legend.position = 'none'
  )+
  ylab('RMSE (River Mile)')+
  xlab('Functional performance')+
  facet_wrap(~site)+
  xlim(-0.1, 0.1)+
  geom_vline(xintercept=0,
             linetype=2, colour="black")

#pdf('04_analysis/fig/fig_6.pdf', height = 5, width = 10)
ggsave('04_analysis/fig/fig_6.pdf',fp, height = 5, width = 10)
#dev.off()
####---------------------------------####

