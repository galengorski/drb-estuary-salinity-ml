#===================================================================================#
# NOTES: functions for plotting the results figures for the manuscript
#-----------------------------------------------------------------------------------#
# Galen Gorski                                                                      #
# ggorski@usgs.gov                                                                  #
# 10-28-2022                                                                        #  
#-----------------------------------------------------------------------------------#
#===================================================================================#

#===================================================================================#
#####INSTALL PACKAGES#####
# install.packages('tidyverse')
library(tidyverse)
# install.packages('docstring')
library(docstring)
# install.packages('roxygen2')
library(roxygen2)
#####
#===================================================================================#


#===================================================================================#

#read in the model results for each replicate and cbind them in a data frame
aggr_reps <- function(run, reps){
  #' @describeIn 
  #' Join replicate model runs into a data frame and take the mean 
  #' 
  #' @param run the directory where the run can be found e.g. "Run_00"
  #' @param reps list of replicates e.g. c('00','01','02)
  #' 
  #' @returns tibble with observed salt front, each replicate predictions, and mean predictions
  #' as columns
  #' 
  model_results_df <- read_csv(paste0('03_model/out/', run,'/',reps[1],'/ModelResults.csv'), lazy = FALSE, show_col_types = FALSE)
  colnames(model_results_df)[c(1,3)] <- c('date','saltfront_pred_00')
  model_results_df <- model_results_df[,c('date','saltfront_obs','saltfront_pred_00')]
  for(i in 2:length(reps)){
    temp_data <- read_csv(paste0('03_model/out/', run,'/',reps[i],'/ModelResults.csv'), lazy = FALSE, show_col_types = FALSE)
    colnames(temp_data)[3] <- paste('saltfront_pred',reps[i], sep = '_')
    model_results_df <- cbind(model_results_df, temp_data[,3])
  }
  
  model_results_df$saltfront_pred_mean <- rowMeans(model_results_df[,c(3:ncol(model_results_df))])
  model_results_df <- model_results_df %>%
    as_tibble() %>%
    mutate(train_val = temp_data$`train/val`)
  
  return(model_results_df)
}


rmse_by_int <- function(data, sf_obs, sf_pred, model_run){
  #' @description calculate performance statisticcs by river mile
  #' 
  #' @param data dataframe with predictions, output from aggr_reps
  #' @sf_obs sring, column name of salt front observations
  #' @sf_pred string, column name of salt front predictions
  #' @model_run string, name of model run
  #' 
  #' @returns dataframe with performance metrics by river mile interval
  data$interval <- NA
  if(TRUE %in% is.na(data[sf_obs])){
    data[is.na(data[sf_obs]),][sf_obs]  <- -1  
  }
  
  data[data[sf_obs] <58 & data[sf_obs] >=0,]$interval <- '< 58'
  data[data[sf_obs] <68 & data[sf_obs] >=58,]$interval <- '58-68'
  data[data[sf_obs] <70 & data[sf_obs] >=68,]$interval <- '68-70'
  data[data[sf_obs] <78 & data[sf_obs] >=70,]$interval <- '70-78'
  data[data[sf_obs] <82 & data[sf_obs] >=78,]$interval <- '78-82'
  data[data[sf_obs] >=82,]$interval <- '> 82'
  
  if(TRUE %in% (data[sf_obs]  < 0)){
    data[data[sf_obs]  < 0,][sf_obs]  <- NA
  } 
  colnames(data)[1] <- 'date'
  n_years <- length(unique(year(data$date)))
  
  dat <- data[,c('date',sf_obs, sf_pred, 'interval')]
  names(dat) <- c('date', 'obs', 'pred', 'int')
  full_rmse <- round(hydroGOF::rmse(dat$pred, dat$obs), digits = 2)
  
  rmse_df_mean <- dat %>%
    group_by(int) %>%
    filter(!is.na(obs)) %>%
    summarise(rmse = round(hydroGOF::rmse(pred,obs), digits = 2), 
              nse = round(hydroGOF::NSE(pred,obs), digits = 2), 
              r = round(hydroGOF::rPearson(pred,obs), digits = 2),
              nobs_yr =round(n()/n_years,0), nobs = n()) %>%
    mutate(run = model_run)
  rmse_df_mean <- rbind(rmse_df_mean, c('All River Miles', full_rmse, 
                              round(hydroGOF::NSE(dat$pred,dat$obs), digits = 2),
                              round(hydroGOF::rPearson(dat$pred,dat$obs), digits = 2),
                              round(nrow(dat[!is.na(dat$obs),])/n_years,0), 
                              nrow(dat[!is.na(dat$obs),]), 
                              model_run))
  #if else statement for adding min and max rmses from replicates for river mile interval
  if(!grepl('coawst',model_run)){
    
    std_rmse_all_reps <- data %>%
      filter(!is.na(saltfront_obs)) %>%
      summarise(across(contains('saltfront_pred_0'), hydroGOF::rmse, obs = saltfront_obs)) %>%
      rowwise() %>%
      mutate(std_rmse = sd(c_across(saltfront_pred_00:saltfront_pred_09)))%>%
      dplyr::select(std_rmse)
      
    
    rmse_df <- data %>%
      group_by(interval)%>%
      filter(!is.na(saltfront_obs)) %>%
      summarise(across(contains('saltfront_pred_0'), hydroGOF::rmse, obs = saltfront_obs)) %>%
      rowwise() %>%
      mutate(std_rmse = sd(c_across(saltfront_pred_00:saltfront_pred_09)))%>%
      dplyr::select(interval, std_rmse) %>%
      add_row(interval = 'All River Miles', std_rmse = std_rmse_all_reps$std_rmse) %>%
      full_join(rmse_df_mean, by = c('interval' = 'int'))
    
  }else{
    
    rmse_df <- rmse_df_mean %>%
      mutate(std_rmse = NA) %>%
      relocate(std_rmse, .after = int) %>%
      rename('interval' = 'int')
  }
  
  
  rmse_df <- transform(rmse_df, rmse = as.numeric(rmse))
  
  return(rmse_df)
}

#calculate the lagged correlation between salt front and discharge
lagged_correlation <- function(sf_q, lag_col, lags){
  #' @description generic function to calculate the lagged correlation between observed and modeled
  #' salt front and another variable
  #' 
  #' @param sf_q dataframe with observed and modeled saltfront and discharge
  #' @param lag_col string, column to be lagged
  #' @param lags list of integers, lags to be investigated
  #' 
  #' @returns dataframe of correlation coefficients and lags
  corr_list <- vector()
  corr_list_pred <- vector()
  corr_list_coawst <- vector()
  
  for(lag in lags){
    sf_q[paste0(lag_col, '_', lag)] <- c(rep(NA,lag),(sf_q%>%pull(lag_col))[lag:nrow(sf_q)-lag])
    
    #obs
    corr <- hydroGOF::rPearson(sf_q[paste0(lag_col, '_', lag)], sf_q$saltfront_obs)
    corr_list <- c(corr_list, corr)
    #ml pred
    corr_pred <- hydroGOF::rPearson(sf_q[paste0(lag_col, '_', lag)], sf_q$saltfront_pred_mean)
    corr_list_pred <- c(corr_list_pred, corr_pred)
    #coawst pred
    corr_coawst <- hydroGOF::rPearson(sf_q[paste0(lag_col, '_', lag)], sf_q$coawst_pred)
    corr_list_coawst <- c(corr_list_coawst, corr_coawst)
  }
  corr_df <- data.frame(corr_list, corr_list_pred, corr_list_coawst, lags)
  return(corr_df)
}

#calculate annual lagged correlation
ann_lagged_correlation <- function(sf_q, lag_col, lags){
  #' @description a function for calculating the annual lagged correlation and merge it with the cumulative discharge
  #' 
  #' @param sf_q dataframe with observed and modeled saltfront and discharge
  #' @param lag_col string, column to be lagged
  #' @param lags list of integers, lags to be investigated
  #' 
  #' @returns dataframe of correlation coefficients and lags
  cum_flow <- sf_q %>% 
    rename('discharge' = lag_col) %>%
    #filter(month(date) %in% c(7,8,9,10)) %>%
    mutate(year = year(date)) %>% 
    group_by(year) %>% 
    summarise(cum_flow = sum(discharge))
  
  years <- unique(year(sf_q$date))
  corr_df_all_years <- data.frame()
  for(y in years){
    
    corr_list <- vector()
    corr_list_pred <- vector()
    corr_list_coawst <- vector()
    
    for(lag in lags){
      sf_q_year <- sf_q %>%
        filter(year(date) == y)
      sf_q_year[paste0(lag_col, '_', lag)] <- c(rep(NA,lag),(sf_q_year%>%pull(lag_col))[lag:nrow(sf_q_year)-lag])
      
      #obs
      corr <- hydroGOF::rPearson(sf_q_year[paste0(lag_col, '_', lag)], sf_q_year$saltfront_obs)
      corr_list <- c(corr_list, corr)
      #ml pred
      corr_pred <- hydroGOF::rPearson(sf_q_year[paste0(lag_col, '_', lag)], sf_q_year$saltfront_pred_mean)
      corr_list_pred <- c(corr_list_pred, corr_pred)
      #coawst pred
      corr_coawst <- hydroGOF::rPearson(sf_q_year[paste0(lag_col, '_', lag)], sf_q_year$coawst_pred)
      corr_list_coawst <- c(corr_list_coawst, corr_coawst)
    }
    
    corr_df_ann <- data.frame(corr_list, corr_list_pred, corr_list_coawst, lags) %>%
      arrange(corr_list) %>%
      dplyr::summarise(max_corr = first(corr_list), max_lag = first(lags)) %>%
      mutate(year = y)
    corr_df_all_years <- rbind(corr_df_all_years, corr_df_ann)
  }
  corr_df_all_years <- corr_df_all_years %>%
    left_join(cum_flow, by = 'year')
  
  return(corr_df_all_years)
}
