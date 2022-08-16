#===================================================================================#
# NOTES: create a finalized record of the salf front location from data sent 
# from Amy Shallcross of DRBC on 8/16/2022
#-----------------------------------------------------------------------------------#
# Galen Gorski                                                                      #
# ggorski@usgs.gov                                                                  #
# 08-16-2022                                                                        #  
#-----------------------------------------------------------------------------------#
#===================================================================================#

#===================================================================================#
#####INSTALL PACKAGES#####
# install.packages('tidyverse')
library(tidyverse)
library(readxl)
library(lubridate)
#####
#===================================================================================#


#===================================================================================#
sfs <- read_excel('02_munge/in/Saltfront_Historic_Record_AvgRM_WithInterp_Updated_2022_08_15.xlsx')
colnames(sfs) <- c('datetime','saltfront_daily','saltfront7_weekly')
sfs <- sfs[sfs$datetime >= as_date('2000-01-01') ,]
write_csv(sfs, '03a_it_analysis/in/saltfront_updated.csv')
