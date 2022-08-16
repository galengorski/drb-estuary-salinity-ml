#===================================================================================#
# NOTES: create a finalized record of the salf front location from data sent 
# from Amy Shallcross of DRBC on 2/3/2022
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
sfs <- read_excel('../../99_extra_data/saltfront_record/Copy of Saltfront_Historic_Official_Corrected_UPDATED_JAN_2022.xlsx')
colnames(sfs) <- c('datetime','saltfront_daily','saltfront7_weekly')
sfs <- sfs[sfs$datetime >= as_date('2000-01-01') ,]
write_csv(sfs, '03a_it_analysis/in/saltfront_updated.csv')
