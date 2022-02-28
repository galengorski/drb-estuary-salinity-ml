#===================================================================================#
# NOTES: this script is for querying gridMET for daily meteorological data for 
# specific sites
#-----------------------------------------------------------------------------------#
# Galen Gorski                                                                      #
# ggorski@usgs.gov                                                                  #
# 02-24-2022                                                                        #  
#-----------------------------------------------------------------------------------#
#===================================================================================#

#===================================================================================#
#####INSTALL PACKAGES#####
# install.packages('tidyverse')
library(tidyverse)
#remotes::install_github("mikejohnson51/climateR")
library(climateR)
# install.packages('sf')
library(sf)
# install.packages('dplyr')
library(dplyr)
#####
#===================================================================================#


#===================================================================================#
lat = c(
  38.7816,
  39.9538,
  39.50083)

long = c(-75.119,
         -75.137,
         -75.56861)

site = c('lewes',
         'ben_franklin',
         'reedy_island_jetty')

df <- data.frame(site = site, lat = lat, long = long)
sites_sf <- st_as_sf(df, coords = c("long","lat"), crs = 4326, agr = "constant")

for(i in 1:3){
  ts  = getGridMET(sites_sf[i,], param = c('prcp','tmax','tmin','wind_dir','wind_vel'), startDate = "2000-12-01", endDate = "2019-12-31")
  
  ts %>%
    mutate(wspdir = -1*wind_vel*cos(wind_dir*(pi/180))) %>%
    mutate(datetime = date) %>%
    dplyr::select(datetime, prcp, tmax, tmin, wspdir) %>%
    write_csv('01_fetch/out/gridmet_',site[i],'.csv')
}
