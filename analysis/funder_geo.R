## Purpose: Analyze funders by country w/ associated agency and biodata resource names and counts
## Parts: 1) reshape and analyze and 2) save output
## Package(s): tidyverse
## Input file(s): funders_geo_200.csv
## Output file(s): funders_geo_counts_2023-01-21.csv

library(tidyverse)

## manually curated file with countries determiend 
top <- read.csv("funders_geo_200.csv")

## count number of agencies per country

##remove extra spaces or won't deduplicate cleanly
top$associated_biodata_resources <- gsub('[\" ]', '', top$associated_biodata_resources)

com <- top %>% 
  group_by(country) %>% 
    mutate(count_agencies = length(agency)) %>%
        mutate(agency_names = str_c(agency, collapse = ", ")) %>%
          mutate(resource_names = str_c(unique(associated_biodata_resources), collapse = ","))

## need to same reshaping to be able to dedupliate
com$resource_names_split <- strsplit(com$resource_names, ",")
com2 <- unique(select(com, 2,3, 10:13))
com2$resource_names_split_u <- sapply(com2$resource_names_split, unique)

## count and assemble table
com2 <- com2 %>% 
  group_by(country) %>% 
    mutate(count_resources = length(unlist(resource_names_split_u)))  %>% 
      mutate(biodata_resource_names = str_c(flatten(resource_names_split_u), collapse = ", "))
com3 <- select(com2, 1, 2, 3, 8, 4, 9)

##=====================================##
####### PART 2: Save output files ####### 
##=====================================##

write.csv(com3,"funders_geo_counts_2023-01-21.csv", row.names = FALSE)

