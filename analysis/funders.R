## Purpose: Retrieve and analyze funder metadata
## Parts: 1) Retrieve funder metadata from Europe PMC, 2) analyze, and 3) save output files
## Package(s): europepmc, tidyverse, reshape2
## Input file(s): final_inventory_2022.csv
## Output file(s): inventory_funders_2023-01-20.csv

##========================================================================##
####### PART 1: Retrieve funder metadata per article from Europe PMC ####### 
##========================================================================##

## Note: biodata resources in the inventory have concatenated funder info when they have >1 article. The step below retrieves metadata for each article for analysis by article. 

library(europepmc)
library(tidyverse)
library(reshape2)

## split and melt IDs from the inventory into a list of just IDs 
inv <- read.csv("final_inventory_2022.csv") ## reminder - setwd to data folder
inventory <- inv ## will use later in script
inv <- separate(inv, 'ID', paste("ID", 1:30, sep="_"), sep=",", extra="drop")
inv <- inv[,colSums(is.na(inv))<nrow(inv)]
inv[, c(1:14)] <- sapply(inv[, c(1:14)],as.numeric)

ids <- select(inv, 1:14)
ids <- melt(ids, na.rm = TRUE, value.name = "ID")
id_list <- ids$ID

## Retrieve funder metadata via Europe PMC API

## Retrieve funder metadata
## Takes 10-15 minutes on several thousand

a  <- NULL;
for (i in id_list) {
  r <- sapply(i, epmc_details) 
  id <- r[[1]]["id"]
  title <- r[[1]]["title"]
  agency <- tryCatch(r[[9]]["agency"], error = function(cond) {
    message(paste("funder issue"))
    message(cond, sep="\n")
    return(NA)
    force(do.next)})
  report <- cbind(id, title, agency)
  a <- rbind(a, report)
}

## double check for any lost PMIDs 
a_id <- as.data.frame(a$id)
names(a_id)[1] ="id"
id_l <- as.data.frame(id_list)
names(id_l)[1] ="id"
a_id$id <- as.numeric(a_id$id)
id_l$id <- as.numeric(id_l$id)
lost <- anti_join(id_l, a_id)
lost_id <- lost$id

##===========================================================##
####### PART 2: Analyze funder metadata from Europe PMC ####### 
##===========================================================##

## number of articles that have funder metadata

hasFunderInfo <- a %>% 
  group_by(id) %>% 
      mutate(funderpresent = (test = ifelse((is.na(agency)),
                                  yes = "N",
                                  no = "Y")))

hasFunderInfo <- unique(select(hasFunderInfo, 1, 4))
sumhasFunderInfo <- sum(hasFunderInfo$funderpresent == "Y")

## isolate funders returned to tally number of unique papers and biodata resources
f <- a %>% filter(complete.cases(.))
f_ids <- unique(f$id)
f <- select(f, -2)
## get resource names to be able to associate with funders
sep <- separate_rows(inventory, ID, sep = ",")
names(sep)[1] ="id"
sep_ids <- unique(sep$id)
best_names <- select(sep, 1, 2)

##join
best_names$id <- trimws(best_names$id)
f$id <- trimws(f$id)
f_dbs <- left_join(f, best_names)

funders <- f_dbs %>%
  group_by(agency) %>%
    mutate(count_all_article_instances = length(id)) %>%
      mutate(count_unique_articles = length(unique(id))) %>%
        mutate(count_unique_biodata_resources = length(unique(best_name))) %>%
           mutate(associated_PMIDs = str_c(unique(id), collapse = ", ")) %>%
              mutate(associated_biodata_resources = str_c(unique(best_name), collapse = ", "))

## simplifying to just "unique" funders
funders <- unique(select(funders, 2,4:8)) 

##=====================================##
####### PART 3: Save output files ####### 
##=====================================##

write.csv(funders,"inventory_funders_2023-01-20.csv", row.names = FALSE)

