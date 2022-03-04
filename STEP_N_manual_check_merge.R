## Processing of manual curation checks
## Parts: 1) merge kes and hji (Dec 2021/Jan 2022) and 2) merge hji summer 2021 with 1)
## Package(s): tidyverse
## Input file(s): manual_checks_kes_2021-12-17.csv, manual_checks_hji_2022-01-10.csv
## Output file(s): manual_checks_kes_hji_2022-01-10.csv (intermediate), manual_checks_all_2022-01-10.csv
## Notes: once kes and hji merged here, opened in excel to determine variables to add for
## for summing, scoring, and curators, etc. Then row bound in hji_summer once that was set.

library(tidyverse)

kes_checks <- read.csv("manual_checks_kes_2021-12-17.csv")
hji_checks <- read.csv("manual_checks_hji_2022-01-10.csv")

##check what won't join first
lost <- anti_join(kes_checks, hji_checks, by = "id") ## 7 not in hji; not scored in kes, import errors

kes_hji <- right_join(kes_checks, hji_checks, by = "id") 
write.csv(kes_hji,"manual_checks_kes_hji_2022-01-10.csv", row.names = FALSE) 

## post excel (see Note above)
kes_hji2 <- read.csv("manual_checks_kes_hji_2022-01-10.csv")

##reformat hji_summer to prep for merge with kes_hji2
hji_summer <- read.csv("manual_checks_hji_2021-08-20.csv")

##rename columns to match
names(hji_summer)[names(hji_summer)=="abstractText"]<-"abstract"
names(hji_summer)[names(hji_summer)=="hji_recheck"]<-"hji_check"
names(hji_summer)[names(hji_summer)=="NOTES"]<-"hji_notes"

##select necessary columns
hji_summer <- select(hji_summer, id, title, hji_check, abstract, hji_notes)

##harmonize types
hji_summer$id <- as.character(hji_summer$id)

##add columns
hji_summer$checked_by <- as.character("hji")
hji_summer$curation_sum <- hji_summer$hji_check
hji_summer$number_of_checks <- 1
hji_summer$curation_score <- hji_summer$curation_sum/hji_summer$number_of_checks

all_checks <- bind_rows(hji_summer, kes_hji2)
names(all_checks)
all_checks <- select(all_checks, 1,2,4,6,10,3,7,8,9,11,5)

## noticed now that a few duplicates snuck in earlier
all_checks <- distinct(all_checks)

write.csv(all_checks,"manual_checks_all_2022-01-10.csv", row.names = FALSE) 

## just double checking that records returned from query for kes and hji curation Dec/Jan
all <- read.csv("pmc_seed_all_2021-08-06.csv")
check <- kes_hji2$id[!kes_hji2$id %in% all$id]
check2 <- all$id[!all$id %in% kes_hji2$id]



