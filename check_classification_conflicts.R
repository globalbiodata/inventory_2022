## Purpose: CHECK MANUAL CLASSIFICATION CONFLICTS B/T KES and HJI 
## Input files: manual_checks_all_2022-02-15.csv
## Output files: manual_checks_conflicts_2022-02-25.csv

library(tidyverse)

checks <- read.csv("manual_checks_all_2022-02-15.csv")
checks$index <- 1:nrow(checks) ## in case there are duplicates w/ diff scores
conflict <- filter(checks, checks$curation_score == 0.5)
## FYI current percent agreement = 89% = (172/1634) - 1
write.csv(conflict,"manual_checks_conflicts_2022-02-25.csv", row.names = FALSE) 

## did manual review in excel and saved over as manual_checks_conflicts_2022-02-25.csv

conflicts_hji_checked <- read.csv("manual_checks_conflicts_2022-02-25.csv")
total_changed <- conflicts_hji_checked %>% filter(curation_score == 1 | curation_score == 0)
total_to_discuss <- conflicts_hji_checked %>% filter(next_step == "discuss")
total_for_kes_review <- conflicts_hji_checked %>% filter(next_step == "kes review")






