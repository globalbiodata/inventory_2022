## Purpose: Determine which articles associated with the biodata resource inventory are Open Access, have full text available, have text-mined terms, etc. 
## Parts: 1) Retrieve additional metadata from Europe PMC and 2) analyze metadata
## Package(s): europepmc, tidyverse, reshape2
## Input file(s): final_inventory_2022.csv
## Output file(s): text_mining_potential.csv

##================================================================##
####### PART 1: Retrieve additional metadata from Europe PMC ####### 
##================================================================##

library(europepmc)
library(tidyverse)
library(reshape2)

## split and melt IDs from the inventory into a list of just IDs 
inv <- read.csv("final_inventory_2022.csv") ## reminder - setwd to data folder
inv <- separate(inv, 'ID', paste("ID", 1:30, sep="_"), sep=",", extra="drop")
inv <- inv[,colSums(is.na(inv))<nrow(inv)]
inv[, c(1:14)] <- sapply(inv[, c(1:14)],as.numeric)

ids <- select(inv, 1:14)
ids <- melt(ids, na.rm = TRUE, value.name = "ID")
id_list <- ids$ID

## Retrieve Y/N metadata via Europe PMC API
## Note: number of returns should = number of IDs input
## Note: takes 10-15 minutes on several thousand IDs

y  <- NULL;
for (i in id_list) {
  r <- sapply(i, epmc_details) 
  id <- r[[1]]["id"]
  oa <- r[[1]]["isOpenAccess"]
  terms <- r[[1]]["hasTextMinedTerms"]
  acc_num <- r[[1]]["hasTMAccessionNumbers"]
  license <- tryCatch(r[[1]]["license"], error = function(cond) {
    message(paste("licence issue"))
    message(cond, sep="\n")
    return(NA)
    force(do.next)})
  report <- cbind(id, oa, terms, acc_num, license)
  y <- rbind(y, report)
}

## Retrieve full text metadata using comparison via original query restricted to HAS_FT:Y AND OPEN_ACCESS:Y

oa_ft <- '(ABSTRACT:(www OR http*) AND ABSTRACT:(data OR resource OR database*)) NOT (TITLE:(retract* OR withdraw* OR erratum)) NOT (ABSTRACT:(retract* OR withdraw* OR erratum OR github.* OR cran.r OR youtube.com OR bitbucket.org OR links.lww.com OR osf.io OR bioconductor.org OR annualreviews.org OR creativecommons.org OR sourceforge.net OR bit.ly OR zenodo OR onlinelibrary.wiley.com OR proteomecentral.proteomexchange.org/dataset OR oxfordjournals.org/nar/database OR figshare OR mendeley OR .pdf OR "clinical trial" OR registration OR "trial registration" OR clinicaltrial OR "registration number" OR pre-registration OR preregistration)) AND (SRC:(MED OR PMC OR AGR OR CBA)) AND (FIRST_PDATE:[2011 TO 2021]) AND ((HAS_FT:Y AND OPEN_ACCESS:Y))'

oa_ft_list <- epmc_search(query=oa_ft, limit = 25000)
oa_ft_list <- select(oa_ft_list, 1)

## this provides all FT/OA articles in original corpus, restrict now to those found in the inventory

id_list2 <- as.data.frame(id_list)
names(id_list2)[1] ="id"
id_list2$id <- as.character(id_list2$id)

found <- inner_join(id_list2, oa_ft_list, keep = TRUE) ## 2820 found
names(found)[1] ="inventory_ids"

not_found <- anti_join(id_list2, oa_ft_list, keep = TRUE) ## 915 not found
names(not_found)[1] ="inventory_ids"

##==============================================##
####### PART 2: Analyze  article metadata  ####### 
##==============================================##

## analyze FT availability
hasFT <- matrix(, nrow=1, ncol=1)
hasFT$Y <- length(found$inventory_ids)
hasFT$N <- length(not_found$inventory_ids)
hasFT$type <- "Full Text XML Available"
hasFT <- as.data.frame(hasFT)
hasFT <- select(hasFT, 4,2,3)

## analyze license availability
license <- as.data.frame(table(y['license'], useNA = "always"))
license$Freq <- as.numeric(license$Freq)
names(license)[1] ="Article_License"
names(license)[2] ="Count"

hasCC <- license %>% 
    mutate(total = sum(license$Count)) %>% 
        mutate(N = license$Count[[6]])

hasCC <- hasCC %>% 
          mutate(Y = hasCC$total - hasCC$N)

hasCC$type <- "CC Licensed"
hasCC <- select(hasCC, 6,5,4)
hasCC <- unique(hasCC)

## Y/N metadata 
isOpenAccess <- table(y['isOpenAccess'])
isOpenAccess$type <- "Open Access"
isOpenAccess <- as.data.frame(isOpenAccess)
isOpenAccess <- select(isOpenAccess, 3,2,1)

hasTextMinedTerm <- table(y['hasTextMinedTerms'])
hasTextMinedTerm$type <- "Text Mined Terms"
hasTextMinedTerm <- as.data.frame(hasTextMinedTerm)
hasTextMinedTerm <- select(hasTextMinedTerm, 3,2,1)

hasTMAccessionNumbers <- table(y['hasTMAccessionNumbers'])
hasTMAccessionNumbers$type <- "Text Mined Accession Numbers"
hasTMAccessionNumbers <- as.data.frame(hasTMAccessionNumbers)
hasTMAccessionNumbers <- select(hasTMAccessionNumbers, 3,2,1)

sum <- rbind (hasCC, isOpenAccess, hasFT, hasTextMinedTerm, hasTMAccessionNumbers)
sum <- as.data.frame(sum)
sum <- sum %>%
  mutate("percent" = (sum$Y/(sum$Y+sum$N))*100)

##=====================================##
####### PART 3: Save output files ####### 
##=====================================##

write.csv(sum,"text_mining_potential.csv", row.names = FALSE)

