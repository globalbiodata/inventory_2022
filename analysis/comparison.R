## Purpose: Extract records for biodata resources from re3data and FAIRsharing APIs and compare with biodata resources found in GBC inventory 
## Parts: 1) Retrieve records from re3data.org 2) retrieve records from FAIRsharing, 3) compare with GBC inventory
## Package(s): httr, xml2, dplyr, tidyr, tidyverse, RCurl, jsonlite, data.table
## Input file(s): final_inventory_2022.csv, FAIRsharing login credential script
## Output file(s): inventory_re3data_FAIRsharing_2022-11-21.csv
## Notes for re3data.org:
## correct schema (2.2) is here: https://gfzpublic.gfz-potsdam.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_758898
## https://www.re3data.org/api/doc
## Scripts found at: https://github.com/re3data/using_the_re3data_API/blob/main/re3data_API_certification_by_type.ipynb
## Notes for FAIRsharing: 
## data is under CC-BY-SA - do not push any output files to Github! Run FAIRsharing login credential script first to obtain "hji_login" argument for the below. For rest, see API documentation on https://fairsharing.org/API_doc and https://api.fairsharing.org/model/database_schema.json

library(httr)
library(xml2)
library(dplyr)
library(tidyr)
library(tidyverse)
library(RCurl)
library(jsonlite)
library(data.table)

##=======================================================##
######### PART 1: Retrieve Records from re3data.org ####### 
##=======================================================##

re3data_request <- GET("http://re3data.org/api/v1/repositories")
re3data_IDs <- xml_text(xml_find_all(read_xml(re3data_request), xpath = "//id"))
URLs <- paste("https://www.re3data.org/api/v1/repository/", re3data_IDs, sep = "")

extract_repository_info <- function(url) {
  list(
    re3data_ID = xml_text(xml_find_all(repository_metadata_XML, "//r3d:re3data.orgIdentifier")),
    type = paste(unique(xml_text(xml_find_all(repository_metadata_XML, "//r3d:type"))), collapse = "_AND_"),
    repositoryURL = paste(unique(xml_text(xml_find_all(repository_metadata_XML, "//r3d:repositoryURL"))), collapse = "_AND_"),
    repositoryName = paste(unique(xml_text(xml_find_all(repository_metadata_XML, "//r3d:repositoryName"))), collapse = "_AND_"),
    subject = paste(unique(xml_text(xml_find_all(repository_metadata_XML, "//r3d:subject"))), collapse = "_AND_")
  )
}

repository_info <- data.frame(matrix(ncol = 12, nrow = 0))
colnames(repository_info) <- c("re3data_ID","repositoryName", "repositoryURL", "subject", "type")

for (url in URLs) {
  repository_metadata_request <- GET(url)
  repository_metadata_XML <-read_xml(repository_metadata_request) 
  results_list <- extract_repository_info(repository_metadata_XML)
  repository_info <- rbind(repository_info, results_list)
}

## Notes:
## filtering down to only life science (domain focused) -- many nat sci look to be life sci too, but not all - to stay consistent for the comparison with FAIRsharing, restricting to Life Science only

life_sci_re3data <- filter(repository_info, grepl("Life", subject))

## remove any strictly institutional as these are too general purpose and any that are "other" as most of these could not count for the GBC Inventory either (e.g. city data portals and generalist non-institutional repos like figshare)
life_sci_re3data <- filter(life_sci_re3data, life_sci_re3data$type != "institutional")
life_sci_re3data <- filter(life_sci_re3data, life_sci_re3data$type != "other")

life_sci_r3 <- life_sci_re3data

##=======================================================##
######### PART 2: Retrieve Records from FAIRsharing ####### 
##=======================================================##

## NOTE: run FAIRsharing login script to get hji_login argument

url<-'https://api.fairsharing.org/users/sign_in'
request <- POST(url,
                add_headers(
                  "Content-Type"="application/json",
                  "Accept"="application/json"),
                body=hji_login)
con <- jsonlite::fromJSON(rawToChar(request$content))
auth<-con$jwt

## just life science only
query_url<-"https://api.fairsharing.org/search/fairsharing_records?fairsharing_registry=database&subjects=life%20science&page[number]=1&page[size]=3600"

## note that this had a tendency to time out sometimes - just keep trying until works. Other days it seemed fine, so not sure if my connection or theirs.

get_res<-POST(
  query_url,
  add_headers(
    "Content-Type"="application/json",
    "Accept"="application/json",
    "Authorization"=paste0("Bearer ",auth,sep="")
  )
)

query_con <- fromJSON(rawToChar(get_res$content))

## get record info of interest

dbs1 <- as.data.frame(query_con[["data"]][["attributes"]][["metadata"]][["doi"]])
dbs2 <- as.data.frame(query_con[["data"]][["attributes"]][["metadata"]][["name"]])
dbs3 <- as.data.frame(query_con[["data"]][["attributes"]][["metadata"]][["homepage"]])
dbs4 <- as_tibble_col(query_con[["data"]][["attributes"]][["subjects"]])
dbs <- cbind(dbs1, dbs2, dbs3, dbs4)

##rename
names(dbs)[names(dbs)=="query_con[[\"data\"]][[\"attributes\"]][[\"metadata\"]][[\"doi\"]]"]<- "doi"
names(dbs)[names(dbs)=="query_con[[\"data\"]][[\"attributes\"]][[\"metadata\"]][[\"name\"]]"]<- "name"
names(dbs)[names(dbs)=="query_con[[\"data\"]][[\"attributes\"]][[\"metadata\"]][[\"homepage\"]]"]<- "url"
names(dbs)[names(dbs)=="value"]<- "subjects"

life_sci_fs <- apply(dbs,2,as.character)

##=============================================================##
######### PART 3: Compare with records with GBC Inventory ####### 
##=============================================================##

## start by cleaning data frames to prep for comparison

## re3data

life_sci_r3 <- select(life_sci_r3, 1, 4, 3)

## trim white space
life_sci_r3 %>% 
  mutate(across(where(is.character), str_trim))

## remove any blank urls
life_sci_r3 <- life_sci_r3[(which(nchar(life_sci_r3$repositoryURL) > 0)),]

## clean urls
life_sci_r3$repositoryURL <- sub("^http://(?:www[.])", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <- sub("^https://(?:www[.])", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <- sub("^http://", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <- sub("^https://", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <- sub("/$", "", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <- tolower(life_sci_r3$repositoryURL)

names(life_sci_r3)[1] <- "r3_id"
names(life_sci_r3)[2] <- "r3_name"
names(life_sci_r3)[3] <- "r3_url"
  
## FAIRsharing

life_sci_fs <- as.data.frame(life_sci_fs)
life_sci_fs <- select(life_sci_fs, 1, 2, 3)

## trim white space
life_sci_fs %>% 
  mutate(across(where(is.character), str_trim)) 

## remove any blank urls
life_sci_fs <- life_sci_fs[(which(nchar(life_sci_fs$url) > 0)),]

## clean urls
life_sci_fs$url <- sub("^http://(?:www[.])", "\\1", life_sci_fs$url)
life_sci_fs$url <- sub("^https://(?:www[.])", "\\1", life_sci_fs$url)
life_sci_fs$url <- sub("^http://", "\\1", life_sci_fs$url)
life_sci_fs$url <- sub("^https://", "\\1", life_sci_fs$url)
life_sci_fs$url <- sub("/$", "", life_sci_fs$url)
life_sci_fs$url <- tolower(life_sci_fs$url)

names(life_sci_fs)[1] <- "fs_id"
names(life_sci_fs)[2] <- "fs_name"
names(life_sci_fs)[3] <- "fs_url"

## inventory

inv <- read.csv("final_inventory_2022.csv") ## reminder - setwd to data folder
## select just id, best name and URL columns
inv <- select(inv, 1, 2, 9)

## note that 2 URLs extracted in inventory for ~5% of inventory resources - testing for matches on first URL only
inv <- separate(inv, 'extracted_url', paste("url", 1:2, sep="_"), sep=",", extra="drop")
inv <- select(inv, 1:3)

inv %>% 
  mutate(across(where(is.character), str_trim)) 

names(inv)[1] <- "inv_id"
names(inv)[2] <- "inv_name"
names(inv)[3] <- "inv_url_1"

inv$inv_url_1 <- sub("^http://(?:www[.])", "\\1", inv$inv_url_1)
inv$inv_url_1 <- sub("^https://(?:www[.])", "\\1", inv$inv_url_1)
inv$inv_url_1 <- sub("^http://", "\\1", inv$inv_url_1)
inv$inv_url_1 <- sub("^https://", "\\1", inv$inv_url_1)
inv$inv_url_1 <- sub("/$", "", inv$inv_url_1)
inv$inv_url_1 <- tolower(inv$inv_url_1)

## comparison

## inventory and re3data
same_name_inv_re3 <- inner_join(inv, life_sci_r3, by = c("inv_name" = "r3_name"))
same_url_inv_re3 <- inner_join(inv, life_sci_r3, by = c("inv_url_1" = "r3_url"))

unique_inv_re3 <- as.data.frame(unique(c(same_name_inv_re3$inv_name, same_url_inv_re3$inv_name)))
names(unique_inv_re3)[1] <- "unique_inv_re3"

## create table
summary <- NULL
summary$count_same_name_inv_re3 <- length(same_name_inv_re3$inv_id)
summary$count_same_url_inv_re3 <- length(same_url_inv_re3$inv_id)
summary$count_unique_inv_re3 <- length(unique_inv_re3$unique_inv_re3)

## inventory and FAIRsharing

same_name_inv_fs <- inner_join(inv, life_sci_fs, by = c("inv_name" = "fs_name"), keep = TRUE)
same_url_inv_fs <- inner_join(inv, life_sci_fs, by = c("inv_url_1" = "fs_url"))

unique_inv_fs <- as.data.frame(unique(c(same_name_inv_fs$inv_name, same_url_inv_fs$inv_name)))
names(unique_inv_fs)[1] <- "unique_inv_fs"

## add to table
summary$count_same_name_inv_fs <- length(same_name_inv_fs$inv_id)
summary$count_same_url_inv_fs <- length(same_url_inv_fs$inv_id)
summary$count_unique_inv_fs <- length(unique_inv_fs$unique_inv_fs)

## find unique names between re3data and fairsharing
total_unique <- as.data.frame(unique(unique(c(unique_inv_fs$unique_inv_fs, unique_inv_re3$unique_inv_re3))))
names(total_unique)[1] <- "names_unique_inv_re3_fs"

## add to table
summary$count_total_unique <- length(total_unique$names_unique_inv_re3_fs)
summary$percent <- ((summary$count_total_unique)/3112)*100

summary <- as.data.frame(summary)

## write.csv(summary,"inventory_re3data_FAIRsharing_2022-11-21.csv", row.names = FALSE)
