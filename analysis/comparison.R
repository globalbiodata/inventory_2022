#!/usr/bin/env Rscript

# Author : Heidi Imker <hjimker@gmail.com>
#          Kenneth Schackart <schackartk1@gmail.com>
# Date   : 2023-01-19
# Purpose: Extract records for biodata resources from re3data and FAIRsharing
#          APIs and compare with biodata resources found in GBC inventory
# Notes  :
#     re3data.org: correct schema (2.2) is here:
#                  https://gfzpublic.gfz-potsdam.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_758898
#                  https://www.re3data.org/api/doc
#                  Scripts found at:
#                  https://github.com/re3data/using_the_re3data_API/blob/main/re3data_API_certification_by_type.ipynb
#     FAIRsharing: data is under CC-BY-SA Don't push any output files to Github!
#                  Run FAIRsharing login credential script first to obtain "hji_login" argument for the below.
#                  For rest, see API documentation on
#                  https://fairsharing.org/API_doc
#                  and
#                  https://api.fairsharing.org/model/database_schema.json

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

# library(data.table)
library(dplyr)
library(glue)
library(httr)
library(jsonlite)
library(magrittr)
# library(RCurl)
library(tibble)
library(tidyr)
library(xml2)

# Function Definitions ------------------------------------------------------

## get_args -----------------------------------------------------------------

#' Parse command-line arguments
#'
#' @return args list with input filenames
get_args <- function() {
  parser <- argparse::ArgumentParser()
  
  parser$add_argument("credentials",
                      help  = "FAIRsharing login credentials file",
                      metavar = "JSON",
                      type = "character")
  parser$add_argument(
    "-o",
    "--out-dir",
    help  = "Output directory",
    metavar = "DIR",
    type = "character",
    default = "analysis/figures"
  )
  
  args <- parser$parse_args()
  
  return(args)
}

## extract_repository_info ---------------------------------------------------

#' Extract re3data repository information
#'
#' @param metadata Repository metadata (XML)
#'
#' @return List of repository metadata
extract_repository_info <- function(metadata) {
  metadata_list <- list(
    re3data_ID = xml_text(
      xml_find_all(metadata, "//r3d:re3data.orgIdentifier")
    ),
    type = paste(unique(xml_text(
      xml_find_all(metadata, "//r3d:type")
    )), collapse = "_AND_"),
    repositoryURL = paste(unique(xml_text(
      xml_find_all(metadata, "//r3d:repositoryURL")
    )), collapse = "_AND_"),
    repositoryName = paste(unique(xml_text(
      xml_find_all(metadata, "//r3d:repositoryName")
    )), collapse = "_AND_"),
    subject = paste(unique(xml_text(
      xml_find_all(metadata, "//r3d:subject")
    )), collapse = "_AND_")
  )
  
  return(metadata_list)
}

## extract_re3data_info ----------------------------------------------------

#' Extract re3data return information
#'
#' @param re3data_return Return from re3data
#'
#' @return dataframe of re3data repositories
extract_re3data_info <- function(re3data_return) {
  repositories <- data.frame(matrix(ncol = 12, nrow = 0))
  colnames(repositories) <-
    c("re3data_ID",
      "repositoryName",
      "repositoryURL",
      "subject",
      "type")
  
  for (url in re3data_return) {
    repository_metadata_request <- GET(url)
    
    repository_metadata_XML <- read_xml(repository_metadata_request)
    
    results_list <- extract_repository_info(repository_metadata_XML)
    
    repositories <- rbind(repositories, results_list)
  }
  
  return(repositories)
}

## filter_re3data_contents --------------------------------------------------

#' Filter the contents of re3data return to only include life science
#' and not "institutional" or "other" to be consistent with FAIRsharing
#'
#' @param df re3data contents dataframe
#'
#' @return dataframe with only life science repositories
filter_re3data_contents <- function(df) {
  life_sci_re3data <- df %>%
    filter(grepl("Life", subject),
           type != "institutional",
           type != "other")
  
  return(life_sci_re3data)
}

## get_re3data_contents -----------------------------------------------------

#' Get contents of re3data
#'
#' @return list of urls from re3data query
get_re3data_contents <- function() {
  re3data_request <- GET("http://re3data.org/api/v1/repositories")
  re3data_IDs <-
    xml_text(xml_find_all(read_xml(re3data_request), xpath = "//id"))
  URLs <-
    paste("https://www.re3data.org/api/v1/repository/",
          re3data_IDs,
          sep = "")
  
  return(URLs)
}

## login_fairsharing --------------------------------------------------------

#' Login to FAIRsharing and get session token
#'
#' @param credentials_file FAIRsharing login credentials file
#'
#' @return session JSON web token to access API
login_fairsharing <- function(credentials_file) {
  fair_login_url <- 'https://api.fairsharing.org/users/sign_in'
  
  response <- POST(
    fair_login_url,
    add_headers("Content-Type" = "application/json",
                "Accept" = "application/json"),
    body = upload_file(credentials_file)
  )
  content <- fromJSON(rawToChar(response$content))
  token <- con$jwt
  
  return(token)
}

## extract_fairsharing_info -------------------------------------------------

#' Extract repository information from FAIRsharing return list
#'
#' @param fairsharing_return List of return from FAIRsharing
#'
#' @return dataframe of extracted information
extract_fairsharing_info <- function(fairsharing_return) {
  dois <-
    fairsharing_return[["data"]][["attributes"]][["metadata"]][["doi"]]
  names <-
    fairsharing_return[["data"]][["attributes"]][["metadata"]][["name"]]
  homepages <-
    fairsharing_return[["data"]][["attributes"]][["metadata"]][["homepage"]]
  subjects <-
    as_tibble_col(fairsharing_return[["data"]][["attributes"]][["subjects"]])
  
  fairsharing_repos <-
    tibble(
      "doi" = dois,
      "name" = names,
      "homepage" = homepages,
      "subjects" = subjects$value
    )
  
  return(fairsharing_repos)
}

## get_fairsharing_contents -------------------------------------------------

#' Get contents of FAIRsharing life science contents
#'
#' @note The request from FAIRsharing sometimes times out. Keep trying.
#'
#' @param token session JSON web token to access API
#'
#' @return session JSON web token to access API
get_fairsharing_contents <- function(token) {
  query_url <-
    paste0(
      "https://api.fairsharing.org/search/fairsharing_records?",
      "fairsharing_registry=database&subjects=life%20science",
      "&page[number]=1&page[size]=3600"
    )
  
  response <- POST(
    query_url,
    add_headers(
      "Content-Type" = "application/json",
      "Accept" = "application/json",
      "Authorization" = paste0("Bearer ", token)
    )
  )
  
  query_return <- fromJSON(rawToChar(response$content))
  
  return(query_return)
}

# Main ----------------------------------------------------------------------

args <- get_args()

credentials_file <- args$credentials

## Querying APIs ------------------------------------------------------------

### re3data -----------------------------------------------------------------

re3data_return <- get_re3data_contents()

re3data_repos_all <- extract_re3data_info(re3data_return)

re3data_repos <- filter_re3data_contents(re3data_repos_all)

### FAIRsharing -------------------------------------------------------------

fairsharing_token <- login_fairsharing(credentials_file)

fairsharing_return <- get_fairsharing_contents(fairsharing_token)

fairsharing_repos <- extract_fairsharing_info(fairsharing_return)


# Analysis ------------------------------------------------------------------

## start by cleaning data frames to prep for comparison

## re3data

life_sci_r3 <- select(life_sci_r3, 1, 4, 3)

## trim white space
life_sci_r3 %>%
  mutate(across(where(is.character), str_trim))

## remove any blank urls
life_sci_r3 <-
  life_sci_r3[(which(nchar(life_sci_r3$repositoryURL) > 0)),]

## clean urls
life_sci_r3$repositoryURL <-
  sub("^http://(?:www[.])", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <-
  sub("^https://(?:www[.])", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <-
  sub("^http://", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <-
  sub("^https://", "\\1", life_sci_r3$repositoryURL)
life_sci_r3$repositoryURL <-
  sub("/$", "", life_sci_r3$repositoryURL)
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
life_sci_fs$url <-
  sub("^https://(?:www[.])", "\\1", life_sci_fs$url)
life_sci_fs$url <- sub("^http://", "\\1", life_sci_fs$url)
life_sci_fs$url <- sub("^https://", "\\1", life_sci_fs$url)
life_sci_fs$url <- sub("/$", "", life_sci_fs$url)
life_sci_fs$url <- tolower(life_sci_fs$url)

names(life_sci_fs)[1] <- "fs_id"
names(life_sci_fs)[2] <- "fs_name"
names(life_sci_fs)[3] <- "fs_url"

## inventory

inv <-
  read.csv("final_inventory_2022.csv") ## reminder - setwd to data folder
## select just id, best name and URL columns
inv <- select(inv, 1, 2, 9)

## note that 2 URLs extracted in inventory for ~5% of inventory resources - testing for matches on first URL only
inv <-
  separate(inv,
           'extracted_url',
           paste("url", 1:2, sep = "_"),
           sep = ",",
           extra = "drop")
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
same_name_inv_re3 <-
  inner_join(inv, life_sci_r3, by = c("inv_name" = "r3_name"))
same_url_inv_re3 <-
  inner_join(inv, life_sci_r3, by = c("inv_url_1" = "r3_url"))

unique_inv_re3 <-
  as.data.frame(unique(c(
    same_name_inv_re3$inv_name, same_url_inv_re3$inv_name
  )))
names(unique_inv_re3)[1] <- "unique_inv_re3"

## create table
summary <- NULL
summary$count_same_name_inv_re3 <- length(same_name_inv_re3$inv_id)
summary$count_same_url_inv_re3 <- length(same_url_inv_re3$inv_id)
summary$count_unique_inv_re3 <-
  length(unique_inv_re3$unique_inv_re3)

## inventory and FAIRsharing

same_name_inv_fs <-
  inner_join(inv,
             life_sci_fs,
             by = c("inv_name" = "fs_name"),
             keep = TRUE)
same_url_inv_fs <-
  inner_join(inv, life_sci_fs, by = c("inv_url_1" = "fs_url"))

unique_inv_fs <-
  as.data.frame(unique(c(
    same_name_inv_fs$inv_name, same_url_inv_fs$inv_name
  )))
names(unique_inv_fs)[1] <- "unique_inv_fs"

## add to table
summary$count_same_name_inv_fs <- length(same_name_inv_fs$inv_id)
summary$count_same_url_inv_fs <- length(same_url_inv_fs$inv_id)
summary$count_unique_inv_fs <- length(unique_inv_fs$unique_inv_fs)

## find unique names between re3data and fairsharing
total_unique <-
  as.data.frame(unique(unique(
    c(unique_inv_fs$unique_inv_fs, unique_inv_re3$unique_inv_re3)
  )))
names(total_unique)[1] <- "names_unique_inv_re3_fs"

## add to table
summary$count_total_unique <-
  length(total_unique$names_unique_inv_re3_fs)
summary$percent <- ((summary$count_total_unique) / 3112) * 100

summary <- as.data.frame(summary)

## write.csv(summary,"inventory_re3data_FAIRsharing_2022-11-21.csv", row.names = FALSE)
