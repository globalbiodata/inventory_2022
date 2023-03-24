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

library(dplyr)
library(glue)
library(httr)
library(jsonlite)
library(magrittr)
library(readr)
library(stringr)
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
  
  parser$add_argument(
    "inventory_file",
    help  = "Final inventory file",
    metavar = "FILE",
    type = "character",
    default = "data/final_inventory_2022.csv"
  )
  parser$add_argument(
    "-c",
    "--credentials",
    help  = "FAIRsharing login credentials file",
    metavar = "JSON",
    type = "character"
  )
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
    re3data_ID = xml_text(xml_find_all(
      metadata, "//r3d:re3data.orgIdentifier"
    )),
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

## clean_re3data ------------------------------------------------------------

#' Clean re3data fields
#'
#' @param df re3data repositories dataframe
#'
#' @return cleaned dataframe
clean_re3data <- function(df) {
  df %>%
    select(re3data_ID, repositoryName, repositoryURL) %>%
    rename("r3_id" = "re3data_ID",
           "r3_name" = "repositoryName",
           "r3_url" = "repositoryURL") %>%
    mutate(across(where(is.character), str_trim)) %>%
    drop_na(r3_url) %>%
    mutate(
      r3_url = str_remove(r3_url, "^https?://(www.)?"),
      r3_url = str_remove(r3_url, "/$"),
      r3_url = str_to_lower(r3_url)
    )
}

## clean_fairsharing -------------------------------------------------------

#' Clean FAIRsharing fields
#'
#' @param df FAIRsharing repositories dataframe
#'
#' @return cleaned dataframe
clean_fairsharing <- function(df) {
  df %>%
    select(doi, name, homepage) %>%
    rename("fs_id" = "doi",
           "fs_name" = "name",
           "fs_url" = "homepage") %>%
    mutate(across(where(is.character), str_trim)) %>%
    drop_na(fs_url) %>%
    mutate(
      fs_url = str_remove(fs_url, "^https?://(www.)?"),
      fs_url = str_remove(fs_url, "/$"),
      fs_url = str_to_lower(fs_url)
    )
}

## clean_inventory -------------------------------------------------------

#' Clean biodata inventory
#'
#' @param df Inventory dataframe
#'
#' @return cleaned dataframe
clean_inventory <- function(df) {
  ## note that 2 URLs extracted in inventory for ~5% of inventory resources
  ## - testing for matches on first URL only
  df %>%
    select(ID, best_name, best_common, best_full, extracted_url) %>%
    rename(
      "inv_id" = "ID",
      "inv_name" = "best_name",
      "inv_comm_name" = "best_common",
      "inv_full_name" = "best_full",
      "inv_url" = "extracted_url"
    ) %>%
    mutate(across(where(is.character), str_trim)) %>%
    mutate(
      inv_url = str_remove(inv_url, ",.*$"),
      inv_url = str_remove(inv_url, "^https?://(www.)?"),
      inv_url = str_remove(inv_url, "/$"),
      inv_url = str_to_lower(inv_url)
    )
}

# Main ----------------------------------------------------------------------

## Parse arguments ----------------------------------------------------------

args <- get_args()

credentials_file <- args$credentials

inventory <-
  read_csv(args$inventory_file,
           show_col_types = FALSE)

out_dir <- args$out_dir

## Query APIs ---------------------------------------------------------------

### re3data -----------------------------------------------------------------

re3data_return <- get_re3data_contents()

re3data_repos_all <- extract_re3data_info(re3data_return)

re3data_repos <- filter_re3data_contents(re3data_repos_all)

### FAIRsharing -------------------------------------------------------------

fairsharing_token <- login_fairsharing(credentials_file)

fairsharing_return <- get_fairsharing_contents(fairsharing_token)

fairsharing_repos <- extract_fairsharing_info(fairsharing_return)

## Clean data ---------------------------------------------------------------

re3data_cleaned <- clean_re3data(re3data_repos)

fairsharing_cleaned <- clean_fairsharing(fairsharing_repos)

inventory_cleaned <- clean_inventory(inventory)

## Analysis ------------------------------------------------------------------

summary <- tibble(
  inventory = logical(),
  re3data = logical(),
  fairsharing = logical(),
  names_shared = numeric(),
  urls_shared = numeric(),
  total_matches = numeric()
)

### inventory and re3data ----------------------------------------------------

same_comm_name_inv_re3 <-
  inner_join(inventory_cleaned,
             re3data_cleaned,
             by = c("inv_comm_name" = "r3_name"))
same_full_name_inv_re3 <-
  inner_join(inventory_cleaned,
             re3data_cleaned,
             by = c("inv_full_name" = "r3_name"))

same_name_inv_re3 <- tibble(
  names_found_in_re3 =
    c(
      same_comm_name_inv_re3$inv_comm_name,
      same_full_name_inv_re3$inv_full_name
    )
) %>%
  distinct(names_found_in_re3)

same_url_inv_re3 <-
  inner_join(inventory_cleaned, re3data_cleaned, by = c("inv_url" = "r3_url"))

unique_inv_re3 <- tibble(
  unique_inv_re3 = c(
    same_comm_name_inv_re3$inv_name,
    same_full_name_inv_re3$inv_name,
    same_url_inv_re3$inv_name
  )
) %>%
  distinct(unique_inv_re3)


res <- tibble(
  inventory = T,
  re3data = T,
  fairsharing = F,
  names_shared = nrow(same_name_inv_re3),
  urls_shared = nrow(same_url_inv_re3),
  total_matches = nrow(unique_inv_re3)
)

summary <- summary %>%
  rbind(res)

rm(same_comm_name_inv_re3,
   same_full_name_inv_re3,
   res)

### inventory and FAIRsharing ------------------------------------------------

same_comm_name_inv_fs <-
  inner_join(inventory_cleaned,
             fairsharing_cleaned,
             by = c("inv_comm_name" = "fs_name"))
same_full_name_inv_fs <-
  inner_join(inventory_cleaned,
             fairsharing_cleaned,
             by = c("inv_full_name" = "fs_name"))

same_name_inv_fs <- tibble(
  names_found_in_fs =
    c(
      same_comm_name_inv_fs$inv_comm_name,
      same_full_name_inv_fs$inv_full_name
    )
) %>%
  distinct(names_found_in_fs)

same_url_inv_fs <-
  inner_join(inventory_cleaned,
             fairsharing_cleaned,
             by = c("inv_url" = "fs_url"))

unique_inv_fs <- tibble(
  unique_inv_fs = c(
    same_comm_name_inv_fs$inv_name,
    same_full_name_inv_fs$inv_name,
    same_url_inv_fs$inv_name
  )
) %>%
  distinct(unique_inv_fs)

res <- tibble(
  inventory = T,
  re3data = F,
  fairsharing = T,
  names_shared = nrow(same_name_inv_fs),
  urls_shared = nrow(same_url_inv_fs),
  total_matches = nrow(unique_inv_fs)
)

summary <- summary %>%
  rbind(res)

rm(same_comm_name_inv_fs,
   same_full_name_inv_fs,
   res)

### re3data and FAIRsharing --------------------------------------------------

same_name_re3_fs <-
  inner_join(re3data_cleaned,
             fairsharing_cleaned,
             by = c("r3_name" = "fs_name"))

same_url_re3_fs <-
  inner_join(re3data_cleaned, fairsharing_cleaned, by = c("r3_url" = "fs_url"))

unique_re3_fs <- tibble(unique_re3_fs = c(same_name_re3_fs$r3_name,
                                          same_url_re3_fs$r3_name)) %>%
  distinct(unique_re3_fs)

res <- tibble(
  inventory = F,
  re3data = T,
  fairsharing = T,
  names_shared = nrow(same_name_re3_fs),
  urls_shared = nrow(same_url_re3_fs),
  total_matches = nrow(unique_re3_fs)
)

summary <- summary %>%
  rbind(res)

rm(res)

### inventory and re3data and FAIRsharing -----------------------------------

same_name_inv_re3_fs <-
  inner_join(
    same_name_inv_re3,
    same_name_inv_fs,
    by = c("names_found_in_re3" = "names_found_in_fs")
  )

same_url_inv_re3_fs <-
  inner_join(same_url_inv_re3,
             same_url_inv_fs, by = c("inv_url" = "inv_url")) %>%
  distinct(inv_url, .keep_all = T)

unique_inv_re3_fs <- tibble(
  unique_inv_re3_fs = c(
    same_name_inv_re3_fs$names_found_in_re3,
    same_url_inv_re3_fs$inv_name.x
  )
) %>%
  distinct(unique_inv_re3_fs)

res <- tibble(
  inventory = T,
  re3data = T,
  fairsharing = T,
  names_shared = nrow(same_name_inv_re3_fs),
  urls_shared = nrow(same_url_inv_re3_fs),
  total_matches = nrow(unique_inv_re3_fs)
)

summary <- summary %>%
  rbind(res)

rm(res)

### pivoting for venn diagram -----------------------------------------------

venn_df <- summary %>%
  select(-names_shared, -urls_shared) %>%
  mutate(
    combo = case_when(
      inventory & re3data & !fairsharing ~ "inv_re3",
      inventory & !re3data & fairsharing ~ "inv_fs",
      !inventory &
        re3data & fairsharing ~ "re3_fs",
      inventory & re3data & fairsharing ~ "inv_re3_fs",
    )
  ) %>%
  select(combo, total_matches) %>%
  pivot_wider(names_from = combo, values_from = total_matches) %>%
  mutate(
    inv_re3 = inv_re3 - inv_re3_fs,
    inv_fs = inv_fs - inv_re3_fs,
    re3_fs = re3_fs - inv_re3_fs,
    inv = nrow(inventory_cleaned) - (inv_re3 + inv_fs + inv_re3_fs),
    re3 = nrow(re3data_cleaned) - (inv_re3 + re3_fs + inv_re3_fs),
    fs = nrow(fairsharing_cleaned) - (inv_fs + re3_fs + inv_re3_fs)
  )

## Outputs ------------------------------------------------------------------

write_csv(summary,
          file.path(out_dir, "inventory_re3data_fairsharing_summary.csv"))
write_csv(venn_df, file.path(out_dir, "venn_diagram_sets.csv"))
