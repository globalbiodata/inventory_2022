#!/usr/bin/env Rscript

# Author : Heidi Imker <hjimker@gmail.com>
#          Kenneth Schackart <schackartk1@gmail.com>
# Date   : 2022-12-27
# Purpose: Determine which articles associated with the biodata resource
#          inventory are Open Access, have full text available,
#          have text-mined terms, etc.

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

library(argparse)
library(dplyr)
library(europepmc)
library(magrittr)
library(readr)
# library(reshape2)
library(stringr)
library(tidyr)

# Function Definitions ------------------------------------------------------

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
    "-q",
    "--query",
    help  = "Original query",
    metavar = "FILE",
    type = "character",
    default = "config/query.txt"
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

#' Get metadata from Europe PMC
#'
#' @param ids list of article IDs
#'
#' @return dataframe with article metadata
get_metadata <- function(ids) {
  out_df <- tibble()
  
  for (id_i in ids) {
    epmc_return <- epmc_details(id_i)
    metadata <- epmc_return[[1]]
    id <- metadata["id"]
    open_access <- metadata["isOpenAccess"]
    text_mined_terms <- metadata["hasTextMinedTerms"]
    tm_accession_nums <- metadata["hasTMAccessionNumbers"]
    license <- tryCatch(
      metadata["license"],
      error = function(cond) {
        return(NA)
        force(do.next)
      }
    )
    
    article_report <-
      cbind(id,
            open_access,
            text_mined_terms,
            tm_accession_nums,
            license)
    
    out_df <- rbind(out_df, article_report)
  }
  
  return(out_df)
}

#' Add dates to query and restrict to full text and open access
#'
#' @param query original query string
#'
#' @return dataframe with article metadata
modify_query <- function(query) {
  # Original query has place holders for date range, fill those in with years
  # then add restrictions to full text and open access
  query <- str_replace(query, "\\{0\\}", "2011") %>%
    str_replace("\\{1\\}", "2021") %>%
    paste("AND ((HAS_FT:Y AND OPEN_ACCESS:Y))")
  
  
  return(query)
}

#' Get IDs in inventory that are open access and full text
#'
#' @param inventory_ids IDs in inventory
#' @param oa_ft_ids IDs that are open access and full text
#'
#' @return dataframe with article metadata
get_oa_ft_inventory <- function(inventory_ids, oa_ft_ids) {
  inventory_oa_ft <- inner_join(inventory_ids, oa_ft_ids)
  
  return(inventory_oa_ft)
}

# Main ----------------------------------------------------------------------

print("Parsing command-line arguments.")

args <- get_args()

# query_string <- read_file(args$query)
query_string <- modify_query(read_file("config/query.txt"))

full_inventory <-
  # read_csv(args$inventory_file,
  read_csv("data/final_inventory_2022.csv",
           show_col_types = FALSE)

# out_dir <- args$out_dir
out_dir <- "analysis"

## Queries ------------------------------------------------------------------

### Metadata from original inventory ---------------------------------

long_inventory <- full_inventory %>%
  rename("id" = "ID") %>%
  mutate(resource_num = row_number()) %>%
  mutate(id = strsplit(id, ", ")) %>%
  unnest(id) %>%
  distinct(id, .keep_all = T)

cat("Getting metadata from Europe PMC... ")

metadata_df <- get_metadata(long_inventory$id)
long_inventory <- full_join(long_inventory, metadata_df)

cat("Done.\n")

### Open access and full text -----------------------------------------------

cat("Querying Europe PMC for articles with full text and open access... ")

open_full_ids <-
  select(epmc_search(query = query_string, limit = 25000), 1)

cat("Done.\n")

oa_ft_inventory <- get_oa_ft_inventory(id_list %>%
                                         rename("id" = "ID"), open_full_ids)

## Analysis -----------------------------------------------------------------

summary <- tibble(
  type = character(),
  resources_yes = numeric(),
  resources_no = numeric(),
  articles_yes = numeric(),
  articles_no = numeric()
)

### Full text availability --------------------------------------------------

articles_w_full_text <- nrow(oa_ft_inventory)
articles_wo_full_text <- nrow(id_list) - nrow(oa_ft_inventory)

summary <- summary %>%
  rbind(
    tibble(
      type = "Full Text XML Available",
      resources_yes = NA,
      resources_no = NA,
      articles_yes = articles_w_full_text,
      articles_no = articles_wo_full_text
    )
  )

rm(articles_w_full_text, articles_wo_full_text)


### License availability --------------------------------------------------

article_licenses <- metadata_df %>%
  select(license) %>%
  mutate(has_license = case_when(!is.na(license) ~ "yes",
                                 T ~ "no")) %>%
  group_by(has_license) %>%
  summarize(count = n())

articles_w_cc_license <- article_licenses %>%
  filter(has_license == "yes") %>%
  select(count)
articles_wo_cc_license <- article_licenses %>%
  filter(has_license == "no") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "CC licensed",
      resources_yes = NA,
      resources_no = NA,
      articles_yes = articles_w_cc_license,
      articles_no = articles_wo_cc_license
    )
  )

rm(article_licenses,
   articles_w_cc_license,
   articles_wo_cc_license)

### Open access -------------------------------------------------------------

article_access <- metadata_df %>%
  select(isOpenAccess) %>%
  group_by(isOpenAccess) %>%
  summarize(count = n())

open_access_articles <- article_access %>%
  filter(isOpenAccess == "Y") %>%
  select(count)
not_open_access_articles <- article_access %>%
  filter(isOpenAccess == "N") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "Open Access",
      resources_yes = NA,
      resources_no = NA,
      articles_yes = open_access_articles,
      articles_no = not_open_access_articles
    )
  )

rm(article_access,
   open_access_articles,
   not_open_access_articles)

### Text mined terms --------------------------------------------------------

text_mined_terms <- metadata_df %>%
  select(hasTextMinedTerms) %>%
  group_by(hasTextMinedTerms) %>%
  summarize(count = n())

has_text_mined_terms <- text_mined_terms %>%
  filter(hasTextMinedTerms == "Y") %>%
  select(count)
no_text_mined_terms <- text_mined_terms %>%
  filter(hasTextMinedTerms == "N") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "Text mined terms",
      resources_yes = NA,
      resources_no = NA,
      articles_yes = has_text_mined_terms,
      articles_no = no_text_mined_terms
    )
  )

rm(text_mined_terms, has_text_mined_terms, no_text_mined_terms)

### Text mined accession numbers --------------------------------------------

text_mined_acc_nums <- metadata_df %>%
  select(hasTMAccessionNumbers) %>%
  group_by(hasTMAccessionNumbers) %>%
  summarize(count = n())

has_text_mined_acc_nums <- text_mined_acc_nums %>%
  filter(hasTMAccessionNumbers == "Y") %>%
  select(count)
no_text_mined_acc_nums <- text_mined_acc_nums %>%
  filter(hasTMAccessionNumbers == "N") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "Text mined accession numbers",
      resources_yes = NA,
      resources_no = NA,
      articles_yes = has_text_mined_acc_nums,
      articles_no = no_text_mined_acc_nums
    )
  )

rm(text_mined_acc_nums,
   has_text_mined_acc_nums,
   no_text_mined_acc_nums)

### Summarization -----------------------------------------------------------

summary <- summary %>%
  mutate(articles_yes = as.numeric(articles_yes),
         articles_no = as.numeric(articles_no)) %>%
  mutate(article_percent = (articles_yes / (articles_yes + articles_no)) *
           100)

## Output -------------------------------------------------------------------

write_csv(summary, file.path(out_dir, "text_mining_potential.csv"))
