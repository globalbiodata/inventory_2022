#!/usr/bin/env Rscript

# Author : Heidi Imker <hjimker@gmail.com>
#          Kenneth Schackart <schackartk1@gmail.com>
# Purpose: Retrieve and analyze funder metadata

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

library(argparse)
library(dplyr)
library(europepmc)
library(magrittr)
library(readr)
library(stringr)

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
    title <- metadata["title"]
    agency <- tryCatch(
      epmc_return[[9]]["agency"],
      error = function(cond) {
        return(NA)
        force(do.next)
      }
    )
    
    article_report <-
      cbind(id,
            title,
            agency)
    
    out_df <- rbind(out_df, article_report)
  }
  
  return(out_df)
}


# Main ----------------------------------------------------------------------

print("Parsing command-line arguments.")

args <- get_args()

full_inventory <-
  read_csv(args$inventory_file,
           show_col_types = FALSE)

out_dir <- args$out_dir

long_inventory <- full_inventory %>%
  rename("id" = "ID") %>%
  mutate(resource_num = row_number()) %>%
  mutate(id = strsplit(id, ", ")) %>%
  unnest(id) %>%
  distinct(id, .keep_all = T)

## Query Europe PMC ---------------------------------------------------------

cat("Getting metadata from Europe PMC... ")

id_list <- long_inventory$id

metadata_df <- get_metadata(id_list)

## Analyze funder metadata --------------------------------------------------

### Number of articles that have funder metadata ----------------------------

num_articles_w_funder_info <- metadata_df %>%
  group_by(id) %>%
  summarize(agencies = paste(agency, collapse = "")) %>%
  filter(agencies != "NA") %>%
  summarize(count = n())

### Analyze funders by resource ---------------------------------------------

funders <- long_inventory %>%
  select(id, best_name) %>%
  drop_na() %>%
  distinct(id, .keep_all = T) %>%
  right_join(metadata_df %>% drop_na()) %>%
  group_by(agency) %>%
  summarize(
    count_all_article_instances = length(id),
    count_unique_articles = length(unique(id)),
    count_unique_biodata_resources = length(unique(best_name)),
    associated_PMIDs = str_c(unique(id), collapse = ", "),
    associated_biodata_resources = str_c(unique(best_name), collapse = ", ")
  )

## Output -------------------------------------------------------------------

cat("Number of articles with funder information:",
    has_funder_info$count)
cat("Number of \"unique\" funders:", nrow(funders))
cat("Greatest # resources per funder:",
    max(funders$count_unique_biodata_resources))
cat("Average # resources per funder:",
    mean(funders$count_unique_biodata_resources))

write_csv(funders, file.path(out_dir, "inventory_funders.csv"))
