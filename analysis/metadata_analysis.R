#!/usr/bin/env Rscript

# Author : Kenneth Schackart <schackartk1@gmail.com>
# Date   : 2022-12-27
# Purpose: Perform simple analyses on final inventory metadata

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

library(argparse)
library(dplyr)
library(magrittr)
library(readr)
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
    type = "character"
  )
  
  args <- parser$parse_args()
  
  return(args)
}

# Main ----------------------------------------------------------------------

print("Parsing command-line arguments.")

#args <- get_args()

full_inventory <-
  read_csv("out/original_query/processed_countries/predictions.csv",
           show_col_types = FALSE)
# read_csv(args$inventory_file,
#          show_col_types = FALSE)

## Articles -----------------------------------------------------------------

num_articles <- full_inventory %>% 
  mutate(ID = strsplit(ID, ", ")) %>%
  unnest(ID) %>%
  distinct(ID) %>%
  count()

print(paste("Number of unique articles: ", num_articles))

## URLs ---------------------------------------------------------------------

### URL statuses ------------------------------------------------------------

num_resources_with_good_url <- full_inventory %>%
  mutate(extracted_url_status = strsplit(extracted_url_status, ", ")) %>%
  unnest(extracted_url_status) %>%
  filter(str_detect(extracted_url_status, "^[23]")) %>%
  distinct(ID) %>%
  count()

print(
  paste(
    "Number of resources with at least 1 URL returning 2XX or 3XX:",
    num_resources_with_good_url
  )
)

### WayBack URLs ------------------------------------------------------------

num_resources_with_wayback <- full_inventory %>%
  mutate(wayback_url = strsplit(wayback_url, ", ")) %>%
  unnest(wayback_url) %>%
  filter(wayback_url != "no_wayback") %>%
  distinct(ID) %>%
  count()

print(
  paste(
    "Number of resources with at least 1 WayBack URL:",
    num_resources_with_wayback
  )
)

## Funding ------------------------------------------------------------------

num_with_grant_agency <- full_inventory %>% 
  drop_na(grant_agencies) %>% 
  count()

print(
  paste(
    "Number of resources with grant agency data:",
    num_with_grant_agency
  )
)
