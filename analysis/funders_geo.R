#!/usr/bin/env Rscript

# Author : Heidi Imker <hjimker@gmail.com>
#          Kenneth Schackart <schackartk1@gmail.com>
# Purpose: Analyze funders by country w/ associated agency and biodata resource names and counts

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

library(argparse)
library(dplyr)
library(ggmap)
library(ggplot2)
library(magrittr)
library(maps)
library(purrr)
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
    "curated_funders",
    help  = "Manually curated output from funders.R",
    metavar = "FILE",
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

# Main ----------------------------------------------------------------------

print("Parsing command-line arguments.")

args <- get_args()

# funders <-
#   read_csv(args$curated_funders,
#            show_col_types = FALSE)
funders <-
  read_csv("analysis/funders_geo_200.csv",
           show_col_types = FALSE)

out_dir <- args$out_dir

## Analysis -----------------------------------------------------------------

funders_by_country <- funders %>%
  select(agency,
         country,
         country_3,
         known_parent,
         associated_biodata_resources) %>%
  mutate(associated_biodata_resources = gsub('[\" ]', '', associated_biodata_resources)) %>%
  group_by(country, country_3) %>%
  summarize(
    count_agencies = length(agency),
    agency_names = str_c(agency, collapse = ", "),
    resource_names = str_c(associated_biodata_resources, collapse = ",")
  ) %>%
  group_by(country) %>%
  mutate(
    names_split = strsplit(resource_names, ","),
    unique_names_split = map(names_split, ~ unique(.x)),
    count_resources = length(unlist(unique_names_split)),
    biodata_resource_names = str_c(flatten(unique_names_split), collapse = ", ")
  ) %>%
  select(-resource_names,-names_split,-unique_names_split)

## Plotting -----------------------------------------------------------------

countries_plotting <- funders_by_country %>% 
  select(country, country_3, count_resources) %>% 
  mutate(
    country = case_when(
      country == "US" ~ "USA",
      country == "United Kingdom" ~ "UK",
      country == "Korea" ~ "South Korea",
      T ~ country
    )
  ) %>%
  right_join(map_data("world"), by = c("country" = "region"))


country_plot <- ggplot() +
  geom_polygon(data = countries_plotting, aes(
    x = long,
    y = lat,
    fill = count_resources,
    group = group
  )) +
  theme_void() +
  labs(fill = "Count")

## Output -------------------------------------------------------------------

write_csv(funders_by_country,
          file.path(out_dir, "funders_geo_counts.csv"))

ggsave(file.path(out_dir, "funder_countries.png"),
       country_plot, height = 4, width = 6.5)
