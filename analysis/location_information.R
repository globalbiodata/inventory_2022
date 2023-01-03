#!/usr/bin/env Rscript

# Author : Kenneth Schackart <schackartk1@gmail.com>
# Date   : 2022-12-27
# Purpose: Create plots of inventory location metadata

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

library(argparse)
library(dplyr)
library(ggmap)
library(ggplot2)
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

args <- get_args()

full_inventory <-
  read_csv(args$inventory_file,
           show_col_types = FALSE)

locations <- full_inventory %>%
  select(extracted_url_country,
         extracted_url_coordinates,
         affiliation_countries)

## URL locations ------------------------------------------------------------

print("Processing URL locations.")

### Coordinates -------------------------------------------------------------

print("Plotting URL coordinates.")

url_coordindates <- locations %>%
  select(extracted_url_coordinates) %>%
  rename(coordinates = extracted_url_coordinates) %>%
  na.omit() %>%
  mutate(coordinates = str_replace(coordinates, ",$", "")) %>%
  mutate(coordinates = strsplit(coordinates, ", ")) %>%
  unnest(coordinates) %>%
  mutate(coordinates = str_replace(coordinates, "\\(", "")) %>%
  mutate(coordinates = str_replace(coordinates, "\\)", "")) %>%
  filter(coordinates != "") %>%
  separate(coordinates, into = c("lat", "long"), sep = ",")

url_coordinate_plot <- url_coordindates %>%
  mutate_all(as.double) %>%
  ggplot(aes(long, lat)) +
  geom_map(
    data = map_data("world"),
    map = map_data("world"),
    aes(long, lat, map_id = region),
    color = "white",
    fill = "lightgray"
  ) +
  geom_point(
    alpha = 0.2,
    color = "#1b2a50",
    size = 1.5,
    shape = 16
  ) +
  theme_void()

ggsave("analysis/figures/ip_coordinates.png",
       url_coordinate_plot)

### Countries ---------------------------------------------------------------

print("Plotting URL countries.")

url_countries <- locations %>%
  select(extracted_url_country) %>%
  rename(country = extracted_url_country) %>%
  na.omit() %>%
  mutate(country = strsplit(country, ", ")) %>%
  unnest(country) %>%
  group_by(country) %>%
  summarize(count = n()) %>%
  filter(country != "Province of China") %>%
  mutate(
    country = case_when(
      country == "United States" ~ "USA",
      country == "United Kingdom" ~ "UK",
      country == "Korea" ~ "South Korea",
      country == "Russian Federation" ~ "Russia",
      country == "Czechia" ~ "Czech Republic",
      T ~ country
    )
  )

url_countries_joined <-
  left_join(map_data("world"), url_countries, by = c("region" = "country"))

url_country_plot <- ggplot() +
  geom_polygon(data = url_countries_joined, aes(
    x = long,
    y = lat,
    fill = count,
    group = group
  )) +
  theme_void() +
  labs(fill = "Count")

ggsave("analysis/figures/ip_countries.png",
       url_country_plot)

## Author locations ---------------------------------------------------------

print("Plotting author affiliation countries.")

author_country_counts <- locations %>%
  select(affiliation_countries) %>%
  na.omit() %>%
  mutate(affiliation_countries = strsplit(affiliation_countries, ", ")) %>%
  unnest(affiliation_countries) %>%
  rename(country = affiliation_countries) %>%
  group_by(country) %>%
  summarize(count = n()) %>%
  filter(country != "Province of China") %>%
  mutate(
    country = case_when(
      country == "United States" ~ "USA",
      country == "United Kingdom" ~ "UK",
      country == "Korea" ~ "South Korea",
      country == "Russian Federation" ~ "Russia",
      country == "Czechia" ~ "Czech Republic",
      T ~ country
    )
  )

author_countries_joined <-
  left_join(map_data("world"),
            author_country_counts,
            by = c("region" = "country"))

author_plot <- ggplot() +
  geom_polygon(data = author_countries_joined, aes(
    x = long,
    y = lat,
    fill = count,
    group = group
  )) +
  theme_void() +
  labs(fill = "Count")

ggsave("analysis/figures/author_countries.png",
       author_plot)

print("Done. Location data processed successfully.")