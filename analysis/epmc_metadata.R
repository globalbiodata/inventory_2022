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
library(ggplot2)
library(magrittr)
library(RColorBrewer)
library(readr)
library(scales)
library(stringr)
library(tidyr)

# Settings ------------------------------------------------------------------

theme_set(theme_light() +
            theme(
              plot.title = element_text(hjust = 0.5),
              plot.subtitle = element_text(hjust = 0.5)
            ))

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

query_string <- read_file(args$query)

full_inventory <-
  read_csv(args$inventory_file,
           show_col_types = FALSE)

out_dir <- args$out_dir

## Queries ------------------------------------------------------------------

### Metadata from original inventory ---------------------------------

long_inventory <- full_inventory %>%
  rename("id" = "ID") %>%
  mutate(resource_num = row_number()) %>%
  mutate(id = strsplit(id, ", ")) %>%
  unnest(id) %>%
  distinct(id, .keep_all = T)

cat("Getting metadata from Europe PMC... ")

id_list <- long_inventory$id

metadata_df <- get_metadata(id_list)

long_inventory <- full_join(long_inventory, metadata_df)

resource_metadata <- long_inventory %>%
  select(resource_num,
         license,
         isOpenAccess,
         hasTextMinedTerms,
         hasTMAccessionNumbers) %>%
  mutate(license = case_when(!is.na(license) ~ "cc",
                             T ~ "no")) %>%
  aggregate(. ~ resource_num, ., unique) %>%
  mutate(
    license = case_when(license == "cc" ~ "cc",
                        license == "no" ~ "no",
                        T ~ "both"),
    isOpenAccess = case_when(isOpenAccess == "Y" ~ "Y",
                             isOpenAccess == "N" ~ "N",
                             T ~ "both"),
    hasTextMinedTerms = case_when(
      hasTextMinedTerms == "Y" ~ "Y",
      hasTextMinedTerms == "N" ~ "N",
      T ~ "both"
    ),
    hasTMAccessionNumbers = case_when(
      hasTMAccessionNumbers == "Y" ~ "Y",
      hasTMAccessionNumbers == "N" ~ "N",
      T ~ "both"
    )
  )

cat("Done.\n")

### Open access and full text -----------------------------------------------

cat("Querying Europe PMC for articles with full text and open access... ")

open_full_ids <-
  select(epmc_search(query = query_string, limit = 25000), 1)

cat("Done.\n")

oa_ft_inventory <-
  get_oa_ft_inventory(long_inventory %>% select(id), open_full_ids)

## Analysis -----------------------------------------------------------------

summary <- tibble(
  type = character(),
  resources_yes = numeric(),
  resources_no = numeric(),
  resources_mixed = numeric(),
  articles_yes = numeric(),
  articles_no = numeric()
)

### Full text availability --------------------------------------------------

articles_w_full_text <- nrow(oa_ft_inventory)
articles_wo_full_text <- length(id_list) - nrow(oa_ft_inventory)

oa_ft_resources <- oa_ft_inventory %>%
  mutate(oa_ft = "true") %>%
  right_join(long_inventory) %>%
  distinct(id, .keep_all = T) %>%
  select(id, oa_ft, resource_num) %>%
  mutate(oa_ft = case_when(is.na(oa_ft) ~ "false", T ~ oa_ft)) %>%
  aggregate(. ~ resource_num, ., unique) %>%
  mutate(oa_ft = case_when(oa_ft == "true" ~ "true",
                           oa_ft == "false" ~ "false",
                           T ~ "both")) %>%
  group_by(oa_ft) %>%
  summarize(count = n())

ft_resources <- oa_ft_resources %>%
  filter(oa_ft == "true") %>%
  select(count)
not_ft_resources <- oa_ft_resources %>%
  filter(oa_ft == "false") %>%
  select(count)
mixed_ft_resources <- oa_ft_resources %>%
  filter(oa_ft == "both") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "Full Text XML Available",
      resources_yes = ft_resources$count,
      resources_no = not_ft_resources$count,
      resources_mixed = mixed_ft_resources$count,
      articles_yes = articles_w_full_text,
      articles_no = articles_wo_full_text
    )
  )

rm(
  articles_w_full_text,
  articles_wo_full_text,
  oa_ft_resources,
  ft_resources,
  not_ft_resources,
  mixed_ft_resources
)


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

resource_licenses <- resource_metadata %>%
  select(license) %>%
  group_by(license) %>%
  summarize(count = n())

resources_w_cc_license <- resource_licenses %>%
  filter(license == "cc") %>%
  select(count)
resources_wo_cc_license <- resource_licenses %>%
  filter(license == "no") %>%
  select(count)
resources_w_mixed_license <- resource_licenses %>%
  filter(license == "both") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "CC Licensed",
      resources_yes = resources_w_cc_license$count,
      resources_no = resources_wo_cc_license$count,
      resources_mixed = resources_w_mixed_license$count,
      articles_yes = articles_w_cc_license$count,
      articles_no = articles_wo_cc_license$count
    )
  )

rm(
  article_licenses,
  articles_w_cc_license,
  articles_wo_cc_license,
  resource_licenses,
  resources_w_cc_license,
  resources_wo_cc_license,
  resources_w_mixed_license
)

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

resource_access <- resource_metadata %>%
  select(isOpenAccess) %>%
  group_by(isOpenAccess) %>%
  summarize(count = n())

open_access_resources <- resource_access %>%
  filter(isOpenAccess == "Y") %>%
  select(count)
not_open_access_resources <- resource_access %>%
  filter(isOpenAccess == "N") %>%
  select(count)
mixed_access_resources <- resource_access %>%
  filter(isOpenAccess == "both") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "Open Access",
      resources_yes = open_access_resources$count,
      resources_no = not_open_access_resources$count,
      resources_mixed = mixed_access_resources$count,
      articles_yes = open_access_articles$count,
      articles_no = not_open_access_articles$count
    )
  )

rm(
  article_access,
  open_access_articles,
  not_open_access_articles,
  resource_access,
  open_access_resources,
  not_open_access_resources,
  mixed_access_resources
)

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

res_text_mined_terms <- resource_metadata %>%
  select(hasTextMinedTerms) %>%
  group_by(hasTextMinedTerms) %>%
  summarize(count = n())

res_has_text_mined_terms <- res_text_mined_terms %>%
  filter(hasTextMinedTerms == "Y") %>%
  select(count)
res_no_text_mined_terms <- res_text_mined_terms %>%
  filter(hasTextMinedTerms == "N") %>%
  select(count)
res_mixed_text_mined_terms <- res_text_mined_terms %>%
  filter(hasTextMinedTerms == "both") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "Text Mined Terms",
      resources_yes = res_has_text_mined_terms$count,
      resources_no = res_no_text_mined_terms$count,
      resources_mixed = res_mixed_text_mined_terms$count,
      articles_yes = has_text_mined_terms$count,
      articles_no = no_text_mined_terms$count
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

res_text_mined_acc_nums <- resource_metadata %>%
  select(hasTMAccessionNumbers) %>%
  group_by(hasTMAccessionNumbers) %>%
  summarize(count = n())

res_has_text_minedacc_nums <- res_text_mined_acc_nums %>%
  filter(hasTMAccessionNumbers == "Y") %>%
  select(count)
res_no_text_mined_acc_nums <- res_text_mined_acc_nums %>%
  filter(hasTMAccessionNumbers == "N") %>%
  select(count)
res_mixed_text_mined_acc_nums <- res_text_mined_acc_nums %>%
  filter(hasTMAccessionNumbers == "both") %>%
  select(count)

summary <- summary %>%
  rbind(
    tibble(
      type = "Text Mined Accession Numbers",
      resources_yes = res_has_text_minedacc_nums$count,
      resources_no = res_no_text_mined_acc_nums$count,
      resources_mixed = res_mixed_text_mined_acc_nums$count,
      articles_yes = has_text_mined_acc_nums$count,
      articles_no = no_text_mined_acc_nums$count
    )
  )

rm(
  text_mined_acc_nums,
  has_text_mined_acc_nums,
  no_text_mined_acc_nums,
  res_text_mined_acc_nums,
  res_has_text_mined_acc_nums,
  res_no_text_mined_acc_nums,
  res_mixed_text_mined_acc_nums
)

### Summarization -----------------------------------------------------------

summary <- summary %>%
  mutate(
    articles_yes = as.numeric(articles_yes),
    articles_no = as.numeric(articles_no),
    resources_yes = as.numeric(resources_yes),
    resources_no = as.numeric(resources_no),
    resources_mixed = as.numeric(resources_mixed)
  ) %>%
  mutate(
    articles_percent_yes = (articles_yes / (articles_yes + articles_no)) *
      100,
    articles_percent_no = (articles_no / (articles_yes + articles_no)) *
      100,
    resources_percent_yes = (resources_yes / (
      resources_yes + resources_no + resources_mixed
    )) * 100,
    resources_percent_no = (resources_no / (
      resources_yes + resources_no + resources_mixed
    )) * 100,
    resources_percent_mixed = (
      resources_mixed / (resources_yes + resources_no + resources_mixed)
    ) * 100
  )

summary_long <- summary %>%
  select(type, contains("percent")) %>%
  pivot_longer(cols = contains("percent"),
               names_to = "asset_label",
               values_to = "percent") %>%
  mutate(asset_label = str_remove(asset_label, "_percent")) %>%
  separate(asset_label, into = c("asset", "label")) %>%
  mutate(
    type = factor(
      type,
      levels = c(
        "Text Mined Accession Numbers",
        "Text Mined Terms",
        "Full Text XML Available",
        "Open Access",
        "CC Licensed"
      )
    ),
    asset = str_to_title(asset),
    label = str_to_title(label),
    label = factor(label, levels = c("No", "Mixed", "Yes"))
  )

### Visualization -----------------------------------------------------------

summary_plot <- summary_long %>%
  ggplot(aes(x = percent / 100, y = type, fill = label)) +
  facet_wrap( ~ asset) +
  geom_col(width = 0.5, alpha = 0.8) +
  scale_fill_manual(values = c("#D95F02", "#666666", "#7570B3")) +
  scale_x_continuous(labels = percent) +
  labs(x = "",
       y = "",
       fill = "") +
  guides(fill = guide_legend(reverse = T)) +
  theme(legend.position = "bottom")

summary_plot

## Output -------------------------------------------------------------------

write_csv(summary, file.path(out_dir, "text_mining_potential.csv"))

ggsave(
  file.path(out_dir, "text_mining_potential_plot.png"),
  summary_plot,
  width = 6.5,
  height = 4
)
ggsave(
  file.path(out_dir, "text_mining_potential_plot.svg"),
  summary_plot,
  width = 6.5,
  height = 4
)
