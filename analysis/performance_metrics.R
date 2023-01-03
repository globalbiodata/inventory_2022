#!/usr/bin/env Rscript

# Author : Kenneth Schackart <schackartk1@gmail.com>
# Date   : 2022-12-27
# Purpose: Create plots and tables of model performance metrics

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

library(argparse)
library(dplyr)
library(forcats)
library(ggplot2)
library(gt)
library(magrittr)
library(readr)
library(tidyr)

# Settings ------------------------------------------------------------------

theme_set(theme_light() +
            theme(
              plot.title = element_text(hjust = 0.5),
              plot.subtitle = element_text(hjust = 0.5)
            ))


# Function definitions ------------------------------------------------------

#' Parse command-line arguments
#'
#' @return args list with input filenames
get_args <- function() {
  parser <- argparse::ArgumentParser()
  
  parser$add_argument(
    "-cv",
    "--class-train",
    help  = "Classification train/val stats",
    metavar = "FILE",
    type = "character",
    default = "data/classif_metrics/combined_train_stats.csv"
  )
  parser$add_argument(
    "-ct",
    "--class-test",
    help  = "Classification test stats",
    metavar = "FILE",
    type = "character",
    default = "data/classif_metrics/combined_test_stats.csv"
  )
  parser$add_argument(
    "-nv",
    "--ner-train",
    help  = "NER train/val stats",
    metavar = "FILE",
    type = "character",
    default = "data/ner_metrics/combined_train_stats.csv"
  )
  parser$add_argument(
    "-nt",
    "--ner-test",
    help  = "NER test stats",
    metavar = "FILE",
    type = "character",
    default = "data/ner_metrics/combined_test_stats.csv"
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

#' Pivot metrics to longer
#'
#' @param df Wide-formatted dataframe of performance metrics
#' @return Input dataframe pivoted longer
pivot_metrics <- function(df) {
  df %>%
    pivot_longer(c(contains("train"), contains("val")),
                 names_to = "metric",
                 values_to = "value") %>%
    separate(metric, c("dataset", "metric"), "_") %>%
    pivot_wider(names_from = "metric", values_from = "value") %>%
    mutate(dataset = case_when(dataset == "val" ~ "Validation",
                               dataset == "train" ~ "Train"))
}

#' Add a new column with simplified model names based on HF model names
#'
#' @param df Dataframe with model_name column of HF model names
#' @return Same dataframe with new model column
relabel_models <- function(df) {
  df %>%
    mutate(
      model = case_when(
        model_name == "bert-base-uncased" ~ "BERT",
        model_name == "dmis-lab/biobert-v1.1" ~ "BioBERT",
        model_name == "kamalkraj/bioelectra-base-discriminator-pubmed" ~ "BioELECTRA",
        model_name == "kamalkraj/bioelectra-base-discriminator-pubmed-pmc" ~ "BioELECTRA-PMC",
        model_name == "allenai/biomed_roberta_base" ~ "BioMed-RoBERTa",
        model_name == "allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169" ~ "BioMed-RoBERTa-CP",
        model_name == "allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500" ~ "BioMed-RoBERTa-RCT",
        model_name == "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12" ~ "BlueBERT",
        model_name == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12" ~ "BlueBERT-MIMIC-III",
        model_name == "giacomomiolo/electramed_base_scivocab_1M" ~ "ELECTRAMed",
        model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" ~ "PubMedBERT",
        model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" ~ "PubMedBERT-Full",
        model_name == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" ~ "SapBERT",
        model_name == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token" ~ "SapBERT-Mean",
        model_name == "allenai/scibert_scivocab_uncased" ~ "SciBERT"
      )
    )
}

# Main ----------------------------------------------------------------------

print("Parsing command-line arguments.")

args <- get_args()

raw_classif_train_stats <-
  read_csv(args$class_train,
           show_col_types = FALSE)

raw_classif_test_stats <-
  read_csv(args$class_test,
           show_col_types = FALSE)

raw_ner_train_stats <-
  read_csv(args$ner_train,
           show_col_types = FALSE)

raw_ner_test_stats <-
  read_csv(args$ner_test,
           show_col_types = FALSE)

out_dir <- args$out_dir

## Plots --------------------------------------------------------------------

print("Generating plots.")

### Classification ----------------------------------------------------------

print("Plotting classification validation metrics.")

classif_train_stats <- raw_classif_train_stats %>%
  pivot_metrics() %>%
  relabel_models()

tidy_class_train_stats <- classif_train_stats %>%
  filter(dataset == "Validation") %>%
  group_by(model) %>%
  slice(which.max(precision)) %>%
  ungroup() %>%
  rename("Precision" = "precision",
         "Recall" = "recall",
         "F1-score" = "f1") %>%
  mutate(model = fct_reorder(model, Precision, .desc = TRUE)) %>%
  pivot_longer(
    names_to = "metric",
    values_to = "value",
    cols = c(Precision, Recall, "F1-score", loss),
  ) %>%
  mutate(metric = factor(metric, levels = c("Recall", "Precision", "F1-score"))) %>%
  filter(metric != "loss")

class_val_plot <- tidy_class_train_stats %>%
  ggplot(aes(y = metric, x = value)) +
  facet_wrap( ~ model, ncol = 3,) +
  geom_col(position = "dodge",
           alpha = 0.8,
           fill = "#29477e") +
  labs(x = "", y = "") +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
  theme(strip.background = element_rect(fill = "#454545"),
        axis.text.y = element_blank())

ggsave(
  file.path(out_dir, "class_val_set_performances.svg"),
  class_val_plot,
  width = 5,
  height = 6
)
ggsave(
  file.path(out_dir, "class_val_set_performances.png"),
  class_val_plot,
  width = 5,
  height = 6
)

### NER ---------------------------------------------------------------------

print("Plotting NER validation metrics.")

ner_train_stats <- raw_ner_train_stats %>%
  pivot_metrics() %>%
  relabel_models()

tidy_ner_train_stats <- ner_train_stats %>%
  filter(dataset == "Validation") %>%
  group_by(model) %>%
  slice(which.max(f1)) %>%
  ungroup() %>%
  rename("Precision" = "precision",
         "Recall" = "recall",
         "F1-score" = "f1") %>%
  mutate(model = fct_reorder(model, .[["F1-score"]], .desc = TRUE)) %>%
  pivot_longer(
    names_to = "metric",
    values_to = "value",
    cols = c(Precision, Recall, "F1-score", loss),
  ) %>%
  mutate(metric = factor(metric, levels = c("Recall", "Precision", "F1-score"))) %>%
  filter(metric != "loss")

ner_val_plot <- tidy_ner_train_stats %>%
  ggplot(aes(y = metric, x = value)) +
  facet_wrap( ~ model, ncol = 3,) +
  geom_col(position = "dodge",
           alpha = 0.8,
           fill = "#29477e") +
  labs(x = "", y = "") +
  theme(strip.text = element_text(color = "#1a1a1a"),
        axis.text.y = element_blank())

ggsave(
  file.path(out_dir, "ner_val_set_performances.svg"),
  ner_val_plot,
  width = 5,
  height = 6
)
ggsave(
  file.path(out_dir, "ner_val_set_performances.png"),
  ner_val_plot,
  width = 5,
  height = 6
)

## Tables -------------------------------------------------------------------

print("Generating metrics tables.")

### Classification ----------------------------------------------------------

print("Generating classification metrics table.")

classif_test_stats <- raw_classif_test_stats %>%
  rename(
    "model_name" = "model",
    "Precision" = "precision",
    "Recall" = "recall",
    "F1-score" = "f1"
  ) %>%
  relabel_models() %>%
  select(-model_name) %>%
  pivot_longer(
    names_to = "metric",
    values_to = "value",
    cols = c(Precision, Recall, "F1-score", loss),
  ) %>%
  mutate(metric = factor(metric, levels = c("Recall", "Precision", "F1-score")))

combined_class_table <- classif_test_stats %>%
  na.omit() %>%
  mutate(value = signif(value, 3)) %>%
  pivot_wider(names_from = "metric", values_from = "value") %>%
  rename(test_precision = Precision,
         test_recall = Recall,
         test_f1 = "F1-score") %>%
  left_join(
    tidy_class_train_stats %>%
      select(-model_name,-dataset,-epoch) %>%
      mutate(value = signif(value, 3)) %>%
      pivot_wider(names_from = "metric", values_from = "value") %>%
      rename(
        val_precision = Precision,
        val_recall = Recall,
        val_f1 = "F1-score"
      ),
    by = "model"
  ) %>%
  mutate(model = as.character(model)) %>%
  ungroup() %>%
  arrange(desc(val_precision)) %>%
  gt(rowname_col = "model") %>%
  tab_header(title = "Classification model performance on validation and test sets") %>%
  cols_move_to_start(columns = c(
    val_f1,
    val_precision,
    val_recall,
    test_f1,
    test_precision,
    test_recall
  )) %>%
  tab_spanner(label = "Validation Set",
              columns = c(val_f1, val_precision, val_recall)) %>%
  tab_spanner(label = "Test Set",
              columns = c(test_f1, test_precision, test_recall)) %>%
  cols_label(
    val_f1 = "F1-score",
    val_precision = "Precision",
    val_recall = "Recall",
    test_f1 = "F1-score",
    test_precision = "Precision",
    test_recall = "Recall"
  )


gtsave(combined_class_table,
       file.path(out_dir, "combined_classification_table.docx"))

### NER ---------------------------------------------------------------------

print("Generating NER metrics table.")

ner_test_stats <- raw_ner_test_stats %>%
  rename(
    "model_name" = "model",
    "Precision" = "precision",
    "Recall" = "recall",
    "F1-score" = "f1"
  ) %>%
  relabel_models() %>%
  select(-model_name) %>%
  pivot_longer(
    names_to = "metric",
    values_to = "value",
    cols = c(Precision, Recall, "F1-score", loss),
  ) %>%
  filter(metric != "loss") %>%
  mutate(metric = factor(metric, levels = c("Recall", "Precision", "F1-score")))

combined_ner_table <- ner_test_stats %>%
  na.omit() %>%
  mutate(value = signif(value, 3)) %>%
  pivot_wider(names_from = "metric", values_from = "value") %>%
  rename(test_precision = Precision,
         test_recall = Recall,
         test_f1 = "F1-score") %>%
  left_join(
    tidy_ner_train_stats %>%
      select(-model_name, -dataset, -epoch) %>%
      mutate(value = signif(value, 3)) %>%
      pivot_wider(names_from = "metric", values_from = "value") %>%
      rename(
        val_precision = Precision,
        val_recall = Recall,
        val_f1 = "F1-score"
      ),
    by = "model"
  ) %>%
  mutate(model = as.character(model)) %>%
  ungroup() %>%
  arrange(desc(val_f1)) %>%
  gt(rowname_col = "model") %>%
  tab_header(title = "NER model performance on validation and test sets") %>%
  cols_move_to_start(columns = c(
    val_f1,
    val_precision,
    val_recall,
    test_f1,
    test_precision,
    test_recall
  )) %>%
  tab_spanner(label = "Validation Set",
              columns = c(val_f1, val_precision, val_recall)) %>%
  tab_spanner(label = "Test Set",
              columns = c(test_f1, test_precision, test_recall)) %>%
  cols_label(
    val_f1 = "F1-score",
    val_precision = "Precision",
    val_recall = "Recall",
    test_f1 = "F1-score",
    test_precision = "Precision",
    test_recall = "Recall"
  )

gtsave(combined_ner_table,
       file.path(out_dir, "combined_ner_table.docx"))

print("Done. Analysis completed successfully.")
