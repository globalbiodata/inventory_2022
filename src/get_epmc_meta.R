#!/usr/bin/env Rscript

# Author : Kenneth Schackart <schackartk1@gmail.com>
# Date   : 2022-09-15
# Purpose: Get additional metadata from EuropePMC

# Imports -------------------------------------------------------------------

## Library calls ------------------------------------------------------------

library(magrittr)

# Argument Parsing ----------------------------------------------------------

#' Parse Arguments
#'
#' Parse command line arguments using argparse.
#'
#' @return args
get_args <- function() {
  parser <- argparse::ArgumentParser()

  parser$add_argument("file",
                      help = "Input file",
                      metavar = "FILE")
  parser$add_argument("-o",
                      "--out-dir",
                      help = "Output directory",
                      metavar = "DIR",
                      type = "character",
                      default = "out"
                      )

  args <- parser$parse_args()

  return(args)

}

# Main ----------------------------------------------------------------------

#' Main Function
#'
#' @return
main <- function() {

  args <- get_args()

  dir.create(args$out_dir, showWarnings = FALSE)
    
  in_df <- read.csv(args$file)
  out_df <- get_metadata(in_df)
}

# Functions

#' Get MetaData from EuropePMC
#' 
#' @param df Input dataframe
#' @return Dataframe with additional metadata
get_metadata <- function(df) {

  
}

# Call Main -----------------------------------------------------------------
if (!interactive()) {
  main()
}

