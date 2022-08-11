# Data directory

This directory contains the data used for model training, testing, and validation.

## DATA DICTIONARIES

### Manual Curation

#### Variables for manual classification csv file `manual_classifications.csv`

* **id** = article id
* **title** = article title
* **abstract** = article abstract
* **checked_by** = curator initials
* **kes_check** = kes determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **hji_check** = hji determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **curation_sum** = sum of curator values (iii_checks)
* **number_of_checks** = number of checks (by different curators)
* **curation_score** = curation_sum/number_of_checks (gives a "confidence score"" as done in Wren 2017); note that value other than 0 or 1 indiciate lack of agreement between curators
* **kes_notes** = raw notes documented by kes
* **hji_notes** = raw notes documented by hji

### Named Entity Extraction

#### Variables for manual extraction of named entities csv file `manual_ner_extraction.csv`

* **id** = article id
* **title** = article title. Adjacent articles were not included (*e.g.* "Protein Ensemble Database" not "The Protein Ensemble Database").
* **abstract** = article abstract
* **name** = resource name
* **acronym** = resource acronym or shortened name, as presented in the title or abstract. This is sometimes the same as **name**.
* **url** = resource URL. Note, other URL's may have been present that were not that of the resource. These extraneous URL's were not extracted into this column.
* **short_description** = short description of the resource, as found in the abstract or title

**Notes**:

Version numbers were generally not included in **name** or **acronym** if there was white space between the element and version number (*e.g.* "CTDB" was recorded for "CTDB (v2.0)" while version number in "PDB-2-PBv3.0" was kept).

Many articles had several of the above elements. This could be for a few reasons:

* Multiple versions of an element, for instance when there are different **short_description**s in the title and abstract.
* Differences in case (*e.g.* "Human transporter database" vs "Human Transporter Database"). These are equivalent when case-insensitive, but case is deliberate in many titles.



In those cases, there will be multiple rows for the same article. For this reason, it would be best to either nest those fields (columns) with multiple entries, or to select columns of interest serially, and deduplicate.

In R this may look something like:

```R
all_elements <- read.csv('extracted_elements_2022-02-15.csv')

names <- all_elements %>%
    select(id, title, abstract, name) %>%
    unique()

acronyms <- all_elements %>%
    select(id, title, abstract, acronym) %>%
    unique()
```