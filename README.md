# inventory_2022 (Work in Progress)
Public repository for the biodata resource inventory performed in 2022.

### DATA DICTIONARY

#### Variables for manual curation CSV files, e.g. manual_checks_all_YYYY-MM-DD.csv:

* **id** article id
* **title** article title
* **abstract** article abstract
* **checked_by** curator initials
* **kes_check** kes determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **hji_check** hji determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **curation_sum** sum of curator values (checks)
* **number_of_checks** number of checks (by different curators)
* **curation_score** curation_sum/number_of_checks (gives a "confidence score"" as done in Wren 2017)
* **kes_notes** notes documented by kes
* **hji_notes** notes documented by hji