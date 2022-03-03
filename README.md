# inventory_2022 (Work in Progress)
Public repository for the biodata resource inventory performed in 2022.

## DATA DICTIONARIES

### Manual Curation

#### Variables for manual curation CSV files, e.g. manual_checks_all_YYYY-MM-DD.csv:

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

#### (likely temporary until we update the above file) For new variables within check classification conflicts CSV files, e.g. manual_checks_conflicts_2022-02-25.csv:

**FYI** 44/172 easily harmonized

* **index** = added index based manual_checks_all_2022-02-15.csv so can keep order as needed (esp needed if any chance of duplication of "id")
* **next steps** = "discuss" = discuss with kes, "kes review" = suggest for review (these seem like potentially clear/clearish mistakes; may be more based on the discussion for those that are less clear), or blank = no step (plan to merge as is into manual_checks_all_YYYY-MM-DD.csv - either now harmonized or so tricky that it should probably stay at 0.5 - that is, we shouldn't try to force agreement, could as Michaela's team to take a look though)

### Element Extraction

#### Variables for manual extraction of named entities *e.g.* extracted_elements_2022-02-15. This set only includes articles which had a **curation_score** of 1 as defined above.

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

* Multpile versions of an element, for instance when there are different **short_description**s in the title and abstract.
* Differences in case (*e.g.* "Human transporter database" vs "Human Transporter Database"). These are equivalent when case-insensitive, but case is deliberate in many titles.



In those cases, there will be multiple rows for the same article. For this reason, it would be best to either nest those fields (columns) with multiple entries, or to select columns of interest serially, and deduplicate.

In R this may look something like:

```
all_elements <- read.csv('extracted_elements_2022-02-15.csv')

names <- all_elements %>%
    select(id, title, abstract, name) %>%
    unique()

acronyms <- all_elements %>%
    select(id, title, abstract, acronym) %>%
    unique()
```

