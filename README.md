# Recipient expenditure tracking analysis

Repo for the fall of 2023 recipient expenditure tracking analysis mini-project. Current version of analysis can be [found here](https://docs.google.com/document/d/15H2iSblt9dTKrPIReEqJcXH3xajhHYhN-1b7Y3IenBk/edit). 

## Steps to recreate analysis

1. Create a python environment with the required packages in the `requirements.txt` file.
2. Create a file named `db_secrets.py` in `src` folder, with a single `ACCESS_TOKEN` global variable with a valid token for Databricks. 
3. Execute `src/pipeline.py` from the root directory. 

This final step regenerates the base queries, pulls the generated queries from Databricks, runs the analysis, and generates the `xfer_flup.xlsx` and `xfer_flup.docx` files (among others) with the analysis and write-up in the output folder. Note that the query output and certain intermediate analysis will be automatically cached in the `data_cache` folder and will not be regenerated if present. (Delete these files to recreate analysis from scratch.)
