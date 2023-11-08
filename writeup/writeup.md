---
title: "Notes on recipient expenditure patterns"
author: Nathaniel Burbank
date: {cur_date} 
geometry: margin=2cm
classoption: table
documentclass: extarticle
urlcolor: blue
fontsize: 10pt
output: pdf_document
---

# Notes on recipient expenditure patterns
_Nathaniel Burbank, Central Data, {cur_date}_

## Key takeaways  

- Among GiveDirectly recipients, {most_popular_category_prct:.1%} report spending part or all of their transfer on {most_popular_category_lc}, making it the most commonly reported expense category.
- {prop_w_resp_in_top_5:.1%} of surveyed respondents indicated that they spent at least part of their transfer on one or more of {top_5_str} expenses.
- Only {vice_prct:.2%} of respondents, or slightly more than 1 in 1000 respondents, report spending any part of their transfer on a "vice activity" (alcohol, drugs, etc.). 
- The number of expense categories recipients typically select varies significantly by age and gender, with female and older recipients choosing more categories on average.
- After adjusting for the differing propensity to select multiple expense categories by demographic factors and the compositional differences across projects, there is evidence for a mild positive association between female gender and food spending, and a negative link between female gender and agricultural activities. Additionally, older recipients report spending more on food and healthcare and less on business activities.
- We have a [full breakdown](https://docs.google.com/spreadsheets/d/184txFsl9PnPbZav1lzhNk_th_r4OkiowhZepRBeaERQ/edit#gid=156678688) of both the raw response percentages and response percentages inverse-weighted by the number of categories selected, subdivided by country, project, recipient gender, and recipient age.

## Dataset construction 

This analysis is derived from follow-up surveys, which are conducted with most recipients after a transfer is completed. The dataset used for analysis was created by coalescing successful follow-up surveys from the research and follow-up objects that could be linked to completed transfers that occurred between October 2019 and October 2023.[^1] 

{prop_xfers_w_flup}% of transfers in the last four years have included a successfully completed follow-up survey.[^2] These follow-up surveys include two sets of questions that provide insight into how recipients spend their transfers. First, a multi-option pick list question known as the _What Did The Recipient Spend On_ or _Spending Categories_ question lets recipients select one or more from a set of approximately 60 predefined spending categories to indicate that at least part of their transfer was spent on the selected category or categories. And second, a series of quantitative questions that enable recipients to provide proportional estimates describing specifically how much of a given transfer was spent within a set of predefined categories. While answers to the second set of questions are more precise than the multi-option question, to date responses to the second set of questions have only been collected within a small number of projects.[^3] For this reason, this analysis is based on the less informative but more numerous answers to the Spending Categories question.

From the initial four year dataset, ineligible, discarded, written off, and refused recipients were filtered out.[^7] Additionally, projects that had fewer than 1000 completed followup surveys or a completion rate of less than 80% for the spending categories question were also removed. This resulted in a dataset of {N_obs}k responses from {N_rcp}k recipients in {N_proj} projects across {N_countries} countries.

###  Aggregating data from the spending categories question 

There are three primary challenges when working with data derived from the Spending Categories question. The first challenge is that there are multiple versions of this question stored in multiple locations. The most common variant has 48 distinct expense category options; a recently expanded version in the research object contains 64 distinct options.[^4] This analysis aggregates over both versions.

The second challenge is the large number of defined spending categories within these questions, some of which overlap. Of the 64 defined categories, only {N_cats_with_resp} were chosen by recipients in the last four years, with just {sel_by_over_1_prct} selected by over 1% of respondents. To identify broader trends, I consolidated these 64 categories into ten main groupings, which can be found in the [detailed_categories](https://docs.google.com/spreadsheets/d/184txFsl9PnPbZav1lzhNk_th_r4OkiowhZepRBeaERQ/edit#gid=572821971) tab of the project results spreadsheet.[^5] This grouping is necessarily somewhat arbitrary.

The third challenge is that the number of expense categories recipients typically choose varies significantly by age and gender. Over the past four years respondents selected {mean_number_of_cats:.2f} expense categories on average. However, there is significant variation in this count across demographic subgroups. Female recipients selected an average of {mean_number_of_cats_female:.2f} categories, or {mean_number_of_cats_gender_diff:.2f} more categories than the average male recipient who selected {mean_number_of_cats_male:.2f} categories. Moreover, older recipients, regardless of gender, selected more categories compared to younger recipients on average. See the [appendix](#more-on-the-relationship-between-demographic-factors-and-the-number-of-categories-selected) for more on these differences.

These two trends make it challenging to distinguish differences in spending patterns among demographic subgroups from differences in the inclination to provide a more or less detailed breakdown of primary expense categories. To make responses across recipients comparable I applied inverse weighting to the counts of response categories. I divide each response indicator by the total number of expense categories each recipient selected. For respondents who choose multiple categories, this normalization encodes the assumption that their transfer spending was split evenly among the categories they selected.[^6]

## Response-weighted aggregated expense categories 

The table below displays the 10 aggregated expense categories with three columns: 'N' represents the number of respondents who chose a category; 'Prct' shows the raw percentage of respondents selecting that category; and 'IW Prct' or inverse-weighted percent provides a percentage normalized based on the number of categories each respondent picked.

{top_aggregated_response_categories_mdtbl}

Among these higher level categories, food is the most popular reported expense category, with {prop_w_resp_in_top_5:.1%} of surveyed recipients indicating that they spent at least part of their transfer on one or more of {top_5_str} expenses. Food accounts for {most_popular_category_iw_prct:.1%} of transfer expenditures on an inverse-weighted basis, with the top five categories accounting for {prop_in_top_5_cats:.1%} overall. 

One potential concern arises from the fact that some programs have more transfers than others. When we normalize the proportions by the number of responses rather than the number of unique recipients, we might overweight the results towards spending patterns from those programs' recipients. To address this, I've included a version of this table in [the appendix](#recipient-weighted-aggregated-expense-categories) filtered to only include one response per recipient. However, the rank ordering of categories in that version is almost identical to the one above.

###  Top response-weighted expense categories

Aggregating categories is useful to see overall trends, but it is somewhat arbitrary. Here are the top {sel_by_over_1_prct} categories that were selected by over 1% of respondents with the same percent and inverse-weighted percent columns as in the table above.

{top_response_categories_by_response_mdtbl}

On this list, food and education expenses maintain their positions in the first and third most popular categories, respectively, while the order of other items is quite different. The full table of response weighted expense category counts and percentages can be found in the [detailed_categories](https://docs.google.com/spreadsheets/d/184txFsl9PnPbZav1lzhNk_th_r4OkiowhZepRBeaERQ/edit#gid=572821971) tab of the project results spreadsheet.


#### Spending on "vice" categories   

As noted above, a small percentage of respondents ({vice_prct:.2%}) reported spending at least part of their transfer on one or more "vice activities". These are defined as commercial sex, gambling, or alcohol/drugs. The large majority of respondents who reported spending within this group of categories selected the final category (alcohol or drugs). However, this likely underestimates the actual number of recipients who spend in these categories due to non-response bias. (Recipients are unlikely to self-report spending on categories that they perceive as controversial.) Within the aggregated analysis these categories are grouped under the larger "Other" category.

### Response-weighted aggregated expense categories by project and demographic characteristics

Using the higher-level expense categories, I calculated both the raw percentage of respondents who reported spending part or all of their transfer in each category and the inverse-weighted percentages across project, country, age, and gender subgroups. These tables can be found in the `by_proj`, `by_country`, etc., tabs within the [project results](https://docs.google.com/spreadsheets/d/184txFsl9PnPbZav1lzhNk_th_r4OkiowhZepRBeaERQ/edit#gid=1154303096) spreadsheet, with the `_iw` suffix indicating an inverse weighted version of the analysis.

## Preliminary analysis on associations between demographic characteristics and spending outcomes

We'd like to know whether there are associations between recipient demographic features and selected expense categories while adjusting for compositional differences across projects. Unfortunately, the number of response categories selected by recipients also varies by demographic factors. To address this, I modeled the inverse-weighted response counts instead of the raw counts, normalizing the differences in the number of category selections. Specifically, I fit a series of linear regression models, one for each expense outcome while controlling for project-specific effects. Each of these models examines the link between the probability of selecting a given expense category and the demographic characteristics of an "average" GD recipient.[^8] We can then compare coefficients across models to assess the relative strength of association between demographic factors and the selection of particular expense categories.[^9]

The following table contains the top five regression coefficients, standard errors, and p-values for gender and recipient age predictors fit within these 10 linear regression models, one for each higher level expense category outcome.

{lpm_by_predictors_sorted_mdtbl}

While this analysis is only preliminary, there appears to be a mild positive association between female gender and spending on food, age and spending on food, male gender and spending on agriculture, and age and spending on healthcare. Additionally, there is a negative association between entrepreneurship (or business activities) and advanced age. 

### Additional notes 

- This dataset is recipient rather than response weighted. Only the most recent followup survey from each recipient was included.
- Recipients whose gender was not recorded as  `Male` or `Female`,  or whose gender or age was unknown were dropped from this part of the analysis.
- The age predictor is standardized.

## Potential next steps 

If I were to spend more time on this analysis, these are the areas I would focus on next:  

- Examining if the proportions in the “How much spending” columns align with the inverse weighted percentages derived from the “Spending Categories” pick-list question. Though the sample is smaller, if the proportions closely resemble the inverse weighted percentages, it would provide further evidence that recipients (roughly) evenly distribute their expenses among the categories they select. This would further validate that the inverse weighted percentages are a reasonable proxy for inferring the final allocation of transfer dollars in aggregate. This would be the first step in enabling us to create dollarized spending allocation estimates (the next item).
- Presuming that the examination of the proportions in the “How much spending” columns does not reveal any surprises, we could then scale the inverse-weighted percentage estimates by the dollar amount of the associated transfers. This would allow us to estimate the number of  transferred dollars spent by recipients in specific categories in aggregate. In theory, we could then make statements like “X$ in Y time period were spent by recipients on food, or Z% of transfer dollars in 2022 were spent on housing.” (But again,  this is dependent on the Spending Categories” pick-list question proving to be a reasonable proxy for spending proportions.).
- Converting the above analysis into a Tableau dashboard that’s updated automatically and able to be filtered on more subcategories.
- More analysis and robustness checks are needed to assess the scale of the demographic associations between recipient characteristics and the stated expenditure patterns described above. What I have in this doc is a quick and dirty analysis only intended to identify potential patterns that might be worth investigating further. We would need to do more work to further verify these associations before relying on them for any public commentary.
- Could do unsupervised expense category clustering (via K-means clustering or Hierarchical clustering) to look for common groupings of expense categories and identify different recipient spending pattern profiles.

\newpage
## Appendix

### More on the relationship between demographic factors and the number of categories selected{#more-on-the-relationship-between-demographic-factors-and-the-number-of-categories-selected}

The plot below presents the mean number of expense categories chosen by recipients, subdivided by both gender and age. Each data point represents the average number of (ungrouped) categories selected by a specific gender within a specific age cohort. The error bars indicate two standard errors from the mean.

![Number of expense categories selected](output/cat_selected.png)

Both the age and gender differences in the number of categories selected are apparent: female recipients are more likely to select a larger number of expense categories than male recipients across all age groups. Additionally, older recipients are more likely to select a larger number of expense categories than younger recipients, regardless of gender.

A potential concern is that the observed variation in the number of expense categories selected across demographic characteristics could stem from a combination of survey design differences and demographic differences across projects rather than a true link between recipient demographics and their choice of categories. To address this, the plot below separates the average number of categories chosen by both project and gender. In this plot, each green dot shows the average number of distinct categories selected by male recipients in a specific project, while each blue dot represents female recipients. The size of the dots indicates the size of the recipient group for each gender within the project: larger dots signify more recipients of that gender.[^10] Additional dashed vertical lines display the average number of categories selected by gender across projects.[^11] 

![Number of expense categories selected by project](output/cat_selected_by_proj.png)

While there are substantial differences in the number of categories selected across projects, there remains a small but consistent difference within projects by recipient gender as well, with female recipients selecting more categories than their male counterparts on average in 26 out of 33 projects. Emergency response projects (clustered in the upper left) seem to have a smaller number of responses and smaller differences by gender. Conversely, gender differences are largest in Uganda and Rwanda's large transfer programs.

### Recipient-weighted aggregated expense categories{#recipient-weighted-aggregated-expense-categories}

One concern with the _Response-weighted aggregated expense categories_ table above is that we are over-weighting responses from recipients in programs that conduct a larger number of transfers. In the following table, I do the same analysis but only include the most recent successful follow-up survey for each recipient.

{top_aggregated_response_categories_rcp_mdtbl}

With the exception of the Entrepreneurship and Household categories, the order is the same as in the version that is weighted by the number of respondents in the main text.

### Regression specification for demographic factor association models{#regression-specification-for-demographic-factor-association}

In this analysis we essentially treat each outcome as a separate binary question and then assess the degree to which these two predictors are associated with that outcome after controlling for project-level fixed effects. This is one way of identifying association between gender and age above and beyond compositional differences resulting from different target populations in different projects.

Here's the regression specification I used for each of the models (one for each of the ten outcome categories):
$$
Y_{ij} = \beta_{0} + \beta_{1}X_{1i} + \beta_{2}X_{2i} + \sum_{k=1}^{K} \delta_k F_k   +  \epsilon_{i}
$$
Where: 

- $Y_{{i,j}}$ is the $I$th recipient's inverse weighted response count for the $j$th expense category (Food, housing, etc).
- $X_1$ is the $I$th recipient's gender.
- $X_2$ is the $I$th recipient's age.
- $F_k$ is an indicator variable for the $k$th project.

We're primarily interested in $\beta_1$ and $\beta_2$, the fit coefficients for gender and age.


[^1]: The base Databricks query is [here](https://github.com/nburbank-givedirectly/xfer-followup/blob/main/queries/base_query.sql). Survey records in the research object were linked to transfers via follow up records with identical recipient ids and similar timestamps. In rare cases transfers can be linked to more than one successfully completed followup survey. In these cases, only the most recent survey record was used.
[^2]: The numerator for [this proportion](https://github.com/nburbank-givedirectly/xfer-followup/blob/main/queries/prop_xfers_w_flup.sql) is the count of successful follow-up records that can be matched to a transfer id and have a `recipient_inactive_stage` of 0. The denominator is the count of all complete transfers with a creation date between October 1st, 2019 and September 30th, 2023.
[^3]: Most notably these questions were asked to recipients within the _Malawi Canva Basic Income Pilot_ and the _Liberia Maryland Basic Income_ programs.  
[^4]: In Field Salesforce, this is the `What_Did_The_Recipient_Spend_On` [question](https://givedirectly-field.my.salesforce.com/00N0b00000BuyeP?appLayout=setup&entityId=01I0b000001NFO0&noS1Redirect=true) within the Followup Object and the `Spending_Categories` [question](https://givedirectly-field.lightning.force.com/lightning/setup/ObjectManager/01I5a0000017dHL/FieldsAndRelationships/00N5a00000CsZNr/view) within the research object.
[^5]: The full mapping, including categories that were never used in the past four years, is defined [here](https://github.com/nburbank-givedirectly/xfer-followup/blob/22fac762e351be90505d6100118df603485aeec9/src/mappings.py#L337).
[^6]: Note that when tallying selections within the aggregated categories, multiple selections by a recipient within a single aggregated category are counted as a single response. When normalizing percentages across aggregated categories, response counts are inversely weighted by the number of aggregated categories selected, not the number of detailed categories.
[^7]: In other words,  `recipient_inactive_stage` is 0.
[^8]: The word "average" is doing a lot of work in this sentence. The point is, we want some sense of what the strongest associations are across countries and projects within the populations that we tend to serve. But the results of this type of observational modeling are always going to be heavily affected by, and downstream of, recipient targeting criteria and methods.
[^9]: Full regression specification is in [the appendix](#regression-specification-for-demographic-factor-association).
[^10]: Note that the _Kenya Urban Women COVID-19 Emergency Relief_ project and the _Kenya Buildher GIC Cash+_ project had no male recipients.
[^11]:  These averages are for the _ungrouped_ number of categories chosen.



