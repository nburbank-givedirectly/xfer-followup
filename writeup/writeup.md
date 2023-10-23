---
title: "Recipient expenditure patterns"
author: Nathaniel Burbank
date: {cur_date} 
geometry: margin=2cm
classoption: table
documentclass: extarticle
urlcolor: blue
fontsize: 10pt
output: pdf_document
---



# Key takeaways  

- Among Give Directly's recipients, {most_popular_category_prct:.1%} report spending part or all of their transfer on {most_popular_category}, making it the most common reported expense category.
- {prop_w_resp_in_top_5:.1%} of surveyed recipients indicated that they spent at least part of their transfer on one or more of {top_5_str} expenses.
- Only {vice_prct:.2%} of respondents, or slightly more than 1 in 1000 respondents, report spending any part of their transfer on a "vice activity" (Alcohol, drugs, etc.). 
- There is substantial variation in response patterns [by project, country, age, and gender](https://docs.google.com/spreadsheets/d/1PuFqGpOftwJiY92f6HUSA-15MTaYTjHdsF1J_lvc0sU/edit?usp=sharing).
- There is significant variation in the number of expense categories selected in the Spending Categories question by both age and gender, with female and older recipients choosing more categories on average.
- After accounting for compositional differences across projects and adjusting for aforementioned response bias, there is evidence for a mild positive association between female gender and spending on food, and a negative association with agriculture activities. Additionally, older recipients are more likely to report spending on food and healthcare, less likely to report spending on business activities.

## Dataset construction 

This analysis is derived from follow-up surveys, which are conducted with most recipients after a transfer is completed. The dataset used for analysis was created by coalescing successful follow-up surveys from the research and follow-up objects that could be linked to completed transfers that occurred between October 2019 and October 2023.[^1] 

{prop_xfers_w_flup}% of transfers in the last four years have included a successfully completed follow-up survey.[^2] These follow-up surveys include two sets of questions that provide insight into how recipients spend their transfers. First, a multi-option pick list question known as the _What Did The Recipient Spend On_ or _Spending Categories_ question lets recipients select one or more from a set of approximately 60 predefined spending categories to indicate that at least part of their transfer was spent on the selected category or categories. And second, a series of quantitative questions that enable recipients to provide proportional estimates describing specifically how much of a given transfer was spent within a set of predefined categories. While answers to the second set of questions are more precise than the multi-option question, to date responses to the second set of questions have only been collected within a small number of projects.[^3] For this reason, this analysis is based on the less informative but more numerous answers to the _Spending Categories_ question.

From the initial four year dataset, ineligible, discarded, written off, and refused recipients were filtered out. Additionally, projects that had fewer than 1000 completed followup surveys or a completion rate of less than 80% for the spending categories question were also removed. This resulted in a dataset of {N_obs}k responses from {N_rcp}k recipients in {N_proj} projects across {N_countries} countries.

####  Spending categories question 

In the _What Did The Recipient Spend On_ or _Spending Categories_ question, recipients choose from a list of pre-defined expense categories to indicate how they used their transfer. The order in which the categories are selected is not recorded. This analysis aggregates over multiple variations of this question that have been used historically.[^4] The most common variant has 48 distinct options; a recently expanded variant in the research object contains 64 distinct options. However, only {N_cats_with_resp} of these categories were ever selected by recipients over the past four years, and only 16 categories were selected by more than 1% of respondents.

Over the past four years respondents selected {mean_number_of_cats} expense categories on average. However, there is significant variation in this count across demographic subgroups. Female recipients selected an average of {mean_number_of_cats_female} categories, which is {mean_number_of_cats_gender_diff} more categories than the average male recipient who selected {mean_number_of_cats_male} categories. Moreover, older recipients, regardless of gender, selected more categories compared to younger recipients on average. These two trends make it challenging to distinguish differences in spending patterns among demographic subgroups from differences in the inclination to provide a more or less detailed breakdown of primary expense categories. See the appendix for more on these differences.

In order to observe higher level trends, it's necessarily to aggregate the pick list categories into higher level groupings.  I created ten higher level groupings, defined [here](https://github.com/nburbank-givedirectly/xfer-followup/blob/22fac762e351be90505d6100118df603485aeec9/src/mappings.py#L337). This aggregation is necessarily somewhat arbitrary. 

## Top aggregated expense categories by response 

The table below displays the 10 aggregated expense categories with three columns: 'N' represents the number of respondents who chose a category; 'Percent' shows the raw percentage of respondents selecting that category; and 'Inverse Weighted Percent' provides a percentage normalized based on the number of categories each respondent picked, presuming that a respondent's expenses were evenly distributed across their chosen categories.

{top_aggregated_response_categories_mdtbl}

Among these higher level categories, food is the most popular reported expense category, with {prop_w_resp_in_top_5:.1%} of surveyed recipients indicated that they spent at least part of their transfer on one or more of {top_5_str} expenses. Food accounts for {most_popular_category_iw_prct:.1%} of transfer expenditures on an inverse-weighted basis, with the top five categories accounting for {prop_in_top_5_cats:.1%} overall. 

### Top aggregated expense categories by recipient 

One concern with the table above is that we are over-weighting responses from recipients in programs that conduct a larger number of transfers. In the following table we conduct the same analysis but only include the most recent successful follow-up survey for each recipient.

{top_aggregated_response_categories_rcp_mdtbl}

With the exception of the Entrepreneurship and household categories, is the same is in the version that is normalized by respondents.

###  Top expense categories by recipient

Finally, aggregating categories is useful to see overall trends, but it's somewhat arbitrary. Here are the top categories in their original form.

{top_response_categories_by_response_mdtbl}


#### Spending on "vice" categories   

As noted above, a small percentage of respondents ({vice_prct:.1%}) reported spending at least part of their transfer on one or more "vice activities". These are defined as commercial sex, gambling, or alcohol/drugs. The large majority of respondents who reported spending within this group of categories selected the final category (alcohol or drugs). However, this likely underestimates the actual number of recipients who spend in these categories due to non-response bias. (Recipients are unlikely to self-report spending on categories that they perceive as controversial.) Within the aggregated analysis these categories are bundled under the larger "other" category.

### Variation by project and demographic characteristics

Using the aggregated response categories, we can calculate the percentage of respondents who reported spending part or all of their transfer in each category, broken down [by project, country, age, and gender](https://docs.google.com/spreadsheets/d/1PuFqGpOftwJiY92f6HUSA-15MTaYTjHdsF1J_lvc0sU/edit?usp=sharing).  

### Associations between demographic characteristics and spending outcomes

We'd like to know whether there are associations between recipient demographic features and selected expense categories while adjusting for compositional differences across projects. Unfortunately, the number of response categories selected by recipients also varies by demographic factors. To address this, we can model the inverse weighted response counts instead of raw counts, normalizing the differences in the number of category selections. We can then fit a series of linear regression models, one for each expense outcome. Each of these models examines the link between the probability of selecting a given expense category and the demographic characteristics of an "average" GD recipient. We can then compare coefficients across models to assess the relative strength of association between demographic factors and the selection of particular expense categories.

Here's the regression specification:
$$
Y_{ij} = \beta_{0} + \beta_{1}X_{1i} + \beta_{2}X_{2i} + \sum_{k=1}^{K} \delta_k F_k   +  \epsilon_{i}
$$
Where: 

- $Y_{{i,j}}$ is the $i$th recipients inverse weighted response count for the $j$th expense category (Food, housing, etc).  
- $X_1$ is the $i$th recipient's gender 
- $X_1$ is the $i$th recipients age  
- $F_k$ is an indicator variable for the $k$th project 

The following table contains the top five regression coefficients, standard errors, and p-values for gender and recipient age predictors fit within 10 linear probability models, one for each higher level expense category outcome. We essentially treat each outcome as a separate binary question and then assess the degree to which these two predictors are associated with that outcome after controlling for project-level fixed effects. This tells us whether there's an association between gender and age above and beyond compositional differences resulting from different target populations in different projects. 

{lpm_by_predictors_sorted_mdtbl}

There appears to be a mild positive associations between female gender and spending on food, age and spending on food, male gender and spending on agriculture, and age and spending on healthcare. Additionally, there is a negative association between entrepreneurship (or business activities) and advanced age. 

### Additional notes 

- Only the most recent followup survey from each recipient was included.
- Recipients whose gender was not `Male` or `Female`,  or whose gender or age was unknown were dropped from this analysis.

## Next steps 

If I were to spend more time on this analysis, this would be the areas I would focus on next: 

- Checking whether the proportions within the quantitative how much spending columns match the patterns observed within the what did the recipient spend on pick list question.
- More analysis and robustness checks to assess the scale of the demographic associations between recipient characteristics and the stated expenditure patterns described above.
- If we're willing to accept the inverse weighting assumption, there's nothing stopping us from also weighting by dollar amount at the transfer level to get to a universal overall aggregate percentage of transferred dollars estimated to be spent in aggregated categories.

## Appendix

The plot below presents the mean number of expense categories chosen by recipients, classified by both gender and age. Each data point represents the average number of categories selected by a specific gender within a specific age group. The error bars indicate two standard errors from the mean.

![Number of expense categories selected](output/cat_selected.png)

Both the age the gender differences in the number of categories selected are apparent: female recipients are more likely to select a larger number of expense categories than male recipients across all age groups. Additionally, older recipients are more likely to select a larger number of expense categories than younger recipients across both genders.


[^1]: The base query is [here](https://github.com/nburbank-givedirectly/xfer-followup/blob/main/queries/base_query.sql). Survey records in the research object were linked to transfers via followup records with identical recipient ids and similar timestamps. In rare cases transfers can be linked to more than one successfully completed followup survey. In these cases, only the most recent survey record was used.

[^2]: The numerator for [this proportion](https://github.com/nburbank-givedirectly/xfer-followup/blob/main/queries/prop_xfers_w_flup.sql) is the count of successful follow-up records that can be matched to a transfer id and have a `recipient_inactive_stage` of 0. The denominator is the count of all complete transfers with a creation date between October 1st, 2019 and September 30th, 2023.

[^3]: Most notably these questions were asked to recipients within the _Malawi Canva Basic Income Pilot_ and the _Liberia Maryland Basic Income_ programs.  

[^4]: In Field Salesforce, this is the `What_Did_The_Recipient_Spend_On` [question](https://givedirectly-field.my.salesforce.com/00N0b00000BuyeP?appLayout=setup&entityId=01I0b000001NFO0&noS1Redirect=true) within the Followup Object and the `Spending_Categories` [question](https://givedirectly-field.lightning.force.com/lightning/setup/ObjectManager/01I5a0000017dHL/FieldsAndRelationships/00N5a00000CsZNr/view) within the research object.



