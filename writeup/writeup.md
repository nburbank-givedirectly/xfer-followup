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

- Food is the most common reported expense category. 59.7% of respondents report spending part or all of their transfer on food.
- {top_aggregated_response_categories_note}
- Only {vice_prct}% of respondents, or slightly more than 1 in 1000 respondents, report spending any part of their transfer on a "vice activity" (Alcohol, drugs, etc.). 
- There is substantial variation in response patterns [by project, country, age, and gender](https://docs.google.com/spreadsheets/d/1PuFqGpOftwJiY92f6HUSA-15MTaYTjHdsF1J_lvc0sU/edit?usp=sharing).
- After accounting for compositional differences across projects, there is evidence for a mild positive association between female gender and spending on food, education, and entrepreneurship activities, and a negative association with agriculture activities. Additionally, older recipients are more likely to report spending on food and less likely to report spending on entrepreneurship activities.

## Dataset definition 


The data for this analysis is derived from follow-up surveys, which are conducted with most recipients after a transfer is completed. Specifically, {prop_xfers_w_flup}% of transfers in the last three years have included a successfully completed follow-up survey. These follow-up surveys include two sets of questions that provide insight into how recipients spend their transfers. 

First, a multi-option pick list question lets recipients select one or more from a set of approximately 60 predefined spending categories to indicate that at least part of their transfer was spent on the selected category or categories. And second, a series of quantitative questions that enables recipients to provide proportional estimates describing specifically how much of a given transfer was spent within a set of predefined categories.[^2] While answers to the second set of questions are more precise than the multi-option question, to date responses to the second set of questions have only been collected within a small number of projects.[^3] For this reason, this analysis focuses on analyzing the less informative but more numerous answers to the first question type. I further filtered the dataset down to projects with completed transfers after January 1st, 2020 that had at least 1000 completed follow up survey responses and at least an 80% completion rate for the `What Did The Recipient Spend On` question. This resulted in a dataset of approximately {N_obs}k observations from {N_rcp}k recipients in {N_proj} projects across {N_countries} countries.

####  Additional notes 
- In the ``What Did The Recipient Spend On` question, recipients can indicate 0, 1, or multiple categories from a pre-selected list of expense categories that they spent some or all of their transfer on. There are multiple variations of this question that this analysis aggregates over. The most common variant has 48 distinct options; a recently expanded variant contains 64 distinct options. However, in practice, only 16 options are commonly selected by recipients. 
- The average response to a followup survey containing a non-null response included  {mean_number_of_categories} selected categories.
- For many aggregations, we can choose to weight aggregations by recipient, including only one (the most recent) response per recipient, or by response, giving higher weight to recipients with multiple transfers. In this document, aggregations are weighted by response, unless specified otherwise. For most of the questions I've analyzed the choice of weighting method does not substantially affect the results.
- Ineligible, discarded, written off, refused recipients are filtered out of this dataset.

[^4]: I need to double check this figure.
[^1]: In Field Salesforce, this is the `What Did The Recipient Spend On` [question](https://givedirectly-field.my.salesforce.com/00N0b00000BuyeP?appLayout=setup&entityId=01I0b000001NFO0&noS1Redirect=true) within the Followup Object and the `Spending_Categories` [question](https://givedirectly-field.lightning.force.com/lightning/setup/ObjectManager/01I5a0000017dHL/FieldsAndRelationships/00N5a00000CsZNr/view) within the research object.
[^2]:These are sometimes referred to as the `how much spent on` questions. 
[^3]: Most notably these questions were asked to recipients within the _Malawi Canva Basic Income Pilot_ and the _Liberia Maryland Basic Income_ programs.  

## Top aggregated expense categories


{top_aggregated_response_categories_mdtbl}

In order to observe higher level trends, it's necessarily to aggregate the pick list categories into higher level groupings.  I created ten higher level groupings, defined [here](https://github.com/nburbank-givedirectly/xfer-followup/blob/22fac762e351be90505d6100118df603485aeec9/src/mappings.py#L337). This aggregation is necessarily somewhat subjective. Among these higher level categories, food is the most popular reported expense category, with {top_aggregated_response_categories_note}. 

{top_response_categories_note_by_response}

{top_response_categories_by_response_mdtbl}

#### Spending on "vice" categories   

As noted above, a small percentage of respondents ({vice_prct}%) reported spending at least part of their transfer on one or more "vice activities". These are defined as commercial sex, gambling, or alcohol/drugs. The large majority of respondents who reported spending within this group of categories selected the final category (alcohol or drugs). However, this likely underestimates the actual number of recipients who spend in these categories due to negative reporting bias. (Recipients are unlikely to self-report spending on categories that they perceive as controversial.)

### Variation by project and demographic characteristics

Using the aggregated response categories, we can calculate the percentage of respondents who reported spending part or all of their transfer in each category, broken down [by project, country, age, and gender](https://docs.google.com/spreadsheets/d/1PuFqGpOftwJiY92f6HUSA-15MTaYTjHdsF1J_lvc0sU/edit?usp=sharing).  

### Associations between demographic characteristics and spending outcomes 

This table contains the regression coefficients, standard errors, and p-values for gender and recipient age predictors fit within 10 linear probability models, one for each higher level expense category outcome. We essentially treat each outcome as a separate binary question and then assess the degree to which these two predictors are associated with that outcome after controlling for project level fixed effects. This tells us whether there's an association between gender and age above and beyond compositional differences resulting from different target populations in different projects. 

{lpm_by_predictors_mdtbl}

As noted above, there appears to be a mild positive association between female gender and spending on food, education, and entrepreneurship activities, and a negative association with agriculture activities. Additionally, older recipients are more likely to report spending on food and less likely to report spending on entrepreneurship activities.

## Next steps 

If I were to spend more time on this analysis, this would be the areas I would focus on next: 

- Checking whether the proportions within the quantitative how much spending columns match the patterns observed within the what did the recipient spend on pick list question.
- More analysis and robustness checks to assess the scale of the demographic associations between recipient characteristics and the stated expenditure patterns described above.

