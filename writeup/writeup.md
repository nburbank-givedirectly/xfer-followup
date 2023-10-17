---
title: "Expenditure analysis"
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

- Food is the most common reported expense category. 
- {top_aggregated_response_categories_note}
- Only {vice_prct}% of respondents, or slightly more 1 in 1000 respondents, report spending any part of their transfer on a vice activity (Alcohol, drugs, etc.). 
- There is substantial heterogeneity in category patterns across projects and countries.  
- There is limited evidence for an association between both gender and age and Food, even after accounting for project-specific cohort effects. Women, older recipients, and older women, are all more likely to spend their transfer on food than other recipients. Men are more likely to spend their transfers on entrepreneurship activities. 

## Data Generating process 



We'd like to be able to make descriptive statements about what GiveDirectly recipients spend their transfers on to identify patterns across countries, demographics, and project types. The data for this analysis is derived from follow-up surveys, which are conducted with most recipients after a transfer is completed. Specifically, <<84%>> of transfers in the last three years have included a successfully completed follow-up survey. These follow-up surveys include two sets of questions that provide insight into how recipients spend their transfers. 

First, a multi-option pick list question lets recipients select one or more from a set of approximately 60 predefined spending categories to indicate that at least part of their transfer was spent on the selected category or categories.[^1] And second, a series of quantitative questions that enables recipients to provide proportional estimates describing specifically how much of a given transfer was spent within a set of predefined categories. [^2] While answers to the second set of questions are more precise than the multi-option question, to date responses to the second set of questions have only been collected within a small number of projects.[^3] For this reason, most of this analysis focuses on analyzing the less informative but more numerous answers to the first question type. I further filtered the dataset down to projects with completed transfers after January 1st, 2020, that had at least 1000 survey responses and at least an 80% completion rate for the `What Did The Recipient Spend On` question. This resulted in a dataset of approximately {N_obs}k observations from {N_rcp}k recipients in {N_proj} projects across {N_countries} countries.

### Pre defined multiple response answers
- In the ``What Did The Recipient Spend On` question, recipients can indicate 0, 1, or multiple categories from a pre-selected list of expense categories that they spent some or all of their transfer on.
- There are multiple variations of this question. The most common variant has 48 distinct options; a recently expanded contains 64 distinct options. However, in practice, only 16 options are commonly selected by recipients. 
- The average response to a followup survey containing a non-null response to question included  {mean_number_of_categories} selected categories. 


[^1]: In Field Salesforce, this is the `What Did The Recipient Spend On` [question](https://givedirectly-field.my.salesforce.com/00N0b00000BuyeP?appLayout=setup&entityId=01I0b000001NFO0&noS1Redirect=true) within the Followup Object and the `Spending_Categories` [question](https://givedirectly-field.lightning.force.com/lightning/setup/ObjectManager/01I5a0000017dHL/FieldsAndRelationships/00N5a00000CsZNr/view) within the research object.
[^2]:These are sometimes refered to the as the `how much spent on` questions. 
[^3]: Most notably theses questions were asked to recipients within the _Malawi Canva Basic Income Pilot_ and the _Liberia Maryland Basic Income_ programs.  


## Results

## Top response categories

{top_response_categories_by_response_mdtbl}

## Top aggregated response categories


{top_aggregated_response_categories_mdtbl}

_{top_aggregated_response_categories_note}_

