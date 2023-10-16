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


## Data Generating process

- We can ascertain this by summarizing the results of follow-up surveys, which are conducted with most recipients after a transfer is completed.
- Specifically, it appears that about <<84%>> of transfers in the last three years have included a successfully completed follow-up survey, either over the phone or in the field.
- These surveys contain two question types that are useful for understanding what recipients spend their transfers on

### Pre defined multiple response answers
- The first is a multiple response question, in which recipients can indicate 0, 1, or multiple categories from a pre-selected list of expense categories that they spend some or all of their transfer on.
- While this list has, up until recently, this list had 48 distinct options and was recently expanded to 64 distinct options, in practice, only about two dozen options are regularly selected by recipients.
- Because this is a multi-select pick-list, all we know is whether a recipient indicated that they spent some of their transfer on a given category.
- Data from these picklist gives us the ability to rank categories that are popular spending categories with recipients in general, but does not let us make statements about the specific proportion of a given transfer that was spent on a given category.
- What this means is for the pick-list questions, what we essentially have are a set of binary indicators that we can further summarize into high-level categories and use these to rank categories.
- We can use these responses to get a sense of the categories that appear to be the most popular with recipients.
- We can further look at crosstabs by project, country, recipient gender, and recipient age to identify other patterns.
- However, because giving an affirmative answer to a question only indicates that some money was spent on the selected category but not how much, all we can make are statements of the form X percent of recipients report spending some of their transfers category Y.
- These yield a series of binary proportions that give us an indication of the relative popularity of a category but don't add up to 100.
- That is to say, some recipients may say they spend their transfer on food, housing, and education. Other recipients may report spending on only food. Other recipients may report spending only on education. But if a recipient reports that they spent on food, housing, and education, that does not mean they spent a third of their transfer on each. Saying yes to a pick-list item gives no indication of the proportion of transfer that went to that pick-list item. It simply indicates that part of their transfer went to that item. But again, we can make general statements about which categories appear to be most popular.



### Quantitative proportional estimates

- The second set of questions are quantitative and do ask for recipients to specify the proportion of a given transfer that was spent on a given category.-
- Unfortunately, these questions are answered in many fewer circumstances, and we only have quantitative proportional responses in X percent of cases.

## Response weighting vs unique recipient weighting
- The additional lens through which we can look at is because most recipients receive multiple transfers and complete multiple follow-up surveys with us, we also have the option of whether to aggregate to our responses at the transfer level, where some recipients will respond more than once, or at the recipient level, where we simply take the most recent response we've received for a given recipient and use that in the data frame.

## Aggregating categories into more general high level categories
- To make high-level patterns categories interpretable, I have summarized the 64 pick-list items down to 12 general categories. See the appendix for the full mapping.
- This aggregation of more specific categories into general categories is of course somewhat subjective.



## Results

## Top response categories

{top_response_categories_mdtbl}

_{top_response_categories_note}_

## Top aggregated response categories


{top_aggregated_response_categories_mdtbl}

_{top_aggregated_response_categories_note}_

## By project


{by_project_mdtbl}

By project name.



## By gender 

{by_recipient_gender_mdtbl}

_Reported recipient expense category proportions by survey respondent gender. Currently sourced from the `recipient_gender` column in `common.field_metrics_transfers`._

## By age 

{by_recipient_age_mdtbl}

_Reported recipient at time of followup contact. Currently sourced from the `recipient_age_at_contact` column in `common.field_metrics_transfers`._


## Appendix 1: Full map of category aggregations


{full_map_of_category_aggregations_mdtbl}



