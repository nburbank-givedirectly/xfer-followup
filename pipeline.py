#!/usr/bin/env python

import os
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 500)

from utils.notebook_utils import get_df
from col_lsts import (
    research_obc_values,
    fu_obj_values,
    research_spending_cols,
    how_much_cols,
    research_spending_cols,
    transfer_cols,
    fu_only,
    res_only,
    flup_to_res,
    pick_lst_to_quants_cols,
    category_aggregations,
    all_other,
)


for k in all_other:
    category_aggregations["Other"] = (
        category_aggregations["Other"] + category_aggregations[k]
    )
    del category_aggregations[k]

category_aggregations_spend = {}
for k, lst in category_aggregations.items():
    category_aggregations_spend[k] = list(
        set([pick_lst_to_quants_cols[c] for c in lst])
    )
category_aggregations_spend["Agriculture"] += ["spending_items_regular_agriculture"]
category_aggregations_spend["Entrepreneurship"] += [
    "spending_categories_business_other"
]
category_aggregations_spend["Other"] += [
    "spending_categories_other",
    "spending_motorcycle_bicycle",
    "spending_usaid_banned_items",
]


def check_cnts():
    query = """
    SELECT count(*) AS num_trans,
             count(distinct transfer_id) AS transfer_ids,
             sum(case WHEN fu.CreatedDate is NULL THEN 1 ELSE 0 end) AS non_null_cnt, 
             sum(case WHEN fu.CreatedDate is NULL THEN 1 ELSE 0 end) / count(*) AS prop
    FROM common.field_metrics_transfers t
    LEFT JOIN silver.field_salesforce_followup fu
        ON fu.Transfer__c = t.transfer_id
            AND fu.IsDeleted = FALSE
            AND fu.Is_Successful__c = TRUE
            AND fu.Has_Transfer_Been_Collected__c = TRUE
    WHERE t.transfer_status = 'Complete'
            AND t.recipient_inactive_stage = 0
            AND t.transfer_created_date > '2020-01-01'

    """
    df_cnts = get_df(query=query, limit=0, print_q=False)
    print(df_cnts)


def get_base_data():
    base_query = f"""
    WITH research AS
      (SELECT *
       FROM silver.field_salesforce_research
       WHERE IsDeleted = FALSE
         AND Research_Checkin_Stage__c = 'FLUP'),
         follow_up AS
      (SELECT *,
              ROW_NUMBER() over(PARTITION BY Recipient__c
                                ORDER BY Date_of_Follow_up__c DESC) AS rcpnt_fu_num,
                       ROW_NUMBER() over(PARTITION BY Transfer__c
                                ORDER BY Date_of_Follow_up__c DESC) AS tfr_fu_num
       FROM silver.field_salesforce_followup fu
       WHERE fu.IsDeleted = FALSE
         AND fu.Is_Successful__c = TRUE )

      SELECT fu.Id AS fu_id,
              res.Id AS res_id,
              fu.rcpnt_fu_num,
              fu.tfr_fu_num,
              ROW_NUMBER() over(PARTITION BY res.Recipient__c, fu.Id
                                ORDER BY Date_of_Follow_up__c DESC) AS res_fu_num,
             {','.join([f't.{c}' for c in transfer_cols])},
            {','.join([f'fu.{c}' for c in fu_only])},
            {','.join([f'COALESCE(res.{res_col}, fu.{fu_col}) as {res_col}' for fu_col,res_col in flup_to_res.items() ])},
            {','.join([f'res.{c}' for c in res_only])}
       
       FROM common.field_metrics_transfers t
       JOIN follow_up fu ON fu.Transfer__c = t.transfer_id
       LEFT JOIN research res ON fu.Recipient__c = res.Recipient__c
       AND abs(UNIX_TIMESTAMP(fu.CreatedDate) - UNIX_TIMESTAMP(res.CreatedDate)) < 60
       WHERE t.transfer_status = 'Complete'
         AND t.recipient_inactive_stage = 0
         AND t.transfer_created_date > '2020-01-01'
    """
    df = get_df(query=base_query, limit=0, print_q=False)
    print(len(df))
    df.columns = [c.lower().rstrip("__c") for c in df.columns]
    df = df[df.res_fu_num == 1].copy()
    df = df.drop("res_fu_num", axis=1)

    df = df[df.tfr_fu_num == 1].copy()
    df = df.drop("tfr_fu_num", axis=1)

    print(len(df))
    assert len(df) == df.transfer_id.nunique() == df.fu_id.nunique()
    return df


# Limit to 1 reponse per recpt


def split_and_ohe_str_lst(df, col, category_aggregations, agg=True):
    # df[col] = df[col].fillna("None")
    ohe = (
        df[~(df[col].isnull())][col]
        .str.split(";", expand=True)
        .stack()
        .str.get_dummies()
        .groupby(level=["recipient_id", "transfer_id", "fu_id"])
        .sum()
    )

    # Flat map from high-level to low-level
    rev_map = {
        col: group for group, cols in category_aggregations.items() for col in cols
    }

    tuples = [(rev_map[col], col) for col in ohe.columns]

    # Assign the MultiIndex to the dataframe's columns
    ohe.columns = pd.MultiIndex.from_tuples(tuples)

    # For now, max over lower level and return high level agg
    if agg:
        return ohe.T.groupby(level=0).max().T
    return ohe


def gen_multi_level_spend(df, category_aggregations_spend):
    rev_map = {
        col: group
        for group, cols in category_aggregations_spend.items()
        for col in cols
        if col in df.columns
    }

    tuples = [(rev_map[col], col) for col in df.columns if col in rev_map.keys()]

    spend_cols = [t[1] for t in tuples]
    spend_df = df[spend_cols].copy()
    spend_df = spend_df[spend_df.notna().any(axis=1)]
    spend_df.columns = pd.MultiIndex.from_tuples(tuples)
    return spend_df.fillna(0).sort_index(axis=1).T.groupby(level=0).sum().T


def run_analysis(df, name):

    expt = ""
    ohe = split_and_ohe_str_lst(
        df, "spending_categories", category_aggregations, agg=False
    )

    summary_counts = pd.DataFrame(
        {
            "cnt": ohe.sum(axis=0).sort_values(ascending=False),
            "prop": ohe.sum(axis=0).sort_values(ascending=False) / len(ohe),
        }
    )

    num_w_over_1_prct = summary_counts[summary_counts.prop > 0.005]
    num_w_under_1_prct = summary_counts[summary_counts.prop <= 0.0051]

    print(
        f"There were {len(num_w_over_1_prct)} reponse types that were selected by over 1% of respondents. Other responses accounted for {num_w_under_1_prct.cnt.sum()} responsese."
    )

    ohe = ohe.T.groupby(level=0).max().T

    spend_df = gen_multi_level_spend(df, category_aggregations_spend)

    fet_cols = [
        "continent",
        "record_type_name",
        "project_name",
        "transfer_created_date",
        "completed_date",
        "transfer_status",
        "transfer_amount_commitment_complete_usd",
        "transfer_amount_commitment_outstanding_usd",
        "recipient_age_at_contact",
        "recipient_gender",
    ]
    cnts = df[fet_cols].join(ohe, how="inner")

    prop_null = (
        cnts[["project_name", "None"]]
        .groupby("project_name")
        .sum()
        .div(cnts[["project_name"]].groupby("project_name").size(), axis=0)
    )
    to_exclude = list(prop_null[prop_null["None"] > 0.5].index)

    print(f"Dropping {to_exclude}")
    cnts = cnts[~cnts["project_name"].isin(to_exclude)]

    ## Catigory props
    overall = pd.DataFrame(
        {
            "cnt": cnts[ohe.columns].sum().sort_values(ascending=False),
            "prop": cnts[ohe.columns].sum().sort_values(ascending=False) / len(cnts),
        }
    )
    print(overall)
    overall_md = overall.round(3).to_markdown()

    def prop_tbl_by_cut(cnts, cut_col, sort_output=True):
        grp = cnts[([cut_col] + list(ohe.columns))].groupby(cut_col).sum()

        row_cnt = cnts[[cut_col]].groupby(cut_col).size()
        grp = grp.div(row_cnt, axis=0)
        grp["row_cnt"] = row_cnt
        grp = grp[(list(overall.index) + ["row_cnt"])]
        if sort_output:
            grp = grp.sort_values("row_cnt", ascending=False)
        return grp

    by_proj = prop_tbl_by_cut(cnts, "project_name")
    print(by_proj)
    by_proj.to_markdown()

    by_gender = prop_tbl_by_cut(cnts, "recipient_gender")

    bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    labels = [
        "0-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80-89",
        "90-99",
    ]
    cnts["age_group"] = pd.cut(
        cnts["recipient_age_at_contact"], bins=bins, labels=labels, right=False
    )
    by_age = prop_tbl_by_cut(cnts, "age_group", sort_output=False).round(3)

    with open(f"output_{name}.md", "w") as file:
        # Write the string to the file
        file.write(
f"""
# Results 
## Overall
{overall.round(3).to_markdown()}

## By Project
{by_proj.round(3).to_markdown()}

## By gender
{by_gender.round(3).to_markdown()}

## By age 
{by_age.round(3).to_markdown()}

""")


check_cnts()

df = get_base_data()

df = df.set_index(["recipient_id", "transfer_id", "fu_id"])
run_analysis(df, "full")
df = df[df.rcpnt_fu_num == 1].copy()
print("Running with 1 row per recipient")
run_analysis(df, "by_rcpt")


import IPython

IPython.embed()
