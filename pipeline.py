#!/usr/bin/env python

import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 500)

from utils.notebook_utils import get_df
from col_lsts import (
    TRANSFER_COLS,
    FU_ONLY,
    RES_ONLY,
    FLUP_TO_RES,
    PICK_LST_TO_QUANTS_COLS,
    CATEGORY_AGGREGATIONS,
    ALL_OTHER,
    ABBREVIATIONS
)



for k in ALL_OTHER:
    CATEGORY_AGGREGATIONS["Other"] = (
        CATEGORY_AGGREGATIONS["Other"] + CATEGORY_AGGREGATIONS[k]
    )
    del CATEGORY_AGGREGATIONS[k]

CATEGORY_AGGREGATIONS_SPEND = {}
for k, lst in CATEGORY_AGGREGATIONS.items():
    CATEGORY_AGGREGATIONS_SPEND[k] = list(
        set([PICK_LST_TO_QUANTS_COLS[c] for c in lst])
    )
CATEGORY_AGGREGATIONS_SPEND["Agriculture"] += ["spending_items_regular_agriculture"]
CATEGORY_AGGREGATIONS_SPEND["Entrepreneurship"] += [
    "spending_categories_business_other"
]
CATEGORY_AGGREGATIONS_SPEND["Other"] += [
    "spending_categories_other",
    "spending_motorcycle_bicycle",
    "spending_usaid_banned_items",
]


def check_cnts():
    query = """
    SELECT count(*) AS num_trans,
             count(distinct transfer_id) AS transfer_ids,
             sum(case WHEN fu.CreatedDate is NULL THEN 1 ELSE 0 end) AS non_null_cnt, 
             sum(case WHEN fu.CreatedDate is NULL THEN 1 ELSE 0 end) / count(*) AS prop_overall,
             sum(case WHEN fu.What_Did_The_Recipient_Spend_On__c is NULL THEN 1 ELSE 0 end) AS spend_cat_non_null_cnt,
             sum(case WHEN fu.What_Did_The_Recipient_Spend_On__c is NULL THEN 1 ELSE 0 end) / count(*) AS spend_cat_prop_overall

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
             {','.join([f't.{c}' for c in TRANSFER_COLS])},
            {','.join([f'fu.{c}' for c in FU_ONLY])},
            {','.join([f'COALESCE(res.{res_col}, fu.{fu_col}) as {res_col}' for fu_col,res_col in FLUP_TO_RES.items() ])},
            {','.join([f'res.{c}' for c in RES_ONLY])}
       
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


# Limit to 1 response per recipient


def split_and_ohe_str_lst(df, col, category_aggregations, agg=True):
    print((len(df)))

    print(
        f"{((~df[col].isnull()).sum()  / len(df)*100):.2f}%, or {(~df[col].isnull()).sum()} are rows for OHE non-null"
    )
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


def gen_multi_level_spend(df, CATEGORY_AGGREGATIONS_SPEND):
    rev_map = {
        col: group
        for group, cols in CATEGORY_AGGREGATIONS_SPEND.items()
        for col in cols
        if col in df.columns
    }

    tuples = [(rev_map[col], col) for col in df.columns if col in rev_map.keys()]

    spend_cols = [t[1] for t in tuples]
    spend_df = df[spend_cols].copy()
    print(len(spend_df))
    print(
        f"{(spend_df.notna().any(axis=1).sum()  / len(df)*100):.2f}%, or {spend_df.notna().any(axis=1).sum()} are rows for spend are non-null"
    )
    spend_df = spend_df[spend_df.notna().any(axis=1)]
    spend_df.columns = pd.MultiIndex.from_tuples(tuples)
    return spend_df.fillna(0).sort_index(axis=1).T.groupby(level=0).sum().T


def gen_results_md(resutls, name):

    output_str = "# Results\n"

    for res in resutls:
        table = res["result"]

        md_tbl = table.to_markdown() if isinstance(table, pd.DataFrame) else table
        output_str += f"\n## {res['title']}\n\n{md_tbl}\n\n{res['note']}\n"

    with open(f"output/output_{name}.md", "w") as file:
        file.write(output_str)


def prop_tbl_by_cut(
    cnts,
    cut_col,
    sum_cols,
    grp_disp_name=None,
    min_grp_cnt=1000,
    sort_output=True,
    abbr_col_names=True,
):
    cnts = cnts.copy()

    if grp_disp_name is None:
        grp_disp_name = cut_col


    # cut_cols = [cut_cols] if isinstance(cut_cols, str) else cut_cols


    row_cnts = cnts[[cut_col]].groupby(cut_col).size().to_dict()

    if min_grp_cnt is not None:
        cnts[cut_col] = np.where(
            (cnts[cut_col].map(row_cnts) > min_grp_cnt), cnts[cut_col], "All other"
        )

    row_cnts = cnts[[cut_col]].groupby(cut_col).size()
    grp = cnts[([cut_col] + sum_cols)].groupby(cut_col).sum()

    grp = grp.div(row_cnts, axis=0) * 100
    grp["Obs."] = row_cnts
    grp = grp[(["Obs."] + sum_cols)]
    if sort_output:
        grp = grp.sort_values("Obs.", ascending=False)
    if "All other" in grp.index:
        grp = grp.loc[[c for c in grp.index if c != "All other"] + ["All other"]]
    if abbr_col_names:
        grp = grp.rename(columns=ABBREVIATIONS)

    grp.index.name = grp_disp_name

    return grp


def prop_tbl_by_cut_multi(
    cnts,
    cut_col,
    sum_cols,
    grp_disp_name=None,
    min_grp_cnt=1000,
    sort_output=True,
    abbr_col_names=True,
):
    cnts = cnts.copy()

    if grp_disp_name is None:
        grp_disp_name = cut_col


    # cut_cols = [cut_cols] if isinstance(cut_cols, str) else cut_cols

 

    row_cnts = cnts[cut_col].groupby(cut_col).size()
    cnts['row_cnts'] = cnts[cut_col].groupby(cut_col).size()


    cnts.apply(lambda row: (row['country'], row['project_name']), axis=1).map(row_cnts)

    import IPython; IPython.embed()



    if min_grp_cnt is not None:
        cnts[cut_col] = np.where(( cnts.apply(lambda row: (row['country'], row['project_name']), axis=1).map(row_cnts) > min_grp_cnt), cnts[cut_col], ('All other',''))

    row_cnts = cnts[[cut_col]].groupby(cut_col).size()
    grp = cnts[([cut_col] + sum_cols)].groupby(cut_col).sum()

    grp = grp.div(row_cnts, axis=0) * 100
    grp["Obs."] = row_cnts
    grp = grp[(["Obs."] + sum_cols)]
    if sort_output:
        grp = grp.sort_values("Obs.", ascending=False)
    if "All other" in grp.index:
        grp = grp.loc[[c for c in grp.index if c != "All other"] + ["All other"]]
    if abbr_col_names:
        grp = grp.rename(columns=ABBREVIATIONS)

    grp.index.name = grp_disp_name

    return grp


def run_analysis(df, name):

    results = []
    ohe = split_and_ohe_str_lst(
        df, "spending_categories", CATEGORY_AGGREGATIONS, agg=False
    )

    # Top responses by original categories
    summary_counts = pd.DataFrame(
        {
            "N": ohe.sum(axis=0).sort_values(ascending=False),
            "Prct": (ohe.sum(axis=0).sort_values(ascending=False) / len(ohe))*100,
        }
    )

    num_w_over_1_prct = summary_counts[summary_counts.Prct > 0.01]
    num_w_under_1_prct = summary_counts[summary_counts.Prct <= 0.01]
    note = f"There were {len(num_w_over_1_prct)} response types that were selected by over 1% of respondents. All other responses only contributed for {num_w_under_1_prct['N'].sum()} responses."

    results.append(
        {
            "title": "Top response categories",
            "result": num_w_over_1_prct.reset_index(names=["Agg. category", "Category"])
            .round(3)
            .to_markdown(index=False),
            "note": note,
        }
    )

    # Sum up to higher level categories
    ohe = ohe.T.groupby(level=0).max().T

    # spend_df = gen_multi_level_spend(df, CATEGORY_AGGREGATIONS_SPEND)

    fet_cols = [
        "country",
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

    ## Category props
    overall = pd.DataFrame(
        {
            "N": cnts[ohe.columns].sum().sort_values(ascending=False),
            "Prct": (cnts[ohe.columns].sum().sort_values(ascending=False) / len(cnts))*100,
        }
    )

    top_5 = list(overall.index[:5])
    prop_w_resp_in_top_5 = cnts[top_5].max(axis=1).sum() / len(cnts)
    top_5_str = [n.lower() for n in top_5]

    results.append(
        {
            "title": "Top aggregated response categories",
            "result": overall.round(1).round(3),
            "note": f"{prop_w_resp_in_top_5*100:.1f}% of surveyed recipients indicated that they spent at least part of their transfer on one or more of {', '.join(top_5_str[:4])}, and {top_5_str[4]} expenses.",
        }
    )

    sum_cols = list(overall.index)

    by_proj = prop_tbl_by_cut_multi(
        cnts, ["country" ,"project_name"], sum_cols, grp_disp_name="Project", min_grp_cnt=5000
    )


    by_proj.round(2).to_csv(f"output/by_proj_{name}.tsv", sep="\t")


    # cut_cols = ['country','project_name']

    # cnts[(cut_cols + sum_cols)].groupby(cut_cols).sum()


    results.append(
        {
            "title": "By project",
            "result": by_proj.round(1),
            "note": "By project name.",
        }
    )

    print(by_proj.round(2))
    by_proj.to_markdown()

    by_gender = prop_tbl_by_cut(cnts, "recipient_gender", sum_cols, grp_disp_name="Gender")
    results.append(
        {
            "title": "By recipient gender",
            "result": by_gender.round(1),
            "note": "Reported recipient expense category proportions by survey respondent gender. Currently sourced from the `recipient_gender` column in `common.field_metrics_transfers`.",
        }
    )

    bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 99, 150]
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
        "99+",
    ]
    cnts["age_group"] = pd.cut(
        cnts["recipient_age_at_contact"], bins=bins, labels=labels, right=False
    )
    by_age = prop_tbl_by_cut(
        cnts, "age_group", sum_cols, grp_disp_name="Age", min_grp_cnt=None, sort_output=False
    )

    results.append(
        {
            "title": "By recipient age",
            "result": by_age.round(1),
            "note": "Reported recipient at time of followup contact. Currently sourced from the `recipient_age_at_contact` column in `common.field_metrics_transfers`.",
        }
    )

    results.append(
        {
            "title": "Appendix 1: Full map of category aggregations",
            "result": pd.DataFrame(
                [(k, col) for k, cols in CATEGORY_AGGREGATIONS.items() for col in cols],
                columns=["Agg. category", "Category"],
            ).set_index("Agg. category"),
            "note": "",
        }
    )

    gen_results_md(results, name)
    return results


check_cnts()

df = get_base_data()

short_names = {'Large Transfer':'LT', 'Emergency Relief':'ER', 'COVID-19':'C19'}
countries = df.country.unique()
for c in countries:
    short_names[c] = ''


df['project_name'] = df['project_name'].replace(short_names, regex=True).str.strip()

df = df.set_index(["recipient_id", "transfer_id", "fu_id"])

run_analysis(df, "full")
df = df[df.rcpnt_fu_num == 1].copy()
# print("Running with 1 row per recipient")
# run_analysis(df, "by_rcpt")
