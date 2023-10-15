#!/usr/bin/env python

import os
import warnings
import subprocess
import datetime

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 500)

from notebook_utils import get_df
from col_lsts import (
    PICK_LST_TO_QUANTS_COLS,
    CATEGORY_AGGREGATIONS,
    ALL_OTHER,
    ABBREVIATIONS,
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
    with open("queries/val_query.sql", "r") as file:
        query = file.read()
    df_cnts = get_df(query=query, limit=0, print_q=False)
    print(df_cnts)


def get_base_data():
    with open("queries/base_query.sql", "r") as file:
        base_query = file.read()

    df = get_df(query=base_query, limit=0, print_q=False)
    # start_len = len(df)
    df.columns = [c.lower().rstrip("__c") for c in df.columns]
    df = df[df.res_fu_num == 1].copy()
    df = df.drop("res_fu_num", axis=1)

    df = df[df.tfr_fu_num == 1].copy()
    df = df.drop("tfr_fu_num", axis=1)

    # end_len = len(df)

    assert len(df) == df.transfer_id.nunique() == df.fu_id.nunique()
    return df


def split_and_ohe_str_lst(df, col, category_aggregations, agg=True):
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


def gen_results_md(results, name)-> None:
    """Stich together markdown results file with string results"""
    with open("writeup/writeup.md", "r") as file:
        templete = file.read()

    output_str = templete.format(**results)

    with open(f"output/output_{name}.md", "w") as file:
        file.write(output_str)


def gen_excel(xls_results, output_str, also_tsv=True):
    """Make excel file with one table per sheet."""
    with pd.ExcelWriter(f"output/{output_str}.xlsx") as writer:
        for sheet_name, df in xls_results:
            df.to_excel(writer, sheet_name=sheet_name, index=True, float_format="%.3f")
            if also_tsv:
                df.to_csv(f"output/{output_str}_{sheet_name}.tsv", sep="\t")


def add_features(df):
    
    short_names = {
        "Large Transfer": "LT",
        "Emergency Relief": "ER",
        "COVID-19": "C19",
        "Basic Income": "BI",
    }

    df["project_name"] = df["project_name"].replace(short_names, regex=True).str.strip()

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
    df["age_group"] = pd.cut(
        df["recipient_age_at_contact"], bins=bins, labels=labels, right=False
    )

    return df


def prop_tbl_by_cut(
    df,
    cut_col,
    sum_cols,
    grp_disp_name=None,
    min_grp_cnt=1000,
    sort_output=True,
    abbr_col_names=True,
):
    df = df.copy()

    if grp_disp_name is None:
        grp_disp_name = cut_col

    if min_grp_cnt is not None:
        category_counts = df[cut_col].value_counts()
        categories_below_threshold = category_counts[
            category_counts < min_grp_cnt
        ].index
        df.loc[df[cut_col].isin(categories_below_threshold), cut_col] = "All other"

    row_cnts = df[cut_col].value_counts()
    grp = df[([cut_col] + sum_cols)].groupby(cut_col).sum()

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
    str_results = {"cur_date": datetime.datetime.now().strftime("%Y-%m-%d")}
    xls_results = []

    # Cacl One-hot encoded counts
    ohe_chache_path = f"data_cache/{name}_ohe.pq"
    if os.path.exists(ohe_chache_path):
        ohe = pd.read_parquet(ohe_chache_path)
    else:
        ohe = split_and_ohe_str_lst(
            df, "spending_categories", CATEGORY_AGGREGATIONS, agg=False
        )
        ohe.to_parquet(ohe_chache_path)

    # Top responses by original categories
    summary_counts = pd.DataFrame(
        {
            "N": ohe.sum(axis=0).sort_values(ascending=False),
            "Prct": (ohe.sum(axis=0).sort_values(ascending=False) / len(ohe)) * 100,
        }
    )
    num_w_over_1_prct = summary_counts[summary_counts.Prct > 0.01]
    num_w_under_1_prct = summary_counts[summary_counts.Prct <= 0.01]
    str_results["top_response_categories_mdtbl"] = (
        num_w_over_1_prct.reset_index(names=["Agg. category", "Category"])
        .round(3)
        .to_markdown(index=False)
    )
    str_results[
        "top_response_categories_note"
    ] = f"There were {len(num_w_over_1_prct)} response types that were selected by over 1% of respondents. All other responses only contributed for {num_w_under_1_prct['N'].sum()} responses."

    # For rest of analysis, sum up to higher level categories
    ohe = ohe.T.groupby(level=0).max().T

    fet_cols = [
        "country",
        "record_type_name",
        "project_name",
        # "transfer_created_date",
        # "completed_date",
        # "transfer_status",
        # "transfer_amount_commitment_complete_usd",
        # "transfer_amount_commitment_outstanding_usd",
        "recipient_gender",
        "age_group",
    ]

    cnts = df[fet_cols].join(ohe, how="inner")

    ## Category props
    overall = pd.DataFrame(
        {
            "N": cnts[ohe.columns].sum().sort_values(ascending=False),
            "Prct": (cnts[ohe.columns].sum().sort_values(ascending=False) / len(cnts))
            * 100,
        }
    )

    top_5 = list(overall.index[:5])
    prop_w_resp_in_top_5 = cnts[top_5].max(axis=1).sum() / len(cnts)
    top_5_str = [n.lower() for n in top_5]

    note = f"{prop_w_resp_in_top_5*100:.1f}% of surveyed recipients indicated that they spent at least part of their transfer on one or more of {', '.join(top_5_str[:4])}, and {top_5_str[4]} expenses."
    str_results["top_aggregated_response_categories_mdtbl"] = overall.round(
        1
    ).to_markdown()
    str_results["top_aggregated_response_categories_note"] = note

    sum_cols = list(overall.index)

    # Caculate by project
    by_proj = prop_tbl_by_cut(
        cnts, "project_name", sum_cols, grp_disp_name="Project", min_grp_cnt=1000
    ).reset_index()

    # Hacky way to add country into index
    proj_to_country = (
        cnts[["project_name", "country"]]
        .drop_duplicates()
        .set_index("project_name")
        .to_dict(orient="dict")["country"]
    )
    proj_to_country["All other"] = "All other"

    by_proj["Country"] = by_proj["Project"].map(proj_to_country)

    country_to_blank = {country: "" for country in cnts["country"].unique()}
    by_proj["Project"] = (
        by_proj["Project"].replace(country_to_blank, regex=True).str.strip()
    )
    by_proj = by_proj.set_index(["Country", "Project"]).sort_index()
    xls_results.append(("by_proj", by_proj))
    str_results["by_project_mdtbl"] = by_proj.round(1)

    # By gender
    by_gender = prop_tbl_by_cut(
        cnts, "recipient_gender", sum_cols, grp_disp_name="Gender"
    )
    str_results["by_recipient_gender_mdtbl"] = by_gender.round(1)
    xls_results.append(("by_gender", by_gender))

    # By age bin
    by_age = prop_tbl_by_cut(
        cnts,
        "age_group",
        sum_cols,
        grp_disp_name="Age",
        min_grp_cnt=None,
        sort_output=False,
    )
    str_results["by_recipient_age_mdtbl"] = by_age.round(1).to_markdown()
    str_results["full_map_of_category_aggregations_mdtbl"] = pd.DataFrame(
        [(k, col) for k, cols in CATEGORY_AGGREGATIONS.items() for col in cols],
        columns=["Agg. category", "Category"],
    ).set_index("Agg. category")
    gen_results_md(str_results, name)
    gen_excel(xls_results, name)

    return str_results

if __name__ == "__main__":

    check_cnts()

    df = get_base_data()
    df = add_features(df)
    df = df.set_index(["recipient_id", "transfer_id", "fu_id"])
    run_analysis(df, "full")


    subprocess.run(
    [
        "pandoc",
        "-f",
        "markdown-auto_identifiers",
        "output/output_full.md",
        "-o",
        "output/output_full.docx",
        "--reference-doc=writeup/custom-reference.docx",
    ])
    df = df[df.rcpnt_fu_num == 1].copy()
    print("Running with 1 row per recipient")
    run_analysis(df, "by_rcpt")

    subprocess.run(
    [
        "pandoc",
        "-f",
        "markdown-auto_identifiers",
        "output/output_by_rcpt.md",
        "-o",
        "output/output_by_rcpt.docx",
        "--reference-doc=writeup/custom-reference.docx",
    ])

