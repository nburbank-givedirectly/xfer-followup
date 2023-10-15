#!/usr/bin/env python

import os
import datetime
from typing import Dict, List
import pandas as pd


from helpers import get_df
from mappings import (
    AGGREGATE_TO_DETAILED_SPEND_CATEGORY,
    OTHER_AGGREGATE_SPEND_CATIGORIES,
    PICK_LST_TO_QUANTS_COLS,
    PROJECT_NAME_ABBREVIATIONS,
)


def further_group_agg_spend_catigories(
    aggregate_to_detailed_spend_category: Dict[str, List[str]],
    other_aggregate_spend_catigories: List[str],
) -> Dict[str, List[str]]:
    """further group rarely used aggregate categories under other for easier analysis"""

    for k in other_aggregate_spend_catigories:
        aggregate_to_detailed_spend_category["Other"] = (
            aggregate_to_detailed_spend_category["Other"]
            + aggregate_to_detailed_spend_category[k]
        )
        del aggregate_to_detailed_spend_category[k]
    return aggregate_to_detailed_spend_category


def mk_category_aggregations_spend(
    aggregate_to_detailed_spend_category, pick_lst_to_quants_cols
):
    """
    Generate an equivalent of the aggregate to detailed spend category
    decked for the quantitative columns, matching the mappings used
    above as closely as possible.
    """

    category_aggregations_spend = {}
    for k, lst in aggregate_to_detailed_spend_category.items():
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
    return category_aggregations_spend


def check_cnts():
    """Run diagnostic counts query and print results."""
    with open("queries/val_query.sql", "r", encoding="utf-8") as file:
        query = file.read()
    df_cnts = get_df(query=query, limit=0, print_q=False)
    print(df_cnts)


def get_base_data():
    """Run the base query and then filter to just one
    follow-up survey / research record per transfer.
    """

    with open("queries/base_query.sql", "r", encoding="utf-8") as file:
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
    """Splits and one-hot encodes the selected column using the supplied category aggregations dict."""
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


def gen_multi_level_spend(df, category_aggregations_spend):
    """
    Creates multi-index for quantitative spend columns so they can be
    aggregated in a similar way as aggregate picklist columns.
    """
    rev_map = {
        col: group
        for group, cols in category_aggregations_spend.items()
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


def add_features(df):
    """Add age group and abbreviated project name columns to dataframe"""
    df["project_name"] = (
        df["project_name"].replace(PROJECT_NAME_ABBREVIATIONS, regex=True).str.strip()
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
    """
    Given a cut or category column and a set of columns to sum over, it
    calculates the proportion of sums within each category value. Also
    calculates an 'All other' row for category values with few observations.
    """
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
        grp = grp.rename(columns=PROJECT_NAME_ABBREVIATIONS)

    grp.index.name = grp_disp_name

    return grp


def run_analysis(df, name, aggregate_to_detailed_spend_category):
    """
    Calculates one hot encoded values, then calculates the
    proportion of those one hot encoded values over a number of cut
    categories. It finds a dict containing a bloat, a dictionary of
    markdown tables, and a dictionary of pandas data frames to be
    later converted into excel files.
    """
    str_results = {"cur_date": datetime.datetime.now().strftime("%Y-%m-%d")}
    xls_results = []

    # Cacl One-hot encoded counts
    ohe_chache_path = f"data_cache/{name}_ohe.pq"
    if os.path.exists(ohe_chache_path):
        ohe = pd.read_parquet(ohe_chache_path)
    else:
        ohe = split_and_ohe_str_lst(
            df, "spending_categories", aggregate_to_detailed_spend_category, agg=False
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

    # Calculate by project
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
        [
            (k, col)
            for k, cols in aggregate_to_detailed_spend_category.items()
            for col in cols
        ],
        columns=["Agg. category", "Category"],
    ).set_index("Agg. category")

    return {
        "str_results": str_results,
        "xls_results": xls_results,
        "df": df,
        "cnts": cnts,
    }


def dl_and_analyze_data():
    """
    Download and analyze data -- runs analysis pipeline.
    """

    check_cnts()

    aggregate_to_detailed_spend_category = further_group_agg_spend_catigories(
        AGGREGATE_TO_DETAILED_SPEND_CATEGORY, OTHER_AGGREGATE_SPEND_CATIGORIES
    )
    category_aggregations_spend = mk_category_aggregations_spend(
        aggregate_to_detailed_spend_category, PICK_LST_TO_QUANTS_COLS
    )

    df = get_base_data()
    df = add_features(df)
    df = df.set_index(["recipient_id", "transfer_id", "fu_id"])

    # df = df[df.rcpnt_fu_num == 1].copy()
    # print("Running with 1 row per recipient")
    # run_analysis(df, "by_rcpt")

    return run_analysis(df, "full", aggregate_to_detailed_spend_category)


if __name__ == "__main__":
    results = dl_and_analyze_data()
    import IPython

    IPython.embed()
