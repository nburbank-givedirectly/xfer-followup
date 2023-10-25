#!/usr/bin/env python

import os
import datetime
from typing import Dict, List
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

from helpers import get_df
from mappings import (
    AGGREGATE_TO_DETAILED_SPEND_CATEGORY,
    OTHER_AGGREGATE_SPEND_CATIGORIES,
    PICK_LST_TO_QUANTS_COLS,
    PROJECT_NAME_ABBREVIATIONS,
    VICE_CATEGORIES,
)
from plots import expense_cats_plot

# Min number of non-null responsese for project to be included
MIN_PROJ_N = 1000
# Min proportion of follow-up surveys for which spending category question is non-null.
MIN_PROJ_PROP = 0.8

RESULTS = {
    "str_results": {},
    "xls_results": [],
    "xls_cnts": [],
    "diagnostics": [],
}


def check_prop_xfers_w_flup_survey():
    """Run diagnostic counts query and print results."""
    with open("queries/prop_xfers_w_flup.sql", "r", encoding="utf-8") as file:
        query = file.read()
    df_cnts = get_df(query=query, limit=0, print_q=False)
    RESULTS["str_results"]["prop_xfers_w_flup"] = (
        (df_cnts["prop_overall"] * 100).values[0].round(1)
    )
    RESULTS["diagnostics"].append(("xfers_and_flup_cnts", df_cnts))
    print(df_cnts)


def further_group_agg_spend_catigories(
    aggregate_to_detailed_spend_category: Dict[str, List[str]],
    other_aggregate_spend_catigories: List[str],
) -> Dict[str, List[str]]:
    """Further group rarely used aggregate categories under other for easier analysis"""

    for k in other_aggregate_spend_catigories:
        aggregate_to_detailed_spend_category["Other"] = (
            aggregate_to_detailed_spend_category["Other"]
            + aggregate_to_detailed_spend_category[k]
        )
        del aggregate_to_detailed_spend_category[k]

    full_map_of_category_aggregations = pd.DataFrame(
        [
            (k, col)
            for k, cols in aggregate_to_detailed_spend_category.items()
            for col in cols
        ],
        columns=["Agg. category", "Category"],
    ).set_index("Agg. category")

    RESULTS["diagnostics"].append(
        ("category_aggregations", full_map_of_category_aggregations)
    )
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


def get_base_data():
    """Run the base query and then filter to just one
    follow-up survey / research record per transfer.
    """

    with open("queries/base_query.sql", "r", encoding="utf-8") as file:
        base_query = file.read()

    df = get_df(query=base_query, limit=0, print_q=False)
    start_len = len(df)
    df.columns = [c.lower().rstrip("__c") for c in df.columns]
    df = df[df.res_fu_num == 1].copy()
    df = df.drop("res_fu_num", axis=1)

    df = df[df.tfr_fu_num == 1].copy()
    df = df.drop("tfr_fu_num", axis=1)

    end_len = len(df)
    print(f"The starting number of rows is {start_len}.")
    print(
        f"Reducing to one followup per transfer and one research row per followup dropped {start_len - end_len} rows resulting in {end_len} rows."
    )

    assert (
        len(df) == df.transfer_id.nunique() == df.fu_id.nunique()
    ), "Number of followup and transfer IDs don't match"
    return df


# Feature creation


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
        .groupby(level=["rcpnt_fu_num", "recipient_id", "transfer_id", "fu_id"])
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


def add_features(df, aggregate_to_detailed_spend_category):
    """Add age group and abbreviated project name columns to dataframe"""

    # Set index
    df = df.set_index(["rcpnt_fu_num", "recipient_id", "transfer_id", "fu_id"])

    df["proj_name"] = (
        df["project_name"].replace(PROJECT_NAME_ABBREVIATIONS, regex=True).str.strip()
    )

    # Add age group feature
    bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 150]
    age_labels = [
        "13-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80-89",
        "90+",
    ]
    df["age_group"] = pd.cut(
        df["recipient_age_at_contact"], bins=bins, labels=age_labels, right=False
    )

    df["year"] = df["completed_date"].dt.year

    proj_nm_to_type = {"Mozambique USAID Agricultural Lump-Sum": "Large Transfer"}
    project_types = ["Large Transfer", "Emergency Relief", "Basic Income", "Cash+"]
    for p in df.project_name.unique():
        for pt in project_types:
            if pt in p:
                proj_nm_to_type[p] = pt
    df["project_type"] = df.project_name.map(proj_nm_to_type)

    ohe = split_and_ohe_str_lst(
        df, "spending_categories", aggregate_to_detailed_spend_category, agg=False
    )
    agg_ohe = ohe.T.groupby(level=0).max().T

    ohe.columns = pd.MultiIndex.from_tuples([("ohe", c[0], c[1]) for c in ohe.columns])
    df.columns = pd.MultiIndex.from_tuples([(c, "", "") for c in df.columns])
    agg_ohe.columns = pd.MultiIndex.from_tuples(
        [("agg_ohe", c, "") for c in agg_ohe.columns]
    )

    # Drop obs with null spend cats
    df = df.join(ohe, how="inner")
    df = df.join(agg_ohe, how="inner")

    # Add in count of categories selected
    df["cat_cnt"] = df["ohe"].sum(axis=1)
    df["agg_cat_cnt"] = df["agg_ohe"].sum(axis=1)

    # Normalize by number of selected categories from each respondent
    norm_ohe = df["ohe"].div(df["cat_cnt"].values, axis=0)
    norm_ohe.columns = pd.MultiIndex.from_tuples(
        [("norm_ohe", c[0], c[1]) for c in norm_ohe.columns]
    )
    norm_agg_ohe = df["agg_ohe"].div(df["agg_cat_cnt"].values, axis=0)
    norm_agg_ohe.columns = pd.MultiIndex.from_tuples(
        [("norm_agg_ohe", c[0], c[1]) for c in norm_agg_ohe.columns]
    )
    df = df.join(norm_ohe, how="inner")
    df = df.join(norm_agg_ohe, how="inner")

    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    df = df.drop("spending_categories", axis=1, level=0)
    return df


def prop_tbl_by_cut(
    df,
    cut_col,
    sum_cols,
    grp_disp_name=None,
    min_grp_cnt=1000,
    sort_output=True,
    abbr_col_names=True,
    prct=False,
):
    """
    Given a cut or category column and a set of columns to sum over, it
    calculates the proportion of sums within each category value. Also
    calculates an 'All other' row for category values with few observations.
    """
    df = df[[cut_col, sum_cols]].copy()
    sum_cols_lst = list(df[sum_cols].columns)
    # Hack for multi-index issues
    df.columns = [
        c[1] if c[0] in ("agg", "agg_ohe", "norm_agg_ohe") else c[0] for c in df.columns
    ]

    if grp_disp_name is None:
        grp_disp_name = cut_col

    if min_grp_cnt is not None:
        category_counts = df[cut_col].value_counts()
        categories_below_threshold = category_counts[
            category_counts < min_grp_cnt
        ].index
        df.loc[df[cut_col].isin(categories_below_threshold), cut_col] = "All other"

    row_cnts = df[cut_col].value_counts()
    grp = df.groupby(cut_col).sum()

    grp = grp.div(row_cnts, axis=0)
    if prct:
        grp *= 100
    grp["N"] = row_cnts

    grp = grp[["N"] + sum_cols_lst]
    if sort_output:
        grp = grp.sort_values("N", ascending=False)
    if "All other" in grp.index:
        grp = grp.loc[[c for c in grp.index if c != "All other"] + ["All other"]]
    if abbr_col_names:
        grp = grp.rename(columns=PROJECT_NAME_ABBREVIATIONS)

    grp.index.name = grp_disp_name

    return grp


def cnts_by_proj(df, min_prop=0.8, min_N=1000):
    """Calculates basic stats by project"""
    df = df.copy()
    start_projs = df["project_name"].unique()
    df["completed_date"] = df["completed_date"].dt.strftime("%Y-%m-%d")

    start_rows = len(df)
    df["non_null_pick_lst"] = ~df["spending_categories"].isnull()
    df["has_res_id"] = ~df["res_id"].isnull()

    quant_spend_cols = list(set([c for c in PICK_LST_TO_QUANTS_COLS.values()]))
    df["how_much_non_null"] = df[quant_spend_cols].notna().any(axis=1)

    grp = (
        df.groupby(["project_name"])
        .agg(
            N=("res_id", "size"),
            N_unique_rcp=("recipient_id", "nunique"),
            min_xfer_dt=("completed_date", "min"),
            max_xfer_dt=("completed_date", "max"),
            N_non_null_spend_cats=("non_null_pick_lst", "sum"),
            N_non_null_how_much=("how_much_non_null", "sum"),
            N_src_res_obj=("has_res_id", "sum"),
        )
        .sort_values("min_xfer_dt")
    )

    grp["Prop_non_null_spend_cats"] = grp["N_non_null_spend_cats"].div(grp["N"])
    grp["Included"] = (grp["Prop_non_null_spend_cats"] > min_prop) & (grp["N"] > min_N)

    return grp


def filter_projects_w_high_null_rates(df, min_prop=0.8, min_N=1000):
    """Filter out observations from projects with either a small number of observations or a high rate of nulls."""
    df = df.copy()
    start_projs = df["project_name"].unique()
    start_rows = len(df)
    df["non_null_pick_lst"] = ~df["spending_categories"].isnull()

    df["has_res_id"] = ~df["res_id"].isnull()

    grp = (
        df[["project_name", "non_null_pick_lst"]]
        .groupby("project_name")
        .agg(N=("non_null_pick_lst", "size"), non_null_cnt=("non_null_pick_lst", "sum"))
        .sort_values("non_null_cnt", ascending=False)
    )

    grp["prop"] = grp["non_null_cnt"].div(grp["N"])

    grp = grp[grp["prop"] > min_prop]
    grp = grp[grp["N"] > min_N]
    included_projs = grp.index

    df = df[df["project_name"].isin(included_projs)].copy()

    print(
        f"Dropping {start_rows - len(df)} rows from {len(start_projs) - len(included_projs)} projects that have a non-null response rate of less than {min_prop} OR a minimum observation count less than {min_N}."
    )

    print(
        f"The remaining dataset is {len(df)} rows with {len(included_projs)} projects."
    )
    return df.drop("non_null_pick_lst", axis=1)


def categories_by_response_rate(ohe: pd.DataFrame, name: str) -> pd.DataFrame:
    """Count and order categories by number of response rate"""

    summary_counts = pd.DataFrame(
        {
            "N": ohe.sum(axis=0).sort_values(ascending=False).round(0),
            "Prct": (ohe.sum(axis=0).sort_values(ascending=False) / len(ohe)),
        }
    )

    if summary_counts.index.nlevels == 2:
        summary_counts.index = summary_counts.index.rename(
            ["Agg. category", "Category"]
        )
    else:
        summary_counts.index = summary_counts.index.rename("Agg. category")

    RESULTS["xls_cnts"].append((name, summary_counts))
    return summary_counts


def add_xls_note():
    notes = [
        f"This analysis was generated on {datetime.datetime.now().strftime('%m/%d/%y')}.",
        f"This data set is a roll-up of responses to the Spending Categories question between October of 2019 and October of 2023."
        f"Projects with fewer than {MIN_PROJ_N} responses or a completion rate of less than {MIN_PROJ_PROP:.0%} in response to the Spending Categories question are excluded. Recipients who are ineligible, discarded, written off, and refused recipients were also filtered out."
        "N represents the number of respondents. Recipients who completed more than one transfer and follow-up survey within the analysis period are counted multiple times.",
        "Percentage columns show the raw percentage of respondents selecting a particular category. Note that most respondents select more than one category, so these percentages will add up to more than 100%.",
        "The Inverse Weighted (IW) percentage columns represent the inverse weighted percentage of respondents, or the percentage normalized based on the number of categories each respondent picked. For respondents who choose multiple categories, these columns encode the assumption that their transfer spending is split evenly among those selected categories.",
    ]

    RESULTS["xls_results"].append(("notes", pd.DataFrame(notes, columns=["Notes"])))


def dataset_desc_stats(df):
    """Overall descriptive stats on filtered dateset"""
    str_res = RESULTS["str_results"]
    str_res["cur_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    str_res["N_obs"] = int(round(len(df) / 10**3, 0))
    str_res["N_rcp"] = int(
        round(df.index.get_level_values("recipient_id").nunique() / 10**3, 0)
    )
    str_res["N_proj"] = df.project_name.nunique()
    str_res["N_countries"] = df.country.nunique()
    str_res["N_cats_with_resp"] = len(df["ohe"].columns)
    str_res["N_agg_cats"] = len(df["agg_ohe"].columns)


def number_of_cats_desc_stats(df):
    """descriptive stats about number of categories selected"""
    str_res = RESULTS["str_results"]
    diagnostics = RESULTS["diagnostics"]
    n_cats = df[["recipient_gender", "age_group", "cat_cnt"]]
    n_cats.columns = [c[0] for c in n_cats.columns]
    by_gender = n_cats.groupby(["recipient_gender"])["cat_cnt"].describe()
    by_age = n_cats.groupby(["age_group"])["cat_cnt"].describe()
    str_res["mean_number_of_cats"] = n_cats["cat_cnt"].mean().round(2)
    str_res["mean_number_of_cats_female"] = by_gender.loc["Female"]["mean"].round(2)
    str_res["mean_number_of_cats_male"] = by_gender.loc["Male"]["mean"].round(2)
    str_res["mean_number_of_cats_gender_diff"] = abs(
        by_gender.loc["Female"]["mean"] - by_gender.loc["Male"]["mean"]
    ).round(2)
    age_and_gen = (
        n_cats[n_cats["recipient_gender"].isin(["Male", "Female"])]
        .groupby(["age_group", "recipient_gender"])
        .describe()
    )
    diagnostics.append(("n_s_cats_by_age_gen", age_and_gen))
    expense_cats_plot(n_cats)
    male_cnts = n_cats[n_cats.recipient_gender == "Male"]["cat_cnt"]
    female_cnts = n_cats[n_cats.recipient_gender == "Female"]["cat_cnt"]
    t_statistic, p_value = stats.ttest_ind(male_cnts, female_cnts)
    str_res["p_val_diff_in_cnt_of_resp_by_gender"] = round(p_value, 3)
    str_res["t_stat_diff_in_cnt_of_resp_by_gender"] = round(t_statistic, 3)


def cut_by_proj(df, sum_cols):
    # Calculate by project
    by_proj = prop_tbl_by_cut(
        df,
        "project_name",
        sum_cols,
        grp_disp_name="Project",
        min_grp_cnt=None,
        sort_output=False,
    ).reset_index()

    # Hacky way to add country into index
    proj_to_country = (
        df[["project_name", "country"]]
        .droplevel(-1)
        .droplevel(-1)
        .drop_duplicates()
        .set_index("project_name")
        .to_dict(orient="dict")[("country", "")]
    )
    proj_to_country["All other"] = "All other"

    by_proj["Country"] = by_proj["Project"].map(proj_to_country)

    country_to_blank = {country: "" for country in df["country"].unique()}
    by_proj["Project"] = (
        by_proj["Project"].replace(country_to_blank, regex=True).str.strip()
    )
    return by_proj.set_index(["Country", "Project"])


def analyze_category_props_by_group(df):
    """
    Analyze the proportion of respondents in diff spending categories at
    the detailed and aggregated level by a number of different cuts.
    """
    # Setup res objs and
    str_res = RESULTS["str_results"]
    xls_res = RESULTS["xls_results"]
    diagnostics = RESULTS["diagnostics"]

    summary_counts = categories_by_response_rate(df["ohe"], "cats_by_respondent")
    summary_counts["IW Prct"] = categories_by_response_rate(
        df["norm_ohe"], "norm_ats_by_respondent"
    )["Prct"]
    num_w_over_1_prct = summary_counts[summary_counts.Prct > 0.01]
    num_w_under_1_prct = summary_counts[summary_counts.Prct <= 0.01]

    str_res[
        "top_response_categories_by_response_mdtbl"
    ] = num_w_over_1_prct.reset_index(names=["Agg. category", "Category"]).to_markdown(
        index=False, floatfmt=(None, None, ",.0f", ".1%", ".1%")
    )
    str_res["sel_by_over_1_prct"] = len(num_w_over_1_prct)
    str_res["total_sel_by_under_1_prct"] = num_w_under_1_prct["N"].sum()

    categories_by_response_rate(df["ohe"].loc[1, :], "cats_by_recipient")

    # Vice count
    str_res["vice_prct"] = df["ohe"][
        [("Other", c) for c in VICE_CATEGORIES]
    ].sum().sum() / len(df)

    # For rest of analysis, sum up to higher level categories
    ## Category props

    df.sort_index(axis=1, inplace=True)
    df = df.drop("ohe", axis=1, level=0)
    df.columns = df.columns.droplevel(-1)

    overall = categories_by_response_rate(df["agg_ohe"], "agg_cats_by_rcp")
    overall["IW Prct"] = categories_by_response_rate(
        df["norm_agg_ohe"], "norm_agg_cats_by_rsp"
    )["Prct"]
    overall = overall.sort_values("IW Prct", ascending=False)
    xls_res.append(("overall", overall))
    xls_res.append(("detailed_cats", summary_counts))

    top_5 = list(overall.index[:5])
    top_5_str = [n.lower() for n in top_5]

    str_res["prop_w_resp_in_top_5"] = df["agg_ohe"][top_5].max(axis=1).sum() / len(df)
    str_res["prop_in_top_5_cats"] = overall.head(5)["IW Prct"].sum()
    str_res["top_5_str"] = f"{', '.join(top_5_str[:4])}, and {top_5_str[4]}"
    str_res["top_aggregated_response_categories_mdtbl"] = overall.to_markdown(
        floatfmt=(None, ",.0f", ".1%", ".1%")
    )
    str_res["most_popular_category"] = overall.head(1).index.values[0]
    str_res["most_popular_category_lc"] = str_res["most_popular_category"].lower()
    str_res["most_popular_category_prct"] = overall.head(1)["Prct"].values[0]
    str_res["most_popular_category_iw_prct"] = overall.head(1)["IW Prct"].values[0]

    # # Calculate response counts by recipient
    by_rcp = categories_by_response_rate(df["agg_ohe"].loc[1], "agg_cats_by_rcp")
    by_rcp["IW Prct"] = categories_by_response_rate(
        df["norm_agg_ohe"].loc[1], "norm_agg_cats_by_rcp"
    )["Prct"]
    str_res["top_aggregated_response_categories_rcp_mdtbl"] = by_rcp.sort_values(
        "IW Prct", ascending=False
    ).to_markdown(floatfmt=(None, ",.0f", ".1%", ".1%"))

    # Calculate response counts by recipient who only picked one category
    categories_by_response_rate(
        df[df["agg_cat_cnt"] == 2]["agg_ohe"].loc[1], "agg_cats_single"
    )

    # Calculate by project
    by_proj = cut_by_proj(df, "agg_ohe")
    xls_res.append(("by_proj", by_proj))
    xls_res.append(("by_proj_iw", cut_by_proj(df, "norm_agg_ohe")))

    # By country
    by_country = prop_tbl_by_cut(df, "country", "agg_ohe", grp_disp_name="Country")
    xls_res.append(("by_country", by_country))
    xls_res.append(
        (
            "by_country_iw",
            prop_tbl_by_cut(df, "country", "norm_agg_ohe", grp_disp_name="Country"),
        )
    )

    # By project type
    by_proj_type = prop_tbl_by_cut(
        df, "project_type", "agg_ohe", grp_disp_name="Project Type"
    )
    xls_res.append(("by_proj_type", by_proj_type))
    xls_res.append(
        (
            "by_proj_type_iw",
            prop_tbl_by_cut(
                df, "project_type", "norm_agg_ohe", grp_disp_name="Project Type"
            ),
        )
    )

    # By year
    by_year = prop_tbl_by_cut(df, "year", "agg_ohe", grp_disp_name="Xfer year")
    xls_res.append(("by_year", by_year))
    xls_res.append(
        (
            "by_year_iw",
            prop_tbl_by_cut(df, "year", "norm_agg_ohe", grp_disp_name="Xfer year"),
        )
    )

    # By gender
    by_gender = prop_tbl_by_cut(
        df, "recipient_gender", "agg_ohe", grp_disp_name="Gender"
    )
    xls_res.append(("by_gender", by_gender))
    xls_res.append(
        (
            "by_gender_iw",
            prop_tbl_by_cut(
                df, "recipient_gender", "norm_agg_ohe", grp_disp_name="Gender"
            ),
        )
    )

    # By age bin
    by_age = prop_tbl_by_cut(
        df,
        "age_group",
        "agg_ohe",
        grp_disp_name="Age",
        min_grp_cnt=None,
        sort_output=False,
    )
    xls_res.append(("by_age", by_age))
    xls_res.append(
        (
            "by_age_iw",
            prop_tbl_by_cut(
                df,
                "age_group",
                "norm_agg_ohe",
                grp_disp_name="Age",
                min_grp_cnt=None,
                sort_output=False,
            ),
        )
    )


def demo_factor_analysis(df):
    """Do basic compositional analysis on Age / Gender"""

    # Setup res objs and
    str_res = RESULTS["str_results"]
    xls_res = RESULTS["xls_results"]
    diagnostics = RESULTS["diagnostics"]

    def extract_model_info(model, feature_names):
        coeffs = model.params[feature_names]
        std_err = model.bse[feature_names]
        p_values = model.pvalues[feature_names]
        return pd.DataFrame(
            {"Coefficient": coeffs, "Standard Error": std_err, "P-value": p_values}
        )

    df = df.copy()
    df.sort_index(axis=1, inplace=True)
    df = df.drop("ohe", axis=1, level=0)
    df.columns = df.columns.droplevel(-1)

    outcomes = df["agg_ohe"].columns
    rcp_cnts = df.loc[1][
        [
            "recipient_gender",
            "recipient_age_at_contact",
            "project_name",
            "agg_cat_cnt",
            "agg_ohe",
        ]
    ].copy()
    # Hack to flatten multi index
    rcp_cnts.columns = [c[1] if c[0] == "agg_ohe" else c[0] for c in rcp_cnts]

    # Drop non-binary genders
    rcp_cnts = rcp_cnts[rcp_cnts.recipient_gender.isin(["Female", "Male"])]

    # Drop obs with missing ages
    rcp_cnts = rcp_cnts[~rcp_cnts.recipient_age_at_contact.isnull()]

    # Standardize age coeff
    rcp_cnts["age"] = (
        rcp_cnts["recipient_age_at_contact"]
        - rcp_cnts["recipient_age_at_contact"].mean()
    ) / rcp_cnts["recipient_age_at_contact"].std()

    df_dummies = pd.get_dummies(rcp_cnts, columns=["recipient_gender"])

    cross_corr = df_dummies[
        ["recipient_gender_Female", "age", "agg_cat_cnt"] + list(outcomes)
    ].corr()
    print(cross_corr)

    res = {}
    fets = ["C(recipient_gender, Treatment('Male'))[T.Female]", "age"]

    # rcp_cnts[outcomes] = rcp_cnts[outcomes].div( rcp_cnts['agg_cat_cnt'].values, axis=0)
    for outcome in outcomes:
        formula = f"{outcome} ~  age + C(recipient_gender, Treatment('Male')) +  C(project_name)"

        model = smf.ols(formula=formula, data=rcp_cnts).fit(
            cov_type="cluster", cov_kwds={"groups": rcp_cnts["project_name"]}
        )
        print(model.summary())

        res[outcome] = extract_model_info(model, fets)
    res = pd.concat(res).round(3)

    res.index = res.index.rename(["Outcome", "Predictor"])

    res = res.reset_index()
    res["Predictor"] = res["Predictor"].replace(
        {
            "C(recipient_gender)[T.Male]": "male_rcp",
            "C(recipient_gender, Treatment('Male'))[T.Female]": "female_rcp",
        }
    )
    res["sv"] = abs(res["Coefficient"])
    str_res["lpm_by_predictors_sorted_mdtbl"] = (
        res[res["P-value"] < 0.05]
        .sort_values("sv", ascending=False)
        .drop("sv", axis=1)
        .head(5)
        .to_markdown(index=False, floatfmt=(None, None, ".3f", ".3f", ".3f"))
    )
    str_res["lpm_by_predictors_mdtbl"] = res.drop("sv", axis=1).to_markdown(index=False)


def dl_and_analyze_data():
    """
    Download and analyze data -- runs analysis pipeline.
    """

    # Count table rows
    check_prop_xfers_w_flup_survey()

    # Setup mapping dicts for switching between detailed and summary categories
    aggregate_to_detailed_spend_category = further_group_agg_spend_catigories(
        AGGREGATE_TO_DETAILED_SPEND_CATEGORY, OTHER_AGGREGATE_SPEND_CATIGORIES
    )
    category_aggregations_spend = mk_category_aggregations_spend(
        aggregate_to_detailed_spend_category, PICK_LST_TO_QUANTS_COLS
    )

    # DL and featureize the dataset
    cache_path = f"data_cache/data.pkl.gz"
    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        proj_report = pd.read_pickle("data_cache/proj_report.pkl")

    else:
        df = get_base_data()
        proj_report = cnts_by_proj(df, min_prop=MIN_PROJ_PROP, min_N=MIN_PROJ_N)
        proj_report.to_pickle("data_cache/proj_report.pkl")
        df = filter_projects_w_high_null_rates(
            df, min_prop=MIN_PROJ_PROP, min_N=MIN_PROJ_N
        )
        df = add_features(df, aggregate_to_detailed_spend_category)
        df.to_pickle(cache_path)

    # Run analysis
    dataset_desc_stats(df)
    number_of_cats_desc_stats(df)
    analyze_category_props_by_group(df)
    demo_factor_analysis(df)
    add_xls_note()
    RESULTS["diagnostics"].append(("by_project_cnts", proj_report))

    return RESULTS


if __name__ == "__main__":
    results = dl_and_analyze_data()
    import IPython

    IPython.embed()
