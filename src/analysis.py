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


RESULTS = {
    "str_results": {},
    "xls_results": [],
    "xls_cnts": [],
    "diagnostics": [],
}


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
    with open("queries/prop_xfers_w_flup.sql", "r", encoding="utf-8") as file:
        query = file.read()
    df_cnts = get_df(query=query, limit=0, print_q=False)
    RESULTS["str_results"]["prop_xfers_w_flup"] = (
        (df_cnts["prop_overall"] * 100).values[0].round(1)
    )
    RESULTS["diagnostics"].append(("xfers_and_flup_cnts", df_cnts))
    print(df_cnts)


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
        f"Reducing to one followup per transfer and one research row per followup results in {end_len} rows."
    )

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
        .groupby(level=["rcpnt_fu_num","recipient_id", "transfer_id", "fu_id"])
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
    # df["project_name"] = (
    #     df["project_name"].replace(PROJECT_NAME_ABBREVIATIONS, regex=True).str.strip()
    # )

    bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 150]
    labels = [
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
    prct=True,
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

    grp = grp.div(row_cnts, axis=0)
    if prct:
        grp *= 100
    grp["N"] = row_cnts
    grp = grp[(["N"] + sum_cols)]
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
    str_results = RESULTS["str_results"]
    start_projs = df["project_name"].unique()
    start_rows = len(df)
    df["non_null_pick_lst"] = ~df["spending_categories"].isnull()

    df["has_res_id"] = ~df["res_id"].isnull()

    non_null_by_projet = prop_tbl_by_cut(
        df[["project_name", "non_null_pick_lst"]],
        "project_name",
        ["non_null_pick_lst"],
        min_grp_cnt=None,
        prct=False,
    ).sort_values("non_null_pick_lst", ascending=False)

    non_null_by_projet = non_null_by_projet[
        non_null_by_projet["non_null_pick_lst"] > min_prop
    ]
    non_null_by_projet = non_null_by_projet[non_null_by_projet["N"] > min_N]

    projs_to_include = non_null_by_projet.index

    df = df[df["project_name"].isin(projs_to_include)].copy()

    print(
        f"Dropping {start_rows - len(df)} rows from { len(start_projs) - len(projs_to_include)} projects that have a non-null response rate of less than {min_prop} OR a minimum observation count less than {min_N}."
    )

    str_results["N_obs"] = int(round(len(df) / 10**3, 0))
    str_results["N_rcp"] = int(round(df.recipient_id.nunique() / 10**3, 0))
    str_results["N_proj"] = df.project_name.nunique()
    str_results["N_countries"] = df.country.nunique()

    print(
        f"The remaining dataset is {len(df)} rows with {len(projs_to_include)} projects."
    )
    return df.drop("non_null_pick_lst", axis=1)


def compositional_analysis(cnts):
    """Do basic compositional analysis on Age / Gender"""

    def extract_model_info(model, feature_names):
        coeffs = model.params[feature_names]

        std_err = model.bse[feature_names]
        p_values = model.pvalues[feature_names]
        return pd.DataFrame(
            {"Coefficient": coeffs, "Standard Error": std_err, "P-value": p_values}
        )

    outcomes = [
        "Agriculture",
        "Education",
        "Entrepreneurship",
        "Food",
        "Healthcare",
        "Household",
        "Housing",
        "Livestock",
        "Other",
        "Savings",
    ]

    rcp_cnts = cnts.loc[1].copy()

    # Drop non-binary genders
    rcp_cnts = rcp_cnts[rcp_cnts.recipient_gender.isin(["Female", "Male"])]

    # Drop missing aggs
    rcp_cnts = rcp_cnts[~rcp_cnts.recipient_age_at_contact.isnull()]

    rcp_cnts["age"] = (
        rcp_cnts["recipient_age_at_contact"]
        - rcp_cnts["recipient_age_at_contact"].mean()
    ) / rcp_cnts["recipient_age_at_contact"].std()

    df_dummies = pd.get_dummies(rcp_cnts, columns=["recipient_gender"])

    cross_corr = df_dummies[
        ["recipient_gender_Female", "age", "n_spend_cats"] + outcomes
    ].corr()
    print(cross_corr)

    res = {}
    fets = ["C(recipient_gender, Treatment('Male'))[T.Female]", "age"]
    for outcome in outcomes:
        formula = f"{outcome} ~ n_spend_cats + age + C(recipient_gender, Treatment('Male'))  +  C(project_name)"

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
    print(res.sort_values("sv", ascending=False).drop("sv", axis=1))

    RESULTS["str_results"]["lpm_by_predictors_mdtbl"] = res.to_markdown(index=False)


def categories_by_response_rate(ohe: pd.DataFrame, name: str) -> pd.DataFrame:
    """Count and order categories by number of response rate"""

    summary_counts = pd.DataFrame(
        {
            "N": ohe.sum(axis=0).sort_values(ascending=False),
            "Prct": (ohe.sum(axis=0).sort_values(ascending=False) / len(ohe))
            * 100,
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


def run_analysis(df, name, aggregate_to_detailed_spend_category):
    """
    Calculates one hot encoded values, then calculates the
    proportion of those one hot encoded values over a number of cut
    categories. It finds a dict containing a bloat, a dictionary of
    markdown tables, and a dictionary of pandas data frames to be
    later converted into excel files.
    """
    str_results = RESULTS["str_results"]
    str_results["cur_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    xls_results = RESULTS["xls_results"]
    diagnostics = RESULTS["diagnostics"]

    # Calc One-hot encoded counts
    ohe_chache_path = f"data_cache/{name}_ohe.pq"
    if os.path.exists(ohe_chache_path):
        ohe = pd.read_parquet(ohe_chache_path)
    else:
        ohe = split_and_ohe_str_lst(
            df, "spending_categories", aggregate_to_detailed_spend_category, agg=False
        )
        ohe.to_parquet(ohe_chache_path)

    # descriptive stats about number of categories selected
    n_spending_cats_selected = ohe.sum(axis=1)
    n_spending_cats_selected.name = "n_spend_cats"

    n_spend_df = pd.merge(
        df[["recipient_gender", "age_group"]],
        n_spending_cats_selected,
        left_index=True,
        right_index=True,
        how="inner",
    )

    by_gender = n_spend_df.groupby(["recipient_gender"]).describe()
    by_age = n_spend_df.groupby(["age_group"]).describe()

    str_results["mean_number_of_categories"] = n_spending_cats_selected.mean().round(2)
    str_results["mean_number_of_categories_female"] = by_gender.loc["Female"][("n_spend_cats", "mean")].round(2)
    str_results["mean_number_of_categories_male"] = by_gender.loc["Male"][("n_spend_cats", "mean")].round(2)
    str_results["mean_number_of_categories_gender_diff"] = abs(by_gender.loc["Female"][("n_spend_cats", "mean")] - by_gender.loc["Male"][("n_spend_cats", "mean")]).round(2)

    diagnostics.append(
        (
            "n_s_cats_by_age_gen",
            n_spend_df[n_spend_df["recipient_gender"].isin(["Male", "Female"])]
            .groupby(["age_group", "recipient_gender"])
            .describe(),
        )
    )
    expense_cats_plot(n_spend_df)

    male_cnts = n_spend_df[n_spend_df.recipient_gender == "Male"]["n_spend_cats"]
    female_cnts = n_spend_df[n_spend_df.recipient_gender == "Female"]["n_spend_cats"]
    t_statistic, p_value = stats.ttest_ind(male_cnts, female_cnts)
    str_results["p_val_diff_in_cnt_of_resp_by_gender"] = round(p_value, 3)
    str_results["t_stat_diff_in_cnt_of_resp_by_gender"] = round(t_statistic, 3)

    # ohe = ohe[n_spend_df['n_spend_cats'] == 1].copy()

    summary_counts = categories_by_response_rate(ohe, "cats_by_respondent")

    num_w_over_1_prct = summary_counts[summary_counts.Prct > 1]
    num_w_under_1_prct = summary_counts[summary_counts.Prct <= 1]

    RESULTS["str_results"]["top_response_categories_by_response_mdtbl"] = (
        num_w_over_1_prct.reset_index(names=["Agg. category", "Category"])
        .round(1)
        .to_markdown(index=False)
    )
    str_results["top_response_categories_note_by_response"] = f"Among the original categories, there were {len(summary_counts)} that received responses, with only {len(num_w_over_1_prct)} response types that were selected by over 1% of respondents. All other responses only contributed for {num_w_under_1_prct['N'].sum()} responses."

    import IPython; IPython.embed()

    categories_by_response_rate(ohe.loc[1, :], "cats_by_recipient")

    # Vice count
    vice_prct = (
        ohe[[("Other", c) for c in VICE_CATEGORIES]].sum().sum() / len(ohe)
    ) * 100
    str_results["vice_prct"] = round(vice_prct, 2)

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
        "recipient_age_at_contact",
    ]

    cnts = df[fet_cols].join(ohe, how="inner")

    ## Category props

    overall = categories_by_response_rate(ohe, "agg_cats_by_rsp")
    top_5 = list(overall.index[:5])
    prop_w_resp_in_top_5 = cnts[top_5].max(axis=1).sum() / len(cnts)
    top_5_str = [n.lower() for n in top_5]

    str_results["prop_w_resp_in_top_5"] = f"{prop_w_resp_in_top_5*100:.1f}"
    str_results["top_5_str"] = f"{', '.join(top_5_str[:4])}, and {top_5_str[4]}"
    note = f"{prop_w_resp_in_top_5*100:.1f}% of surveyed recipients indicating that they spent at least part of their transfer on one or more of {', '.join(top_5_str[:4])}, and {top_5_str[4]} expenses."
    str_results["top_aggregated_response_categories_mdtbl"] = overall.round(
        1
    ).to_markdown()
    str_results["top_aggregated_response_categories_note"] = note
    str_results["most_popular_category"] = overall.head(1).index.values[0]
    str_results["most_popular_category_prct"] = round(
        overall.head(1)["Prct"].values[0], 1
    )

    # Calculate response counts by recipient
    categories_by_response_rate(ohe.loc[1], "agg_cats_by_rcp")

    sum_cols = list(overall.index)

    # Calculate by project
    by_proj = prop_tbl_by_cut(
        cnts, "project_name", sum_cols, grp_disp_name="Project", min_grp_cnt=None
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

    # By country
    by_country = prop_tbl_by_cut(cnts, "country", sum_cols, grp_disp_name="Country")
    str_results["by_country_mdtbl"] = by_country.round(1)
    xls_results.append(("by_country", by_country))

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
    xls_results.append(("by_age", by_age))

    full_map_of_category_aggregations = pd.DataFrame(
        [
            (k, col)
            for k, cols in aggregate_to_detailed_spend_category.items()
            for col in cols
        ],
        columns=["Agg. category", "Category"],
    ).set_index("Agg. category")
    str_results[
        "full_map_of_category_aggregations_mdtbl"
    ] = full_map_of_category_aggregations.to_markdown()
    diagnostics.append(("category_aggregations", full_map_of_category_aggregations))

    cnts["n_spend_cats"] = n_spend_df["n_spend_cats"]
    compositional_analysis(cnts)


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

    proj_report = cnts_by_proj(df, min_prop=0.8, min_N=1000)
    df = filter_projects_w_high_null_rates(df, min_prop=0.8, min_N=1000)
    df = df.set_index(["rcpnt_fu_num","recipient_id", "transfer_id", "fu_id"])
    df = add_features(df)

    run_analysis(df, "full", aggregate_to_detailed_spend_category)
    RESULTS["diagnostics"].append(("by_project_cnts", proj_report))

    return RESULTS


if __name__ == "__main__":
    results = dl_and_analyze_data()
    import IPython

    IPython.embed()
