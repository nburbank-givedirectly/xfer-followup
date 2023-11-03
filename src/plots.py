import matplotlib.pyplot as plt
import pandas as pd


def expense_cats_plot(n_spend_df: pd.DataFrame) -> None:
    means = (
        n_spend_df[n_spend_df["recipient_gender"].isin(["Male", "Female"])]
        .groupby(["recipient_gender", "age_group"])
        .mean()
    )
    std_errors = (
        n_spend_df[n_spend_df["recipient_gender"].isin(["Male", "Female"])]
        .groupby(["recipient_gender", "age_group"])
        .sem()
        * 2
    )

    fig, ax = plt.subplots()
    ax.plot(
        means.loc["Female"].index,
        means.loc["Female"],
        marker="o",
        label="Female recipients",
        color="blue",
    )
    ax.errorbar(
        means.loc["Female"].index,
        means.loc["Female"]["cat_cnt"],
        yerr=std_errors.loc["Female"]["cat_cnt"],
        fmt="none",
        ecolor="blue",
        capsize=4,
    )
    ax.plot(
        means.loc["Male"].index,
        means.loc["Male"],
        marker="o",
        label="Male recipients",
        color="green",
    )
    ax.errorbar(
        means.loc["Male"].index,
        means.loc["Male"]["cat_cnt"],
        yerr=std_errors.loc["Male"]["cat_cnt"],
        fmt="none",
        ecolor="green",
        capsize=4,
    )
    plt.title("Number of expense categories selected by recipient demographics")
    plt.xlabel("Recipient age at time of contact")
    plt.ylabel("Mean number of expense categories selected")
    plt.legend()
    plt.grid()
    plt.ylim(bottom=0)
    plt.savefig("output/cat_selected.png", dpi=250)


def expense_cats_by_proj_plot(n_cats: pd.DataFrame) -> None:
    mf_only = n_cats[n_cats["recipient_gender"].isin(["Male", "Female"])]
    by_proj = (
        mf_only.groupby(["proj_name", "recipient_gender"])["cat_cnt"]
        .describe()[["count", "mean"]]
        .unstack()
    )

    overall_by_proj = mf_only.groupby(["proj_name"])["cat_cnt"].describe()[
        ["count", "mean"]
    ]

    overall_by_proj.columns = pd.MultiIndex.from_tuples(
        [(c, "overall") for c in overall_by_proj.columns]
    )

    sems = mf_only.groupby(["proj_name", "recipient_gender"])["cat_cnt"].sem().unstack()
    sems.columns = pd.MultiIndex.from_tuples([("sem", c) for c in sems.columns])
    # sems = sems.fillna(0)
    by_proj = pd.concat([by_proj, sems, overall_by_proj], axis=1)
    by_proj = by_proj.sort_values(("mean", "overall"), ascending=False)

    means = by_proj["mean"]
    counts = by_proj["count"].copy()

    mf_only.groupby(["recipient_gender"])["cat_cnt"].describe()[
        ["count", "mean"]
    ].unstack()

    overall_gender = mf_only.groupby(["recipient_gender"])["cat_cnt"].describe()[
        ["count", "mean"]
    ]

    stacked = pd.concat([counts["Female"], counts["Male"]])
    counts["female_sizes"] = (counts["Female"] - stacked.mean()) / stacked.std()
    counts["male_sizes"] = (counts["Male"] - stacked.mean()) / stacked.std()

    counts["female_sizes"] = 10 + counts["female_sizes"] * 4.5 
    counts["male_sizes"] = 10 + counts["male_sizes"] * 4.5 


    fig, ax = plt.subplots()

    ax.scatter(
        means["Female"],
        means["Female"].index,
        marker="o",
        label="Female recipients",
        color="blue",
        facecolor="none",
        s=counts["female_sizes"].astype(float),
    )
    ax.axvline(
        x=overall_gender.loc["Female"]["mean"],
        label=f"Avg:{overall_gender.loc['Female']['mean']:.2f} categories",
        ls="dashed",
        color="blue",
        alpha=0.4,
    )

    # ax.errorbar(
    #     means["Female"],
    #     means["Female"].index,
    #     xerr=sems["sem"]["Female"] * 2,
    #     fmt="none",
    #     ecolor="blue",
    #     capsize=2,
    # )

    ax.scatter(
        means["Male"],
        means["Male"].index,
        marker="o",
        label="Male recipients",
        color="green",
        facecolor="none",
        s=counts["male_sizes"].astype(float),
    )
    ax.axvline(
        x=overall_gender.loc["Male"]["mean"],
        label=f"Avg:{overall_gender.loc['Male']['mean']:.2f} categories",
        ls="dashed",
        color="green",
        alpha=0.4,
    )
    plt.xlim(left=1)
    # ax.errorbar(
    #     means["Male"],
    #     means["Male"].index,
    #     xerr=sems["sem"]["Male"] * 2,
    #     fmt="none",
    #     ecolor="green",
    #     capsize=2,
    # )

    plt.title("Number of expense categories selected by project")
    plt.ylabel("Project")
    plt.xlabel("Mean number of expense categories selected")
    plt.legend(prop={"size": 6})
    plt.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=6)
    plt.tight_layout()
    plt.savefig("output/cat_selected_by_proj.png", dpi=250)
