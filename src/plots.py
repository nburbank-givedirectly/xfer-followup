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
