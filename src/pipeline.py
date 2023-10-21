import subprocess
import pandas as pd
from analysis import dl_and_analyze_data


def gen_results_md(results, name) -> None:
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
            df.to_excel(
                writer, sheet_name=sheet_name[:31], index=True, float_format="%.3f"
            )
            if also_tsv:
                df.to_csv(f"output/tsvs/{output_str}_{sheet_name}.tsv", sep="\t")


if __name__ == "__main__":
    results = dl_and_analyze_data()
    name = "full"
    gen_results_md(results["str_results"], name)
    gen_excel(results["xls_results"], name)
    gen_excel(results["diagnostics"], "diagnostics")
    gen_excel(results["xls_cnts"], "cat_cnts")

    subprocess.run(
        [
            "pandoc",
            "-f",
            "markdown-auto_identifiers",
            "output/output_full.md",
            "-o",
            "output/output_full.docx",
            "--reference-doc=writeup/custom-reference.docx",
            "--extract-media", "./figures"
        ]
    )
    # df = df[df.rcpnt_fu_num == 1].copy()
