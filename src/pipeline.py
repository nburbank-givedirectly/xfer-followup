"""Run analysis and generate output report via pandoc"""
import subprocess
import pandas as pd
import os
from datetime import datetime
from analysis import dl_and_analyze_data


def escape_brackets(s):
    """Escape brackets within math sections with markdown docs from replacement"""
    start_delimiter = "$$"
    end_delimiter = "$$"

    escaped_lines = []
    inside_delimiters = False

    lines = s.split("\n")
    for line in lines:

        if line.startswith(start_delimiter) and not inside_delimiters:
            inside_delimiters = True
            escaped_lines.append(line)
        elif line.startswith(end_delimiter):
            inside_delimiters = False
            escaped_lines.append(line)
        elif inside_delimiters or (line.startswith('#') and line.endswith('}')):
            escaped_line = line.replace("{", "{{").replace("}", "}}")
            escaped_lines.append(escaped_line)
        else:
            escaped_lines.append(line)

    return "\n".join(escaped_lines)


def gen_results_md(results, name) -> None:
    """Stich together markdown results file with string results"""
    with open("writeup/writeup.md", "r") as file:
        templete = file.read()

    output_str = escape_brackets(templete)
    output_str = output_str.format(**results)

    with open(f"output/{name}.md", "w") as file:
        file.write(output_str)


def gen_excel(xls_results, output_str, also_tsv=True):
    """Make excel file with one table per sheet."""

    with pd.ExcelWriter(f"output/{output_str}.xlsx", engine="xlsxwriter") as writer:
        percent_format = writer.book.add_format({"num_format": "0.0%"})
        int_format = writer.book.add_format({"num_format": "#,##0"})
        float_format = writer.book.add_format({"num_format": "#,##0.00"})
        for sheet_name, df in xls_results:
            df.to_excel(
                writer, sheet_name=sheet_name[:31], index=True, merge_cells=False
            )

            worksheet = writer.book.get_worksheet_by_name(sheet_name)
            worksheet.autofit()
            if "Agriculture" in df.columns:
                start = df.reset_index().columns.get_loc("Agriculture")
                end = df.reset_index().columns.get_loc("Savings")
                worksheet.set_column(start, end, 8, percent_format)

            if "N" in df.columns:
                start = df.reset_index().columns.get_loc("N")
                worksheet.set_column(start, start, None, int_format)

            if "Prct" in df.columns:
                start = df.reset_index().columns.get_loc("Prct")
                worksheet.set_column(start, start, 8, percent_format)

            if "IW Prct" in df.columns:
                start = df.reset_index().columns.get_loc("IW Prct")
                worksheet.set_column(start, start, 8, percent_format)

            if also_tsv:
                if not os.path.exists("output/tsvs"):
                    os.makedirs("output/tsvs")
                df.to_csv(f"output/tsvs/{output_str}_{sheet_name}.tsv", sep="\t")


if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")
    results = dl_and_analyze_data()
    name = f"{datetime.now().date().strftime('%Y%m%d')}_xfer_flup"
    gen_results_md(results["str_results"], name)
    gen_excel(results["xls_results"], name)
    gen_excel(results["diagnostics"], "diagnostics")
    gen_excel(results["xls_cnts"], "cat_cnts")

    subprocess.run(
        [
            "pandoc",
            "-f",
            "markdown-auto_identifiers",
            f"output/{name}.md",
            "-o",
            f"output/{name}.docx",
            "--reference-doc=writeup/custom-reference.docx",
            "--extract-media",
            "./figures",
            "--filter", "pandoc-docx-pagebreakpy"
        ]
    )
