"""Run analysis and generate output report via pandoc"""
import subprocess
import pandas as pd
import os
from datetime import datetime
from make_base_query import mk_base_query, mk_val_query
from analysis import dl_and_analyze_data


START_DT = "2019-10-01"
END_DT = "2023-10-01"


def escape_brackets(templete: str) -> str:
    """Escape brackets within math sections with markdown docs from replacement"""
    start_delimiter = "$$"
    end_delimiter = "$$"

    escaped_lines = []
    inside_delimiters = False

    lines = templete.split("\n")
    for line in lines:
        if line.startswith(start_delimiter) and not inside_delimiters:
            inside_delimiters = True
            escaped_lines.append(line)
        elif line.startswith(end_delimiter):
            inside_delimiters = False
            escaped_lines.append(line)
        elif inside_delimiters or (line.startswith("#") and line.endswith("}")):
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


def number_to_alphabet(num):
    if 1 <= num <= 26:
        return chr(num + 64)
    else:
        return None

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
                worksheet.set_column(start, end, 12, percent_format)

                i = 1
                for col in df.reset_index().columns[start:end+1]:
                    col_let = number_to_alphabet(start + i)
                    i += 1
                    if col_let:
                        worksheet.conditional_format(f'{col_let}1:{col_let}{len(df)+1}', {'type': '3_color_scale', 'mid_color':'white', 'max_color':'#A9CCE3', 'min_color':'#E6B0AA'})

            if "N" in df.columns:
                start = df.reset_index().columns.get_loc("N")
                worksheet.set_column(start, start, None, int_format)

            if "Prct" in df.columns:
                start = df.reset_index().columns.get_loc("Prct")
                worksheet.set_column(start, start, 12, percent_format)

            for col in df.columns:
                if "Prct" in col:
                    start = df.reset_index().columns.get_loc(col)
                    worksheet.set_column(start, start, 12, percent_format)


            if "IW Prct" in df.columns:
                start = df.reset_index().columns.get_loc("IW Prct")
                worksheet.set_column(start, start, 12, percent_format)

            if "Notes" in df.columns:
                start = df.reset_index().columns.get_loc("Notes")
                worksheet.set_column(start, start, 80)

            if also_tsv:
                if not os.path.exists("output/tsvs"):
                    os.makedirs("output/tsvs")
                df.to_csv(f"output/tsvs/{output_str}_{sheet_name}.tsv", sep="\t")


if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")

    if not os.path.exists("queries"):
        os.makedirs("queries")

    if not os.path.exists("data_cache"):
        os.makedirs("data_cache")

    mk_base_query(filename="queries/base_query")
    mk_val_query(filename="queries/prop_xfers_w_flup")
    results = dl_and_analyze_data()
    name = f"{datetime.now().date().strftime('%Y%m%d')}_xfer_flup"
    gen_results_md(results["str_results"], name)
    gen_excel(results["xls_results"], name)

    # Full break down, non aggregated categories
    gen_excel(
        [(n, df) for (n, df) in results["diagnostics"] if n == "full_cnt_by_proj"]
        + [(n, df) for (n, df) in results["xls_results"] if n == "notes"],
        f"{datetime.now().date().strftime('%Y%m%d')}_full_proj_cnts_xfer_flup",
    )

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
            "--filter",
            "pandoc-docx-pagebreakpy",
        ]
    )
