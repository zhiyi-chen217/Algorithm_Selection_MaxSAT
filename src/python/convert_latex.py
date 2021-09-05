import pandas as pd
import os

def dfToTable(df):
    columns = list(map(str, df.columns))
    col_format = "|"
    newline = " \t\t\\hline\n"
    for c in columns:
        col_format = col_format + "c|"
    preamble = "\\begin{table}[!h]\n" \
               "\t\\centering\n" \
               "\t\\begin{tabular}{" + col_format + "}\n" \
               + newline + "\t\t"\
               + " & ".join(columns) + "\\\\\n"

    table = newline
    for ind in df.index:
        row = list(map("{:.6f}".format, df.loc[ind][1:]))
        row = [str(df.loc[ind][0])] + row
        temp = "\t\t" + " & ".join(row) + "\\\\\n"
        table += temp
        table += newline

    end = "\t\\end{tabular}\n" \
          "\t\\caption{}\n" \
          "\t\\label{}\n" \
          "\\end{table}"
    return preamble + table + end


os.chdir("../../results")
df = pd.read_csv("evaluate_result_direct_3.csv")
result = dfToTable(df)
