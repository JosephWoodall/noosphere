import pandas as pd
import os

try:
    tables = pd.read_html('noosphere_report.html')
    with open('tasks/tables.tex', 'w') as f:
        for i, df in enumerate(tables):
            f.write(f"% Table {i+1}\n")
            f.write(df.to_latex(index=False))
            f.write("\n\n")
    print(f"Successfully extracted {len(tables)} tables to tasks/tables.tex")
except Exception as e:
    print(f"Error: {e}")
