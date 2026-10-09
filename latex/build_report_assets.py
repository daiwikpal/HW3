"""Build report assets from saved results: the full test-result table and the loss-curve figure."""

import shutil
from pathlib import Path

import pandas as pd

LATEX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LATEX_DIR.parent / "results"
MODELS = ["base", "lora_r8", "lora_r64"]
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
COLUMNS = (r"\textbf{qid} & \textbf{Category} & \textbf{Question} & \textbf{Benchmark best answer} & "
           r"\textbf{Target LLM selected answer} & \textbf{B} & \textbf{r8} & \textbf{r64} \\ \hline")
HEADER = rf"""{{\scriptsize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{longtable}}{{|p{{0.035\textwidth}}|p{{0.10\textwidth}}|p{{0.25\textwidth}}|p{{0.20\textwidth}}|p{{0.20\textwidth}}|c|c|c|}}
  \caption{{All 100 test queries with the benchmark best answer, the target LLM's selected MC1 answer (incorrect answers in red), and correctness for the target LLM (B), LoRA $r = 8$, and LoRA $r = 64$.}}\label{{tab:all-test-results}}\\
  \hline
  {COLUMNS}
  \endfirsthead
  \hline
  {COLUMNS}
  \endhead
"""
FOOTER = "\\end{longtable}\n}\n"


def tex(text):
    """Escape LaTeX special characters and convert straight double quotes to TeX quotes."""
    pieces, opening = [], True
    for char in str(text):
        if char == '"':
            pieces.append("``" if opening else "''")
            opening = not opening
        else:
            pieces.append(SPECIAL.get(char, char))
    return "".join(pieces)


def result_row(qid, evals):
    """Format one test query as a longtable row."""
    base = evals["base"].loc[qid]
    answer = tex(base.predicted_answer)
    answer = answer if base.correct else rf"\wrong{{{answer}}}"
    marks = " & ".join(r"\pass" if evals[name].loc[qid, "correct"] else r"\fail" for name in MODELS)
    return (f"  {qid} & {tex(base.category)} & {tex(base.question)} & {tex(base.best_answer)} & "
            f"{answer} & {marks} \\\\ \\hline")


def main():
    """Write appendix_test_results.tex and copy the saved loss curve into figures/."""
    evals = {name: pd.read_csv(RESULTS_DIR / f"eval_{name}.csv").set_index("qid") for name in MODELS}
    rows = [result_row(qid, evals) for qid in evals["base"].index]
    (LATEX_DIR / "appendix_test_results.tex").write_text(HEADER + "\n".join(rows) + "\n" + FOOTER)
    (LATEX_DIR / "figures").mkdir(exist_ok=True)
    shutil.copy2(RESULTS_DIR / "loss_curves.png", LATEX_DIR / "figures" / "loss_curves.png")


if __name__ == "__main__":
    main()
