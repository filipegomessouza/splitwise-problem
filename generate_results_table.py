import json

RESULTS_PATH = 'results/results.json'
INSTANCE_PREFIX = 'instancia_splitwise_'

HEADER = r"""\begin{table}[h]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{|r|rr|rr|rrr|rrr|rrr|rrr|}
\hline
\multirow{2}{*}{$I$} & \multicolumn{2}{c|}{$AC_G$} & \multicolumn{2}{c|}{$AC_R$} &
\multicolumn{3}{c|}{$BL_{FI}(AC_G)$} & \multicolumn{3}{c|}{$BL_{FI}(AC_R)$} &
\multicolumn{3}{c|}{$BL_{BI}(AC_G)$} & \multicolumn{3}{c|}{$BL_{BI}(AC_R)$} \\
\cline{2-17}
 & $sol$ & $t(s)$ & $sol$ & $t(s)$
 & $sol$ & $N_{It}$ & $t(s)$ & $sol$ & $N_{It}$ & $t(s)$
 & $sol$ & $N_{It}$ & $t(s)$ & $sol$ & $N_{It}$ & $t(s)$ \\
\hline"""

FOOTER_TEMPLATE = "\\hline\n\\end{{tabular}}%\n}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}"


def short_name(instance: str) -> str:
    return instance.removeprefix(INSTANCE_PREFIX).replace('_', r'\_')


def fmt_seconds(value: float) -> str:
    return f'{value:.3f}'


def build_row(row: dict) -> str:
    local_search = {
        'fi_g': row['greedy_fi_fitness'],
        'fi_r': row['random_key_fi_fitness'],
        'bi_g': row['greedy_bi_fitness'],
        'bi_r': row['random_key_bi_fitness'],
    }
    best = min(local_search.values())

    def cell(value: int) -> str:
        text = str(value)

        return f'\\textbf{{{text}}}' if value == best else text

    return (
        f"{short_name(row['instance'])} & "
        f"{row['greedy_fitness']} & {fmt_seconds(row['greedy_seconds'])} & "
        f"{row['random_key_fitness']} & {fmt_seconds(row['random_key_seconds'])} & "
        f"{cell(local_search['fi_g'])} & {row['greedy_fi_iterations']} & {fmt_seconds(row['greedy_fi_seconds'])} & "
        f"{cell(local_search['fi_r'])} & {row['random_key_fi_iterations']} & {fmt_seconds(row['random_key_fi_seconds'])} & "
        f"{cell(local_search['bi_g'])} & {row['greedy_bi_iterations']} & {fmt_seconds(row['greedy_bi_seconds'])} & "
        f"{cell(local_search['bi_r'])} & {row['random_key_bi_iterations']} & {fmt_seconds(row['random_key_bi_seconds'])} \\\\"
    )


def build_table(rows: list, caption: str, label: str) -> str:
    body = '\n'.join(build_row(row) for row in rows)

    footer = FOOTER_TEMPLATE.format(caption=caption, label=label)

    return '\n'.join([HEADER, body, footer])


def main() -> None:
    with open(RESULTS_PATH) as file:
        data = json.load(file)

    data.sort(key=lambda row: (row['n'], row['instance']))

    half = len(data) // 2
    first_half, second_half = data[:half], data[half:]

    print(build_table(
        first_half,
        caption=f"Resultados consolidados -- instâncias de {short_name(first_half[0]['instance'])} a {short_name(first_half[-1]['instance'])}.",
        label='tab:resultados-parte1',
    ))
    print()
    print(build_table(
        second_half,
        caption=f"Resultados consolidados -- instâncias de {short_name(second_half[0]['instance'])} a {short_name(second_half[-1]['instance'])}.",
        label='tab:resultados-parte2',
    ))


if __name__ == '__main__':
    main()
