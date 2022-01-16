import pandas as pd
import os

COLS = [
    ["hidden_size", "roc_auc"],
    ["hidden_size", "pr_auc"],
]


def make_result_score_df(df, cols=COLS):
    result_df = pd.DataFrame(pd.DataFrame({"hidden_size": [2, 4, 8]}))
    for cols in COLS:
        base_df = (
            df.loc[df["trainer_name"] == "BaseTrainer", cols]
            .sort_values(by=cols, ascending=False)
            .reset_index(drop=True)
            .drop_duplicates("hidden_size", keep="first")
            .reset_index(drop=True)
            .rename(columns={cols[1]: "Base_" + cols[1]})
        )

        new_df = (
            df.loc[df["trainer_name"] == "NewTrainer", cols]
            .sort_values(by=cols, ascending=False)
            .reset_index(drop=True)
            .drop_duplicates("hidden_size", keep="first")
            .reset_index(drop=True)
            .rename(columns={cols[1]: "New_" + cols[1]})
        )

        tmp_df = pd.merge(base_df, new_df, on="hidden_size")
        result_df = pd.merge(result_df, tmp_df, on="hidden_size")
    return result_df


if __name__ == "__main__":
    PATH = "../run_results_tabular/"
    file_list = os.listdir(PATH)
    file_list_py = [file for file in file_list if file.endswith(".csv")]
    for file_name in file_list_py:
        df = pd.read_csv(PATH + file_name)
        result_df = make_result_score_df(df)
        result_df.to_csv("../best_score/" + "best_score_" + file_name, index=False)
