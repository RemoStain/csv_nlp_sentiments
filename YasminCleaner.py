import pandas as pd


def _is_all_non_ascii(val) -> bool:
    if not isinstance(val, str) or val == "":
        return False
    return all(ord(ch) > 127 for ch in val)


def load_and_filter_csv(
    csv_path: str = "Yasmins_data.csv",
    keep_cols: list = ["Content type", "Content"],
    filter_row: str = "comment",
) -> pd.DataFrame:
    """
    Load a CSV file and filter rows based on a specific column value,
    then remove rows where col1 is fully non-ASCII.
    """
    df = pd.read_csv(csv_path)

    df = df[keep_cols]

    df = df[df[keep_cols[0]].str.contains(filter_row, case=False, na=False)]

    # Drop rows where ONLY col1 is fully non-ASCII
    non_ascii_mask = df[keep_cols[1]].apply(_is_all_non_ascii)

    df = df[~non_ascii_mask]

    return df


if __name__ == "__main__":
    csv_path = "Yasmins_data.csv"
    keep_cols = ["Content type", "Content"]

    comments_df = load_and_filter_csv(csv_path, keep_cols)

    print(comments_df.head())

    comments_df.to_csv(
        "Yasmins_comments.csv",
        index=False,
        encoding="utf-8",
        sep=",",
    )
