import pandas as pd


def load_and_filter_csv(
    csv_path: str = "Yasmins_data.csv",
    keep_cols: list = ["Content type", "Content"],
    filter_row: str = "comment",
) -> pd.DataFrame:
    """
    Load a CSV file and filter rows based on a specific column value.

    Args:
        csv_path (str): Path to the CSV file. (default: "Yasmins_data.csv")
        keep_cols (list): List of columns to keep. (default: ["Content type", "Content"])
        filter_row (str): The string to filter rows by in the first keep_col. (default: "comment")

    Returns:
        pd.DataFrame: Filtered DataFrame with only comments.
    """
    df = pd.read_csv(csv_path)

    df = df[keep_cols]

    filtered_df = df[df[keep_cols[0]].str.contains(filter_row, case=False, na=False)]

    return filtered_df


if __name__ == "__main__":
    # Example usage
    csv_path = "Yasmins_data.csv"
    keep_cols = ["Content type", "Content"]
    comments_df = load_and_filter_csv(csv_path, keep_cols)
    print(comments_df.head())
    comments_df.to_csv(
        "Yasmins_comments.csv",
        index=False,
        encoding="utf-8",  # explicit encoding
        sep=",",  # delimiter
    )