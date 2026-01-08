import pandas as pd


def _is_all_non_ascii(val) -> bool:
    if not isinstance(val, str) or val == "":
        return False
    return all(ord(ch) > 127 for ch in val)


def load_and_filter_csv(
    csv_path: str = "TMS_socials_December_2025.csv",
    keep_cols: list[str] = ["Network", "Content type", "Content"],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize headers
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # resolve column names safely
    cols_norm = {c.lower(): c for c in df.columns}

    def resolve(name: str) -> str:
        col = cols_norm.get(name.lower())
        if col is None:
            raise ValueError(f"Missing column '{name}'. Found: {df.columns.tolist()}")
        return col

    network_col = resolve(keep_cols[0])
    content_type_col = resolve(keep_cols[1])
    content_col = resolve(keep_cols[2])

    # keep only required columns
    df = df[[network_col, content_type_col, content_col]]

    # filter: keep comment in Content, exclude facebook in Network
    mask = (
        df[content_type_col].astype(str).str.lower().eq("comment")
        & ~df[network_col].astype(str).str.contains("facebook", case=False, na=False)
    )
    df = df.loc[mask]

    # drop rows where Content type is fully non-ASCII
    non_ascii_mask = df[content_type_col].apply(_is_all_non_ascii)
    df = df.loc[~non_ascii_mask]

    # drop Network column from output
    df = df.drop(columns=[network_col])

    return df


if __name__ == "__main__":
    comments_df = load_and_filter_csv()

    print(comments_df.head())

    comments_df.to_csv(
        "Yasmins_comments2.csv",
        index=False,
        encoding="utf-8",
    )
