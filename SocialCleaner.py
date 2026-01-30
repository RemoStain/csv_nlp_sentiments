import pandas as pd
import nlp.config_cleaner as config


class CsvCommentCleaner:
    def __init__(self) -> None:
        self._validate_config()

    def _validate_config(self) -> None:
        if not config.KEEP_COLUMNS or len(config.KEEP_COLUMNS) < 2:
            raise ValueError("KEEP_COLUMNS must contain at least two column names.")
        if not isinstance(config.FILTER_ROW, str) or config.FILTER_ROW == "":
            raise ValueError("FILTER_ROW must be a non-empty string.")
        if not isinstance(config.DEFAULT_DATAFILE, str) or config.DEFAULT_DATAFILE == "":
            raise ValueError("DEFAULT_DATAFILE must be a non-empty string.")

    @staticmethod
    def _is_all_non_ascii(val: object) -> bool:
        if not isinstance(val, str) or val == "":
            return False
        return all(ord(ch) > 127 for ch in val)

    def load(self) -> pd.DataFrame: 
        return pd.read_csv(config.DEFAULT_DATAFILE)

    def filter_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_cols = config.KEEP_COLUMNS
        filter_row = config.FILTER_ROW

        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")

        df = df[keep_cols]
        df = df[df[keep_cols[0]].str.contains(filter_row, case=False, na=False)]

        non_ascii_mask = df[keep_cols[1]].apply(self._is_all_non_ascii)
        df = df[~non_ascii_mask]

        return df

    def run(self) -> pd.DataFrame:
        df = self.load()
        return self.filter_comments(df)

    def save(self, df: pd.DataFrame) -> None:
        df.to_csv(
            config.OUTPUT_PATH,
            index=config.OUTPUT_INDEX,
            encoding=config.OUTPUT_ENCODING,
            sep=config.OUTPUT_SEPARATOR,
        )
        print("Saved to: " + config.OUTPUT_PATH)


if __name__ == "__main__":
    cleaner = CsvCommentCleaner()

    comments_df = cleaner.run()
    print(comments_df.head())

    cleaner.save(comments_df)
