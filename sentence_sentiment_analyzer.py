from itertools import tee
from typing import Iterable, Optional
from pathlib import Path

import pandas as pd
import tkinter as tk
import tkinter.font as tkFont
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import config_cleaner as config


col_name = "Content"
limit = 2500


def compound_to_100(score: float) -> float:
    return (score + 1) * 50.0

def build_analyzer() -> SentimentIntensityAnalyzer:
    sid = SentimentIntensityAnalyzer()
    sid.lexicon.update(
        {
            "donated": 0.25,
            "carney": -0.25,
        }
    )
    return sid


class SentimentViewer:
    def __init__(
        self,
        sid: SentimentIntensityAnalyzer,
        title: str = "Sentiment Viewer",
        width: int = 80,
        height: int = 25,
        console_output: bool = False,
    ):
        self.console_output = console_output
        self.sid = sid
        self.root = tk.Tk()
        self.root.title(title)

        self.custom_font = tkFont.Font(family="Courier", size=24, weight="bold")

        self.text_widget = tk.Text(
            self.root,
            font=self.custom_font,
            wrap="word",
            width=width,
            height=height,
            bg="#000000",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
        )
        self.scrollbar = tk.Scrollbar(self.root, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)

        self.text_widget.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.analyzed_sentences: list[str] = []

    @staticmethod
    def value_to_hex(val: float) -> str:
        if val == 0:
            return "#0066ff"
        r = int(max(0, 255 * (1 - (val + 1) / 2)))
        g = int(max(0, 255 * ((val + 1) / 2)))
        b = 0
        return f"#{r:02X}{g:02X}{b:02X}"

    def add_line(
        self,
        sentence: str,
        score: Optional[float] = None,
        colour_override: str = "",
    ) -> None:
        if score is None:
            try:
                score = self.sid.polarity_scores(sentence)["compound"]
                colour = self.value_to_hex(score)
            except Exception as e:
                score = 0.0
                colour = "#FFC0CB"
                sentence += f" [Error analyzing sentiment: {e}]"
        else:
            colour = self.value_to_hex(score)

        if len(colour_override) == 7:
            colour = colour_override

        if self.console_output:
            print(sentence)
            print(f"compound: {score}, colour: {colour}\n")

        sentence = f"{sentence}  [compound: {score}]"
        self.analyzed_sentences.append(sentence)

        self.text_widget.insert("end", sentence + "\n")

        line_number = int(self.text_widget.index("end-1c").split(".")[0]) - 1
        line_start = f"{line_number}.0"
        line_end = f"{line_number}.end"
        tag_name = f"line{line_number}"

        self.text_widget.tag_add(tag_name, line_start, line_end)
        self.text_widget.tag_config(tag_name, foreground=colour)
        self.text_widget.see("end")

    def run(self) -> None:
        self.root.mainloop()


def _normalize_network_filter(network_filter: Optional[str]) -> Optional[str]:
    if network_filter is None:
        return None

    allowed = {n.casefold(): n for n in getattr(config, "NETWORKS", []) or []}
    key = network_filter.strip().casefold()

    if allowed and key not in allowed:
        return ""

    return allowed.get(key, network_filter.strip())


def load_csv_sentences(
    file_name: str = "data.csv",
    col_name: str = "Content",
    chunksize: int = 50000,
    limit: int = 0,
    multiline_bucket: Optional[list[str]] = None,
    treat_blankline_as_paragraph: bool = False,
    network_col: str = "Network",
    network_filter: Optional[str] = None,
) -> Optional[Iterable[str]]:
    if multiline_bucket is None:
        multiline_bucket = []

    network = _normalize_network_filter(network_filter)
    usecols = [col_name] if network is None else [network_col, col_name]

    try:
        _ = next(
            pd.read_csv(
                file_name,
                usecols=usecols,
                encoding="utf-8",
                encoding_errors="ignore",
                chunksize=chunksize,
            )
        )
    except Exception:
        return None

    def is_multiparagraph(text: str) -> bool:
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        return ("\n\n" in t) if treat_blankline_as_paragraph else ("\n" in t)

    def generator() -> Iterable[str]:
        count = 0
        row_offset = 0

        try:
            for chunk in pd.read_csv(
                file_name,
                usecols=usecols,
                encoding="utf-8",
                encoding_errors="ignore",
                chunksize=chunksize,
            ):
                if network is not None:
                    chunk = chunk[
                        chunk[network_col].astype(str).str.casefold()
                        == network.casefold()
                    ]

                series = chunk[col_name].dropna()

                for local_i, s in enumerate(series.astype(str).str.strip()):
                    if not s:
                        continue

                    global_row = row_offset + local_i
                    prefixed = f"{global_row} {s}"

                    if is_multiparagraph(s):
                        multiline_bucket.append(prefixed)
                        yield prefixed.replace("\n", " ").replace("\r", " ")
                    else:
                        yield prefixed
                        count += 1

                    if limit > 0 and count >= limit:
                        return

                row_offset += len(chunk)

        except Exception:
            return

    g = generator()
    g_check, g_live = tee(g, 2)

    try:
        next(g_check)
    except StopIteration:
        return None

    return g_live


def _safe_suffix(val: Optional[str]) -> str:
    import re

    if not val or val == "none":
        return "all"

    val = val.strip().casefold().replace(" ", "_")
    val = re.sub(r"[^a-z0-9_\-]", "", val)
    return val or "all"


def save(
    output_dir: Path,
    df: pd.DataFrame,
    network_filter: Optional[str] = None,
) -> Path:
    base = Path(config.SECOND_OUTPUT_PATH)
    suffix = _safe_suffix(network_filter)

    output_filename = f"{base.stem}_{suffix}{base.suffix}"
    output_path = output_dir / output_filename

    df.to_csv(
        output_path,
        index=config.SECOND_OUTPUT_INDEX,
        encoding=config.SECOND_OUTPUT_ENCODING,
        sep=config.SECOND_OUTPUT_SEPARATOR,
    )

    print(f"Saved to: {output_path}")
    return output_path


def get_sentences(
    csv_path: str,
    network_filter: Optional[str] = None,
) -> Iterable[str]:
    try:
        sentences = load_csv_sentences(
            file_name=csv_path,
            col_name=col_name,
            limit=limit,
            network_filter=network_filter,
            network_col="Network",
        )
        return sentences if sentences is not None else iter(())
    except Exception as e:
        print(f"Error loading CSV sentences: {e}")
        return iter(())


def run_pipeline(
    viewer: SentimentViewer,
    sentences: Iterable[str],
    sid: SentimentIntensityAnalyzer,
) -> None:
    total = 0.0
    count = 0
    excluded_zero = 0

    for s in sentences:
        score = sid.polarity_scores(s)["compound"]

        viewer.add_line(s, score=score)

        if score == 0.0:
            excluded_zero += 1
            continue

        total += score
        count += 1

    avg = total / count if count > 0 else 0.0

    summary = (
        f"Average compound sentiment over {count} sentences: {avg} "
        f"(excluded {excluded_zero} for score 0.0)"
    )

    viewer.add_line(summary, score=avg, colour_override="#45a9cd")
    print(summary)


    avg_100 = compound_to_100(avg)
    summary_100 = f"Sentiment score (0–100): {avg_100:.2f}"

    viewer.add_line(summary_100, score=avg, colour_override="#45a9cd")
    print(summary_100)

    viewer.run()


def main(csv_path: str, network_filter: Optional[str] = None) -> None:
    if network_filter == "none":
        network_filter = None

    sid = build_analyzer()
    sentences = get_sentences(csv_path, network_filter=network_filter)

    viewer = SentimentViewer(sid=sid, title="Comment Sentiment Log")
    run_pipeline(viewer, sentences, sid)

    analyzed_df = pd.DataFrame(
        viewer.analyzed_sentences,
        columns=["Analyzed Sentence"],
    )
    save(Path(csv_path).parent, analyzed_df, network_filter)