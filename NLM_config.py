from dataclasses import dataclass

@dataclass(frozen=True)
class InputPreset:
    preset_value: int
    file_default: str
    col_default: str
    limit_default: int


INPUT_PRESETS: dict[int, InputPreset] = {
    # -1 = Manual Input Only
    -1: InputPreset(
        preset_value=-1,
        file_default="",
        col_default="",
        limit_default=0,
    ),
    # 0 = Massive Data Set
    0: InputPreset(
        preset_value=0,
        file_default="comments.csv",
        col_default="self_text",
        limit_default=5000,
    ),
    # 1 = Small Data Set
    1: InputPreset(
        preset_value=1,
        file_default="Yasmins_comments.csv",
        col_default="Content",
        limit_default=100,
    ),
    # 2 = More Filtered Data Set
    2: InputPreset(
        preset_value=2,
        file_default="Yasmins_comments2.csv",
        col_default="Content",
        limit_default=100,
    ),
}
