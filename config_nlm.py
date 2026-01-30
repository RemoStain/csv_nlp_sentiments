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
    # 0 = default data file from cleaner
    0: InputPreset(
        preset_value=0,
        file_default="data.csv",
        col_default="Content",
        limit_default=0,
    ),
    # 1 = Small Data Set
    1: InputPreset(
        preset_value=1,
        file_default="Yasmins_comments.csv",
        col_default="Content",
        limit_default=100,
    ),
    # 2 = Real Data Set
    2: InputPreset(
        preset_value=2,
        file_default="TMS_socials_December_2025.csv",
        col_default="Content",
        limit_default=100,
    ),
}