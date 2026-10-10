"""Explicit Table 6 labels and verified CWRU 0 HP source files."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassSource:
    label: int
    bearing_position: str
    fault_type: str
    fault_size_inches: float | None
    source_file: str
    signal_field: str
    outer_race_position: str | None = None


def _source(label: int, end: str, fault: str, size: float, number: int) -> ClassSource:
    folder = "fan_end_12k_0HP" if end == "fan_end" else "drive_end_12k_0HP"
    channel = "FE" if end == "fan_end" else "DE"
    return ClassSource(
        label=label,
        bearing_position=end,
        fault_type=fault,
        fault_size_inches=size,
        source_file=f"{folder}/{number}.mat",
        signal_field=f"X{number:03d}_{channel}_time",
        outer_race_position="@6" if fault == "outer_race" else None,
    )


# 类别顺序对应 Table 6；文件号按 CWRU 0 HP 数据表核对，外圈故障均选 @6 位置。
CLASSES = (
    ClassSource(0, "normal", "none", None, "normal/97.mat", "X097_DE_time"),
    _source(1, "fan_end", "inner_race", 0.007, 278),
    _source(2, "fan_end", "inner_race", 0.014, 274),
    _source(3, "fan_end", "inner_race", 0.021, 270),
    _source(4, "fan_end", "ball", 0.007, 282),
    _source(5, "fan_end", "ball", 0.014, 286),
    _source(6, "fan_end", "ball", 0.021, 290),
    _source(7, "fan_end", "outer_race", 0.007, 294),
    _source(8, "fan_end", "outer_race", 0.014, 313),
    _source(9, "fan_end", "outer_race", 0.021, 315),
    _source(10, "drive_end", "inner_race", 0.007, 105),
    _source(11, "drive_end", "inner_race", 0.014, 169),
    _source(12, "drive_end", "inner_race", 0.021, 209),
    _source(13, "drive_end", "ball", 0.007, 118),
    _source(14, "drive_end", "ball", 0.014, 185),
    _source(15, "drive_end", "ball", 0.021, 222),
    _source(16, "drive_end", "outer_race", 0.007, 130),
    _source(17, "drive_end", "outer_race", 0.014, 197),
    _source(18, "drive_end", "outer_race", 0.021, 234),
)
