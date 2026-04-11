"""Default output filename helpers for spectroscopy CLIs."""

from __future__ import annotations

from pathlib import Path


def _format_value(value: float | int | str, unit: str = "") -> str:
    if isinstance(value, str):
        text = value
    else:
        text = f"{float(value):g}"
    return f"{text.replace('-', 'm').replace('.', 'p')}{unit}"


def _clean_token(value: object) -> str:
    text = str(value).strip().lower()
    pieces: list[str] = []
    previous_was_separator = False
    for char in text:
        if char.isalnum():
            pieces.append(char)
            previous_was_separator = False
        elif char == ".":
            pieces.append("p")
            previous_was_separator = False
        elif not previous_was_separator:
            pieces.append("_")
            previous_was_separator = True
    return "".join(pieces).strip("_") or "output"


def default_output_path(
    *,
    target_name: object,
    plot_type: str,
    temperature_k: float | str,
    pressure_bar: float | str,
    wn_range: tuple[float, float],
    suffix: str,
    output_dir: Path = Path("output"),
) -> Path:
    """Return a default output path using the spectroscopy CLI filename pattern."""
    wn_min, wn_max = wn_range
    stem = "_".join(
        [
            _clean_token(target_name),
            _clean_token(plot_type),
            _format_value(pressure_bar, "bar"),
            _format_value(temperature_k, "K"),
            _format_value(wn_min),
            _format_value(wn_max, "cm1"),
        ]
    )
    return output_dir / f"{stem}{suffix}"
