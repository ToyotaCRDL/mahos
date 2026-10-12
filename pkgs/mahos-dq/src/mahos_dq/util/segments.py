#!/usr/bin/env python3

"""
Segment sample utility functions.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

DEFAULT_SEGMENT_SAMPLES_MIN = 16
DEFAULT_SEGMENT_SAMPLES_BASE = 16


def validate_segment_granularity(
    granularity: int, base: int = DEFAULT_SEGMENT_SAMPLES_BASE
) -> int:
    """Validate and return a segment sample granularity."""

    granularity = int(granularity)
    base = int(base)
    if base < 1:
        raise ValueError("Segment sample base must be positive.")
    if granularity < base:
        raise ValueError(f"Segment granularity must be at least {base} samples.")
    if granularity % base:
        raise ValueError(f"Segment granularity must be an integer multiple of {base} samples.")
    return granularity


def offset_base_inc(value: int, base: int) -> int:
    """Round ``value`` up to an integer multiple of ``base``."""

    value = int(value)
    base = int(base)
    residual = value % base
    if residual:
        return value + base - residual
    return value


def round_segment_samples_up(
    samples: int,
    granularity: int = 16,
    minimum: int = DEFAULT_SEGMENT_SAMPLES_MIN,
    base: int = DEFAULT_SEGMENT_SAMPLES_BASE,
) -> int:
    """Round a sample count up to segment constraints."""

    granularity = validate_segment_granularity(granularity, base=base)
    samples = max(int(minimum), int(samples))
    return offset_base_inc(samples, granularity)


def round_segment_samples_down(
    samples: int,
    granularity: int = 16,
    minimum: int = DEFAULT_SEGMENT_SAMPLES_MIN,
    base: int = DEFAULT_SEGMENT_SAMPLES_BASE,
) -> int:
    """Round a sample count down to segment constraints."""

    minimum = int(minimum)
    granularity = validate_segment_granularity(granularity, base=base)
    rounded = int(samples) // granularity * granularity
    if rounded < minimum:
        raise ValueError(f"Segment samples became too small ({rounded} < {minimum}).")
    return rounded


def valid_segment_samples(
    samples: int,
    granularity: int = 16,
    minimum: int = DEFAULT_SEGMENT_SAMPLES_MIN,
    base: int = DEFAULT_SEGMENT_SAMPLES_BASE,
) -> bool:
    """Return whether a sample count satisfies segment constraints."""

    granularity = validate_segment_granularity(granularity, base=base)
    samples = int(samples)
    return samples >= int(minimum) and samples % granularity == 0


def round_duration_for_segment_samples(
    duration: int,
    duration_step: int,
    sample_factor: int,
    sample_divisor: int,
    granularity: int = 16,
    minimum: int = DEFAULT_SEGMENT_SAMPLES_MIN,
    base: int = DEFAULT_SEGMENT_SAMPLES_BASE,
    max_iterations: int = 1_000_000,
) -> tuple[int, int]:
    """Increment ``duration`` until derived segment samples are valid.

    The derived segment samples are ``duration * sample_factor // sample_divisor``.
    ``duration`` is incremented by ``duration_step``.

    """

    if sample_factor < 1:
        raise ValueError("sample_factor must be positive.")
    if sample_divisor < 1:
        raise ValueError("sample_divisor must be positive.")
    validate_segment_granularity(granularity, base=base)

    duration_step = int(duration_step)
    duration = offset_base_inc(max(1, int(duration)), duration_step)
    for _ in range(max_iterations):
        samples = duration * sample_factor // sample_divisor
        if valid_segment_samples(samples, granularity=granularity, minimum=minimum, base=base):
            return duration, samples
        duration += duration_step

    raise ValueError(
        "could not find a compatible duration "
        f"from duration={duration}, duration_step={duration_step}, "
        f"sample_factor={sample_factor}, sample_divisor={sample_divisor}, "
        f"granularity={granularity}"
    )
