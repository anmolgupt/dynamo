# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for consistently parsing boolean environment variables."""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("", "0", "false", "no", "off"))


def optional_env_bool(name: str) -> bool | None:
    """Return an optional boolean environment value.

    Missing variables return None. Invalid values raise ValueError so operator
    configuration mistakes fail clearly instead of silently enabling a
    feature.
    """

    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}"
    )


def env_bool(name: str, *, default: bool = False) -> bool:
    """Return a boolean environment value, falling back when it is missing."""

    value = optional_env_bool(name)
    return default if value is None else value
