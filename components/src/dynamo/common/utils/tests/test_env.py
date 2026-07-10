# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.utils.env import env_bool, optional_env_bool


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_optional_env_bool_true(monkeypatch, value):
    monkeypatch.setenv("DYN_TEST_BOOL", value)

    assert optional_env_bool("DYN_TEST_BOOL") is True


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off"])
def test_optional_env_bool_false(monkeypatch, value):
    monkeypatch.setenv("DYN_TEST_BOOL", value)

    assert optional_env_bool("DYN_TEST_BOOL") is False


def test_optional_env_bool_missing(monkeypatch):
    monkeypatch.delenv("DYN_TEST_BOOL", raising=False)

    assert optional_env_bool("DYN_TEST_BOOL") is None
    assert env_bool("DYN_TEST_BOOL", default=True) is True


def test_optional_env_bool_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("DYN_TEST_BOOL", "sometimes")

    with pytest.raises(ValueError, match="DYN_TEST_BOOL must be one of"):
        optional_env_bool("DYN_TEST_BOOL")
