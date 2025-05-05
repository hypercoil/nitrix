# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for documentation utility functions.
"""
import pytest
from nitrix._internal import DocTemplateFormat


def test_nested_doc_parse():
    teststr = '{a} {b} {c}'
    with pytest.raises(KeyError):
        teststr.format_map({'a': 1, 'b': 2}).format_map({'c': 3})
    teststr.format_map(DocTemplateFormat(
        {'a': 1, 'b': 2}
    )).format_map({'c': 3})
