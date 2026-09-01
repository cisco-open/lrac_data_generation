# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Compatibility exporters for downstream recipes."""

from .kaldi import export_kaldi

__all__ = ["export_kaldi"]
