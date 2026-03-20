"""Thin wrapper around rules_python requirement labels."""

load("@pip_requirements//:requirements.bzl", _requirement = "requirement")

def requirement(name):
    return _requirement(name)
