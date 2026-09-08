#!/usr/bin/env python

from pyharp.opacity import OpacityOptions


def test_opacity_options_nmom_getter_setter():
    op = OpacityOptions()
    assert hasattr(op, "nmom")
    assert op.nmom() == 0
    assert op.nmom(1) is op
    assert op.nmom() == 1


def test_opacity_options_warn_out_of_bounds_getter_setter():
    op = OpacityOptions()
    assert hasattr(op, "warn_out_of_bounds")
    assert op.warn_out_of_bounds() is False
    assert op.warn_out_of_bounds(True) is op
    assert op.warn_out_of_bounds() is True
