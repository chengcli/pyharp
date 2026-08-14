#!/usr/bin/env python

from pyharp.opacity import OpacityOptions


def test_opacity_options_nmom_getter_setter():
    op = OpacityOptions()
    assert hasattr(op, "nmom")
    assert op.nmom() == 0
    assert op.nmom(1) is op
    assert op.nmom() == 1
