#!/usr/bin/env python3

"""
Tests for Tweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from util import get_some

from fixtures import ctx, gconf, server, tweaker, tweaker_conf
import mahos.msgs.param_msgs as P


def test_tweaker(server, tweaker, tweaker_conf):
    poll_timeout_ms = tweaker_conf["poll_timeout_ms"]

    tweaker.wait()

    param_dict_ids = get_some(tweaker.get_status, poll_timeout_ms).param_dict_ids
    assert param_dict_ids == ["paramX::labelA", "paramY::labelB"]
    pX_A = tweaker.read("paramX::labelA")
    pX_A_A: P.IntParam = pX_A["paramA"]
    pX_A_B: P.FloatParam = pX_A["paramB"]
    pX_A_A.set(1)
    pX_A_B.set(0.1)

    assert tweaker.write("paramX::labelA", pX_A)
    pX_A_new = tweaker.read("paramX::labelA")
    assert P.isclose(pX_A, pX_A_new)

    success, param_dicts = tweaker.read_all()
    assert success
    pY_B = param_dicts["paramY::labelB"]
    pY_B_C: P.StrParam = pY_B["paramC"]
    pY_B_D: P.BoolParam = pY_B["paramD"]
    pY_B_C.set("bbb")
    pY_B_D.set(True)

    assert tweaker.write_all(param_dicts)
    success, param_dicts_new = tweaker.read_all()
    assert success
    assert P.isclose(P.ParamDict(param_dicts), P.ParamDict(param_dicts_new))
