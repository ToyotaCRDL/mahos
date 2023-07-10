#!/usr/bin/env python3

from pprint import pformat

from .common_msgs import Request, Status


class ParamServerStatus(Status):
    def __init__(self, params: dict):
        self.params = params

    def __repr__(self):
        return f"ParamServerStatus({self.params})"

    def __str__(self):
        return "ParamServer->params:\n" + pformat(self.params)


class SetParamReq(Request):
    def __init__(self, name: str, value):
        self.name = name
        self.value = value
