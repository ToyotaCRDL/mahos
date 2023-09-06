#!/usr/bin/env python3

from pprint import pformat

from .common_msgs import Request, Status


class GlobalParamsStatus(Status):
    def __init__(self, params: dict):
        self.params = params

    def __repr__(self):
        return f"GlobalParamsStatus({self.params})"

    def __str__(self):
        return "GlobalParams->params:\n" + pformat(self.params)


class SetParamReq(Request):
    def __init__(self, key: str, value):
        self.key = key
        self.value = value
