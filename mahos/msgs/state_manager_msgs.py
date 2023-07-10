#!/usr/bin/env python3

from pprint import pformat

from .common_msgs import Request, Status


class ManagerStatus(Status):
    def __init__(self, states: dict):
        self.states = states

    def __repr__(self):
        return f"ManagerStatus({self.states})"

    def __str__(self):
        return "Manager->states:\n" + pformat(self.states)


class CommandReq(Request):
    def __init__(self, name: str):
        self.name = name


class RestoreReq(Request):
    def __init__(self, name: str):
        self.name = name
