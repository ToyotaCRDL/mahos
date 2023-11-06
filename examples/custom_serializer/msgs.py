#!/usr/bin/env python3

from __future__ import annotations
import datetime

import msgpack

from mahos.msgs.common_msgs import Message, Request, Reply


class DatetimeMP(Message):
    def __init__(self, dt: datetime.datetime):
        self.datetime = dt

    def serialize(self) -> bytes:
        msg = {
            "year": self.datetime.year,
            "month": self.datetime.month,
            "day": self.datetime.day,
            "hour": self.datetime.hour,
            "minute": self.datetime.minute,
            "second": self.datetime.second,
            "microsecond": self.datetime.microsecond,
        }
        return msgpack.dumps(msg)

    @classmethod
    def deserialize(cls, b: bytes) -> DatetimeMP | None:
        try:
            msg = msgpack.loads(b)
            return cls(
                datetime.datetime(
                    msg["year"],
                    msg["month"],
                    msg["day"],
                    msg.get("hour", 0),
                    msg.get("minute", 0),
                    msg.get("second", 0),
                    msg.get("microsecond", 0),
                )
            )
        except:
            return None


class CalcReqMP(Request):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def serialize(self) -> bytes:
        return msgpack.dumps({"a": self.a, "b": self.b})

    @classmethod
    def deserialize(cls, b: bytes) -> CalcReqMP | None:
        try:
            msg = msgpack.loads(b)
            return CalcReqMP(msg["a"], msg["b"])
        except:
            return None


class CalcRepMP(Reply):
    def __init__(self, success, message, product, quotient):
        self.success = success
        self.message = message
        self.product = product
        self.quotient = quotient

    def __repr__(self):
        return f"CalcRepMP({self.success}, {self.message}, {self.product}, {self.quotient})"

    def serialize(self) -> bytes:
        return msgpack.dumps(
            {
                "success": self.success,
                "message": self.message,
                "product": self.product,
                "quotient": self.quotient,
            }
        )

    @classmethod
    def deserialize(cls, b: bytes) -> CalcReqMP | None:
        try:
            msg = msgpack.loads(b)
            return CalcRepMP(msg["success"], msg["message"], msg["product"], msg["quotient"])
        except:
            return None
