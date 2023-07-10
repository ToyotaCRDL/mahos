#!/usr/bin/env python3

from ..msgs.common_msgs import Message, Request, Resp

from typing import NewType, Callable, Union, Tuple

#: NodeName is str hostname::nodename or 2-tuple of str (hostname, nodename)
NodeName = NewType("NodeName", Union[Tuple[str, str], str])

#: SubHandler receives a Message and handle it
SubHandler = NewType("SubHandler", Callable[[Message], None])

#: RepHandler receives a Request and return Resp
RepHandler = NewType("RepHandler", Callable[[Request], Resp])

#: MessageGetter: getter function for Message
MessageGetter = NewType("MessageGetter", Callable[[], Message])
