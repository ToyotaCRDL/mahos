#!/usr/bin/env python3

"""
Common measurement Workers.

.. This file is a part of MAHOS project.

"""

import typing as T

from ..msgs.inst_pg_msgs import Block, Blocks
from ..inst.interface import InstrumentInterface
from ..inst.pg_interface import PGInterface
from ..inst.server import MultiInstrumentClient


class Worker(object):
    """Base class for measurement Workers.

    Implements minimal interfaces (start() and stop()), and convenient functions.

    """

    def __init__(self, cli: MultiInstrumentClient, logger):
        self.cli = cli
        self.logger = logger
        self._instruments: T.List[InstrumentInterface] = []

    def add_instrument(self, inst: InstrumentInterface):
        self._instruments.append(inst)

    def add_instruments(self, *insts: T.Optional[InstrumentInterface]):
        """Add instruments. If None is contained, it is silently ignored."""

        self._instruments.extend([i for i in insts if i is not None])

    def lock_instruments(self) -> bool:
        """lock all the instruments registered by add_instrument[s]().

        :returns: True on success.

        """

        return all([inst.lock() for inst in self._instruments])

    def release_instruments(self) -> bool:
        """release all the instruments registered by add_instrument[s]().

        :returns: True on success.

        """

        return all([inst.release() for inst in self._instruments])

    def fail_with_release(self, msg: str = "") -> bool:
        """Report error, release instruments, and return False.

        Useful to exit during start() on failure case.

        """

        if msg:
            self.logger.error(msg)
        self.release_instruments()
        return False

    def start(self, params: T.Optional[dict] = None) -> bool:
        """Start the worker. Returns True on success."""

        self.logger.error("start() is not implemented.")
        return False

    def stop(self) -> bool:
        """Stop the worker. Returns True on success."""

        self.logger.error("stop() is not implemented.")
        return False


class DummyWorker(Worker):
    """Dummy of an optional worker object, which always successes in operations (do nothing)."""

    def __init__(self, cli=None, logger=None):
        Worker.__init__(self, cli, logger)
        self.running = False

    def start(self, params: T.Optional[dict] = None) -> bool:
        self.running = True
        return True

    def stop(self) -> bool:
        self.running = False
        return True


class PulseGen_CW(Worker):
    """A worker object to setup a pulse generator ('pg') for CW output mode."""

    def __init__(self, cli, logger):
        Worker.__init__(self, cli, logger)
        self.pg = PGInterface(cli, "pg")
        self.add_instrument(self.pg)
        self.running = False

    def start(self) -> bool:
        freq = 1.0e6
        b = Block("LaserOn", [("laser", 1000)])
        blocks = Blocks([b])
        success = (
            self.pg.lock()
            and self.pg.set("clear")
            and self.pg.stop()
            and self.pg.configure_blocks(blocks, freq)
            and self.pg.start()
            and self.pg.get("opc")
        )

        if not success:
            return self.fail_with_release("Error starting pulse generator.")

        self.running = True
        self.logger.info("Started pulse generator for CW laser output.")
        return True

    def stop(self) -> bool:
        # avoid double-stop
        if not self.running:
            return False

        success = self.pg.stop() and self.pg.release()

        if success:
            self.running = False
            self.logger.info("Stopped pulse generator.")
        else:
            self.logger.error("Error stopping pulse generator.")

        return success


class Switch(Worker):
    """A worker object to send command to 'switch' instrument (daq.DigitalOut)."""

    def __init__(self, cli, logger, command_name):
        Worker.__init__(self, cli, logger)
        self.switch = InstrumentInterface(cli, "switch")
        self.add_instrument(self.switch)

        self.command_name = command_name
        self.running = False

    def start(self) -> bool:
        success = self.switch.lock() and self.switch.set("command", self.command_name)

        if not success:
            return self.fail_with_release("Error starting switch.")

        self.running = True
        self.logger.info(f"Turned switch for {self.command_name}.")
        return True

    def stop(self) -> bool:
        # avoid double-stop
        if not self.running:
            return False

        success = self.switch.release()

        if success:
            self.running = False
            self.logger.info("Released lock of switch.")
        else:
            self.logger.error("Error releasing lock of switch.")

        return success
