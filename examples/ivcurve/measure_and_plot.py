#!/usr/bin/env python3

import time

import numpy as np
import matplotlib.pyplot as plt

from mahos.node.node import load_gconf

from ivcurve import IVCurveClient


def get_data(cli, min_sweeps):
    for i in range(100):
        data = cli.get_data()
        if data is not None and data.sweeps() >= min_sweeps:
            return data
        time.sleep(0.1)


def main():
    gconf = load_gconf("conf.toml")
    cli = IVCurveClient(gconf, "localhost::ivcurve")
    cli.wait()

    # set parameters
    params = cli.get_param_dict()
    params["start"].set(-5)
    params["stop"].set(+5)

    # do measurement
    cli.start(params)
    data1 = get_data(cli, 1)
    data20 = get_data(cli, 10)
    cli.stop(params)

    # plot results
    x = np.linspace(params["start"].value(), params["stop"].value(), params["num"].value())
    plt.plot(x, np.mean(data1.data, axis=1), label=f"{data1.sweeps()} sweeps")
    plt.plot(x, np.mean(data20.data, axis=1), label=f"{data20.sweeps()} sweeps")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
