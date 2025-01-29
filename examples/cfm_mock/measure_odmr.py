#!/usr/bin/env python3

import matplotlib.pyplot as plt

from mahos import load_gconf
from mahos.meas.odmr import ODMRClient, BinaryState


def main():
    gconf = load_gconf("conf.toml")
    cli = ODMRClient(gconf, "localhost::odmr")
    params = cli.get_param_dict("cw")
    params["start"].set(2.0e9)
    params["stop"].set(2.1e9)

    # Enable and store measurement identifier in params.
    # (it's generated at ODMR logic on get_param_dict() above.)
    params["ident"].set_enable(True)
    ident = params["ident"].value()

    params.pprint()

    if not cli.start(params, "cw"):
        print("failed to start!")
        return

    sweeps = 0
    while True:
        status = cli.get_state()
        data = cli.get_data()
        if status == BinaryState.IDLE or data is None or data.ident != ident:
            continue
        if sweeps != data.sweeps():
            sweeps = data.sweeps()
            print(f"{sweeps} sweeps done")
        if sweeps >= 20:
            break

    cli.stop()
    x = data.get_xdata()
    y, _ = data.get_ydata()
    plt.plot(x, y)
    plt.show()
    cli.save_data("test.odmr.h5")


if __name__ == "__main__":
    main()
