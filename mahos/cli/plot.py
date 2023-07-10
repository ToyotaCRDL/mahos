#!/usr/bin/env python3

import sys
import argparse
from os import path
import pprint
from distutils.util import strtobool
import toml

from ..msgs.fit_msgs import str_to_peak_type
from ..meas.confocal_io import ConfocalIO
from ..meas.odmr_io import ODMRIO
from ..meas.podmr_io import PODMRIO
from ..meas.iodmr_io import IODMRIO
from ..meas.hbt_io import HBTIO
from ..meas.spectroscopy_io import SpectroscopyIO
from ..util.cui import prompt


def plot_data(args, fn, data, plot, force=False):
    for ext in args.exts.split(","):
        if not ext.startswith("."):
            ext = "." + ext
        export_fn = path.splitext(fn)[0] + args.suffix + ext
        if (
            force
            or args.force
            or not path.exists(export_fn)
            or prompt(f"Overwrite {export_fn} ?", "y")
        ):
            plot(export_fn, data)


def plot_files(args, load, plot):
    for fn in args.names:
        data = load(fn)
        if data is None:
            continue
        plot_data(args, fn, data, plot)


def plot_confocal_image(args):
    io = ConfocalIO()
    params = {
        "vmax": args.vmax,
        "vmin": args.vmin,
        "cmap": args.cmap,
        "invX": args.invX,
        "invY": args.invY,
        "only": args.only,
    }
    params.update(make_default_params(args))
    plot_files(args, io.load_image, lambda fn, d: io.export_image(fn, d, params))


def plot_confocal_trace(args):
    io = ConfocalIO()
    plot_files(args, io.load_trace, io.export_trace)


def plot_odmr(args):
    io = ODMRIO()

    if args.method or args.fit_params:
        if args.fit_params:
            fit_params = toml.load(args.fit_params)
            if "method" not in fit_params:
                fit_params["method"] = args.method
            fit_params["peak_type"] = str_to_peak_type(fit_params.get("peak_type", args.peak_type))
            if fit_params["method"] == "multi" and "n_peaks" not in fit_params:
                fit_params["n_peaks"] = args.n_peaks
            if "n_guess" not in fit_params:
                fit_params["n_guess"] = args.n_guess
        else:
            fit_params = {
                "method": args.method,
                "peak_type": str_to_peak_type(args.peak_type),
                "n_guess": args.n_guess,
            }
            if fit_params["method"] == "multi":
                fit_params["n_peaks"] = args.n_peaks
    else:
        fit_params = {}

    params = {
        "fit": fit_params,
        "show_fit": args.show_fit,
        "normalize_n": args.normalize_n,
        "color": args.color,
        "color_bg": args.color_bg,
        "color_fit": args.color_fit,
        "marker": args.marker,
        "marker_bg": args.marker_bg,
        "offset": args.offset,
        "linewidth": args.linewidth,
        "linewidth_fit": args.linewidth_fit,
        "base_line": args.base_line,
    }
    params.update(make_default_params(args))
    if not args.one_figure:
        plot_files(args, io.load_data, lambda fn, d: io.export_data(fn, d, params))
    else:
        data = [io.load_data(n) for n in args.names]
        io.export_data(args.one_figure, data, params)


def plot_podmr(args):
    def print_params(_, data):
        pprint.pp(data.params["plot"])

    io = PODMRIO()
    if args.print:
        args.force = True  # force here because print_params won't write file
        plot_files(args, io.load_data, print_params)
        return

    plot_params = {}
    if args.timings is not None:
        for key, value in zip(("sigdelay", "sigwidth", "refdelay", "refwidth"), args.timings):
            plot_params[key] = value * 1e-9  # ns to sec
    for key in ("plotmode", "taumode", "xlogscale", "ylogscale", "fft", "refmode", "refaverage"):
        value = getattr(args, key)
        if value is not None:
            plot_params[key] = value
    if args.method or args.fit_params:
        if args.fit_params:
            fit_params = toml.load(args.fit_params)
            if "method" not in fit_params:
                fit_params["method"] = args.method
        else:
            fit_params = {"method": args.method}
    else:
        fit_params = {}

    params = {
        "plot": plot_params,
        "fit": fit_params,
        "show_fit": args.show_fit,
        "color0": args.color0,
        "color1": args.color1,
        "color_fit": args.color_fit,
        "marker0": args.marker0,
        "marker1": args.marker1,
        "label": args.label,
        "offset": args.offset,
        "linewidth": args.linewidth,
        "linewidth_fit": args.linewidth_fit,
    }
    params.update(make_default_params(args))
    if not args.one_figure:
        plot_files(args, io.load_data, lambda fn, d: io.export_data(fn, d, params))
    else:
        data = [io.load_data(n) for n in args.names]
        io.export_data(args.one_figure, data, params)


def plot_iodmr(args):
    io = IODMRIO()
    params = {
        "wslice": args.wslice,
        "hslice": args.hslice,
        "freq": args.freq,
        "color": args.color,
        "marker": args.marker,
        "offset": args.offset,
        "linewidth": args.linewidth,
        "vmax": args.vmax,
        "vmin": args.vmin,
        "cmap": args.cmap,
    }
    params.update(make_default_params(args))
    if not args.one_figure:
        plot_files(args, io.load_data, lambda fn, d: io.export_data(fn, d, params))
    else:
        data = [io.load_data(n) for n in args.names]
        io.export_data(args.one_figure, data, params)


def fit_iodmr(args):
    io = IODMRIO()
    fit_params = {
        "method": args.method,
        "peak": args.peak,
        "resize": {"width": args.width, "height": args.height},
        "flim": (args.fmin, args.fmax),
        "n_guess": args.n_guess,
        "n_workers": args.n_workers,
    }
    plot_params = {
        "all": args.all,
        "vmax": args.vmax,
        "vmin": args.vmin,
        "cmap": args.cmap,
    }
    plot_params.update(make_default_params(args))

    for fn in args.names:
        head, ext = path.splitext(fn)
        if ext != ".fit":
            # perform fit
            data = io.load_data(fn)
            if data is None:
                continue
            xfn = f"{head}{args.suffix}.fit"
            if args.force or not path.exists(xfn) or prompt(f"Overwrite {xfn} ?", "y"):
                fit = io.fit_save_data(xfn, data, fit_params)
        else:
            fit = io.load_fit(fn)

        plot_data(args, fn, fit, lambda fn, d: io.export_fit(fn, d, plot_params), force=True)


def plot_hbt(args):
    io = HBTIO()

    plot_params = {}
    for key in ("t0", "ref_start", "ref_stop"):
        value = getattr(args, key)
        if value is not None:
            plot_params[key] = value * 1e-9
    if args.bg_ratio is not None:
        plot_params["bg_ratio"] = args.bg_ratio * 0.01

    if args.method or args.fit_params:
        if args.fit_params:
            fit_params = toml.load(args.fit_params)
            if "method" not in fit_params:
                fit_params["method"] = args.method
        else:
            fit_params = {"method": args.method}
    else:
        fit_params = {}
    params = {
        "plot": plot_params,
        "fit": fit_params,
        "show_fit": args.show_fit,
        "normalize": args.normalize,
        "color": args.color,
        "color_fit": args.color_fit,
        "marker": args.marker,
        "offset": args.offset,
        "linewidth": args.linewidth,
        "linewidth_fit": args.linewidth_fit,
        "label": args.label,
    }
    params.update(make_default_params(args))
    if not args.one_figure:
        plot_files(args, io.load_data, lambda fn, d: io.export_data(fn, d, params))
    else:
        data = [io.load_data(n) for n in args.names]
        io.export_data(args.one_figure, data, params)


def plot_spec(args):
    io = SpectroscopyIO()

    if args.method or args.fit_params:
        if args.fit_params:
            fit_params = toml.load(args.fit_params)
            if "method" not in fit_params:
                fit_params["method"] = args.method
            fit_params["peak_type"] = str_to_peak_type(fit_params.get("peak_type", args.peak_type))
            if fit_params["method"] == "multi" and "n_peaks" not in fit_params:
                fit_params["n_peaks"] = args.n_peaks
            if "n_guess" not in fit_params:
                fit_params["n_guess"] = args.n_guess
        else:
            fit_params = {
                "method": args.method,
                "peak_type": str_to_peak_type(args.peak_type),
                "n_guess": args.n_guess,
            }
            if fit_params["method"] == "multi":
                fit_params["n_peaks"] = args.n_peaks
    else:
        fit_params = {}
    params = {
        "fit": fit_params,
        "show_fit": args.show_fit,
        "filter_n": args.filter_n,
        "color": args.color,
        "color_fit": args.color_fit,
        "marker": args.marker,
        "offset": args.offset,
        "linewidth": args.linewidth,
        "linewidth_fit": args.linewidth_fit,
        "label": args.label,
    }
    params.update(make_default_params(args))
    if not args.one_figure:
        plot_files(args, io.load_data, lambda fn, d: io.export_data(fn, d, params))
    else:
        data = [io.load_data(n) for n in args.names]
        io.export_data(args.one_figure, data, params)


def make_default_params(args):
    figsize = [float(i) for i in args.figsize.split(",")]
    if len(figsize) != 2:
        raise ValueError(f"Invalid figsize: {args.figsize}")

    return {
        "figsize": figsize,
        "fontsize": args.fontsize,
        "dpi": args.dpi,
        "legend": args.legend,
        "xmax": args.xmax,
        "xmin": args.xmin,
        "ymax": args.ymax,
        "ymin": args.ymin,
    }


def add_common_args(parser, default_limits=((None, None), (None, None))):
    """default_limits: ((xmin, xmax), (ymin, ymax))"""

    parser.add_argument(
        "-e",
        "--exts",
        default=".png",
        help="(comma-delimited) file extension(s). Default to .png.",
    )
    parser.add_argument(
        "-s", "--suffix", default="", type=str, help="suffix for output file name."
    )
    parser.add_argument("-f", "--force", action="store_true", help="do not ask about overwrite.")
    parser.add_argument(
        "--figsize",
        type=str,
        default="14,12",
        help="figsize (width,height) in inches. default is '14,12'",
    )
    parser.add_argument("--fontsize", type=float, default=28.0, help="fontsize for labels.")
    parser.add_argument("-D", "--dpi", type=float, default=100.0, help="resolution in dpi.")
    parser.add_argument(
        "-L",
        "--legend",
        metavar="LOC",
        type=str,
        help="Show legend at location (best|upper right|upper left|...)",
    )
    parser.add_argument(
        "-x", "--xmin", type=float, default=default_limits[0][0], help="Lower bound of x-axis"
    )
    parser.add_argument(
        "-X", "--xmax", type=float, default=default_limits[0][1], help="Upper bound of x-axis"
    )
    parser.add_argument(
        "-y", "--ymin", type=float, default=default_limits[1][0], help="Lower bound of y-axis"
    )
    parser.add_argument(
        "-Y", "--ymax", type=float, default=default_limits[1][1], help="Upper bound of y-axis"
    )
    parser.add_argument("names", nargs="+", help="file names")


def add_confocal_image_parser(sub_parsers):
    p = sub_parsers.add_parser("image", help="confocal image.")
    p.add_argument("-V", "--vmax", type=float, help="Upper bound of color map")
    p.add_argument("-v", "--vmin", type=float, help="Lower bound of color map")
    p.add_argument("-c", "--cmap", type=str, default="inferno", help="color map")
    p.add_argument("--invX", action="store_true", help="invert X axis")
    p.add_argument("--invY", action="store_true", help="invert Y axis")
    p.add_argument(
        "-O",
        "--only",
        action="store_true",
        help="enable only mode: save image only without axis, colorbar, or labels",
    )
    add_common_args(p)
    p.set_defaults(func=plot_confocal_image)


def add_confocal_trace_parser(sub_parsers):
    p = sub_parsers.add_parser("trace", help="confocal trace.")
    add_common_args(p)
    p.set_defaults(func=plot_confocal_trace)


def add_odmr_parser(sub_parsers):
    p = sub_parsers.add_parser("odmr", help="odmr spectrum.")

    p.add_argument(
        "-o", "--one-figure", metavar="FILENAME", type=str, help="Filename for one-figure mode"
    )
    p.add_argument(
        "-F", "--no-fit", dest="show_fit", action="store_false", help="Don't show fitting result"
    )
    p.add_argument(
        "-n", "--normalize-n", type=int, default=0, help="Normalize data using top n intensities."
    )
    p.add_argument(
        "-m",
        "--method",
        type=str,
        help="[fit] Fitting method (single|multi|nvb|nvba). Invokes re-fitting.",
    )
    p.add_argument(
        "-P",
        "--fit-params",
        type=str,
        help="[fit] Fitting parameters file name. Invokes re-fitting.",
    )
    p.add_argument(
        "-p",
        "--peak-type",
        type=str,
        default="voigt",
        help="[fit] Peak function for fitting (gaussian|lorentzian|voigt)",
    )
    p.add_argument("-N", "--n-peaks", type=int, default=2, help="[fit/multi] Number of peaks.")
    p.add_argument(
        "-g",
        "--n-guess",
        type=int,
        default=20,
        help="[fit] Number of samples to use for peak position guess",
    )
    p.add_argument("--color", type=str, nargs="+", help="matplotlib colors for data")
    p.add_argument("--color_bg", type=str, nargs="+", help="matplotlib colors for background")
    p.add_argument("--color_fit", type=str, nargs="+", help="matplotlib colors for fitting lines")
    p.add_argument("--marker", type=str, nargs="+", help="matplotlib markers for data")
    p.add_argument("--marker_bg", type=str, nargs="+", help="matplotlib markers for background")
    p.add_argument("--linewidth", type=float, help="linewidth for data")
    p.add_argument("--linewidth_fit", type=float, default=1.0, help="linewidth for fitting line")
    p.add_argument("-O", "--offset", type=float, nargs="+", help="offset along y-axis")
    p.add_argument(
        "--base-line", action="store_true", help="draw horizontal lines for normalization baseline"
    )

    add_common_args(p)
    p.set_defaults(func=plot_odmr)


def add_podmr_parser(sub_parsers):
    p = sub_parsers.add_parser(
        "podmr",
        help=" ".join(
            [
                "Pulse odmr.",
                "Reanalyze data if any [plot] options are set.",
                "Replot data if -m (--method) is set.",
            ]
        ),
    )

    p.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Print plot parameters stored in data file (no plot performed)",
    )
    p.add_argument(
        "-o", "--one-figure", metavar="FILENAME", type=str, help="Filename for one-figure mode"
    )

    p.add_argument(
        "-F", "--no-fit", dest="show_fit", action="store_false", help="Don't show fitting result"
    )
    p.add_argument(
        "-t",
        "--timings",
        type=float,
        nargs=4,
        metavar=("sigdelay", "sigwidth", "refdelay", "refwidth"),
        help="[plot] Timings in ns",
    )
    p.add_argument(
        "-M",
        "--plotmode",
        type=str,
        help="[plot] Plot mode (data01|data0|data1|diff|average|normalize|concatenate|ref)",
    )
    p.add_argument("-T", "--taumode", type=str, help="[plot] Tau mode (raw|total|freq|index|head)")
    p.add_argument(
        "-R", "--refmode", type=str, help="[plot] Reference mode (subtract|divide|ignore)"
    )
    p.add_argument("--fft", type=strtobool, help="[plot] FFT mode")
    p.add_argument("--xlog", dest="xlogscale", type=strtobool, help="[plot] logscale X axis")
    p.add_argument("--ylog", dest="ylogscale", type=strtobool, help="[plot] logscale Y axis")
    p.add_argument("-a", "--refaverage", type=strtobool, help="[plot] Reference avaraging")

    p.add_argument(
        "-m",
        "--method",
        type=str,
        help="[fit] Fitting method (rabi|fid|spinecho|gaussian|lorentzian). Invokes re-fitting.",
    )
    p.add_argument(
        "-P",
        "--fit-params",
        type=str,
        help="[fit] Fitting parameters file name. Invokes re-fitting.",
    )

    p.add_argument("-l", "--label", type=str, nargs="+", help="matplotlib labels")
    p.add_argument("-O", "--offset", type=float, nargs="+", help="offset along y-axis")
    p.add_argument("--color0", type=str, nargs="+", help="matplotlib colors for data0")
    p.add_argument("--color1", type=str, nargs="+", help="matplotlib colors for data1")
    p.add_argument("--color_fit", type=str, nargs="+", help="matplotlib colors for fitting line")
    p.add_argument("--marker0", type=str, nargs="+", help="matplotlib markers for data0")
    p.add_argument("--marker1", type=str, nargs="+", help="matplotlib markers for data1")
    p.add_argument("--linewidth", type=float, help="linewidth for data0 and data1")
    p.add_argument("--linewidth_fit", type=float, default=1.0, help="linewidth for fitting line")

    add_common_args(p)
    p.set_defaults(func=plot_podmr)


def add_iodmr_parser(sub_parsers):
    p = sub_parsers.add_parser("iodmr", help="imaging odmr spectrum.")

    p.add_argument(
        "-o", "--one-figure", metavar="FILENAME", type=str, help="Filename for one-figure mode"
    )
    p.add_argument(
        "-W", "--wslice", type=str, help="Slice (e.g., 100:200) for width (x) axis in image"
    )
    p.add_argument(
        "-H", "--hslice", type=str, help="Slice (e.g., 100:200) for height (y) axis in image"
    )
    p.add_argument("-F", "--freq", type=float, help="frequency to save sliced image")
    p.add_argument("--color", type=str, nargs="+", help="matplotlib colors for data")
    p.add_argument("--marker", type=str, nargs="+", help="matplotlib markers for data")
    p.add_argument("--linewidth", type=float, help="linewidth for data")
    p.add_argument("-O", "--offset", type=float, nargs="+", help="offset along y-axis")

    p.add_argument("-V", "--vmax", type=float, help="Upper bound of color map")
    p.add_argument("-v", "--vmin", type=float, help="Lower bound of color map")
    p.add_argument("-c", "--cmap", type=str, default="inferno", help="color map")
    add_common_args(p)
    p.set_defaults(func=plot_iodmr)


def add_iodmr_fit_parser(sub_parsers):
    p = sub_parsers.add_parser("iodmr-fit", help="fitting imaging odmr spectrum.")

    # fit
    p.add_argument(
        "-m",
        "--method",
        type=str,
        help="Fitting method (single|double|quad|nvba). Required to perform fitting",
    )
    p.add_argument(
        "-p",
        "--peak",
        type=str,
        default="lorentzian",
        help="Peak function for fitting (gaussian|lorentzian|voigt)",
    )
    p.add_argument("--fmin", type=float, help="Lower bound of frequency (MHz) used for fitting")
    p.add_argument("--fmax", type=float, help="Upper bound of frequency (MHz) used for fitting")
    p.add_argument("-W", "--width", type=int, help="Resize width")
    p.add_argument("-H", "--height", type=int, help="Resize height")
    p.add_argument(
        "-g", "--n-guess", type=int, help="Number of samples to use for peak position guess"
    )
    p.add_argument("-j", "--n-workers", type=int, default=1, help="Workers for concurrent fitting")
    # plot
    p.add_argument(
        "-a",
        "--all",
        type=int,
        default=0,
        metavar="STEP",
        help="Save fit plot for each STEP pixels",
    )
    p.add_argument("-V", "--vmax", type=float, help="Upper bound of color map")
    p.add_argument("-v", "--vmin", type=float, help="Lower bound of color map")
    p.add_argument("-c", "--cmap", type=str, default="inferno", help="color map")

    add_common_args(p)
    p.set_defaults(func=fit_iodmr)


def add_hbt_parser(sub_parsers):
    p = sub_parsers.add_parser("hbt", help="HBT data.")
    p.add_argument(
        "-o", "--one-figure", metavar="FILENAME", type=str, help="Filename for one-figure mode"
    )
    p.add_argument(
        "-N",
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Don't normalize and plot raw",
    )
    p.add_argument(
        "-F", "--no-fit", dest="show_fit", action="store_false", help="Don't show fitting result"
    )
    p.add_argument("-t", "--t0", type=float, help="[plot] Delay in ns")
    p.add_argument("-r", "--ref-start", type=float, help="[plot] Start of reference window in ns")
    p.add_argument("-R", "--ref-stop", type=float, help="[plot] Stop of reference window in ns")
    p.add_argument("-b", "--bg-ratio", type=float, help="[plot] Back ground ratio in %%")
    p.add_argument(
        "-m", "--method", type=str, help="[fit] Fitting method (threelevel). Invokes re-fitting."
    )
    p.add_argument(
        "-P",
        "--fit-params",
        type=str,
        help="[fit] Fitting parameters file name. Invokes re-fitting.",
    )
    p.add_argument("--color", type=str, nargs="+", help="matplotlib colors for data")
    p.add_argument("--color_fit", type=str, nargs="+", help="matplotlib colors for fitting lines")
    p.add_argument("--marker", type=str, nargs="+", help="matplotlib markers for data")
    p.add_argument("--linewidth", type=float, help="linewidth for data")
    p.add_argument("--linewidth_fit", type=float, default=1.0, help="linewidth for fitting line")
    p.add_argument("-O", "--offset", type=float, nargs="+", help="offset along y-axis")
    p.add_argument("-l", "--label", type=str, nargs="+", help="matplotlib labels")
    add_common_args(p, default_limits=((-100.0, 100.0), (0.0, None)))
    p.set_defaults(func=plot_hbt)


def add_spec_parser(sub_parsers):
    p = sub_parsers.add_parser("spec", help="Spectroscopy data.")
    p.add_argument(
        "-o", "--one-figure", metavar="FILENAME", type=str, help="Filename for one-figure mode"
    )
    p.add_argument(
        "-F", "--no-fit", dest="show_fit", action="store_false", help="Don't show fitting result"
    )
    p.add_argument("-n", "--filter-n", type=float, help="outlier filter's n")
    p.add_argument(
        "-m", "--method", type=str, help="[fit] Fitting method (single|multi). Invokes re-fitting."
    )
    p.add_argument(
        "-P",
        "--fit-params",
        type=str,
        help="[fit] Fitting parameters file name. Invokes re-fitting.",
    )
    p.add_argument(
        "-p",
        "--peak-type",
        type=str,
        default="voigt",
        help="[fit] Peak function for fitting (gaussian|lorentzian|voigt)",
    )
    p.add_argument("-N", "--n-peaks", type=int, default=2, help="[fit/multi] Number of peaks.")
    p.add_argument(
        "-g",
        "--n-guess",
        type=int,
        default=20,
        help="[fit] Number of samples to use for peak position guess",
    )
    p.add_argument("--color", type=str, nargs="+", help="matplotlib colors for data")
    p.add_argument("--color_fit", type=str, nargs="+", help="matplotlib colors for fitting lines")
    p.add_argument("--marker", type=str, nargs="+", help="matplotlib markers for data")
    p.add_argument("--linewidth", type=float, help="linewidth for data")
    p.add_argument("--linewidth_fit", type=float, default=1.0, help="linewidth for fitting line")
    p.add_argument("-O", "--offset", type=float, nargs="+", help="offset along y-axis")
    p.add_argument("-l", "--label", type=str, nargs="+", help="matplotlib labels")
    add_common_args(p)
    p.set_defaults(func=plot_spec)


def parse_args(args):
    parser = argparse.ArgumentParser(prog="mahos data plot", description="(Re-)plot data file(s).")
    sub_parsers = parser.add_subparsers(help="type of the files")
    add_confocal_image_parser(sub_parsers)
    add_confocal_trace_parser(sub_parsers)
    add_odmr_parser(sub_parsers)
    add_podmr_parser(sub_parsers)
    add_iodmr_parser(sub_parsers)
    add_iodmr_fit_parser(sub_parsers)
    add_hbt_parser(sub_parsers)
    add_spec_parser(sub_parsers)

    args = parser.parse_args(args)

    # why do we have to manually handle error when subcommand is not given.
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    return args


def main(args=None):
    args = parse_args(args)
    args.func(args)
