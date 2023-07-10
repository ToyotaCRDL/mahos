=========
Changelog
=========

[0.2.0] - 2023-07-10
--------------------

On 0.2.0, we have got prepared for open-sourcing this project by following.

- The project is renamed from "meow2" to "mahos", in order to avoid potential trademark infringement.
- The git repository has been re-initialized to avoid exposing internal information of the company.

Added
^^^^^

- ParamDict type: provides functions like unwrap, flattened view, isclose comparison
- gui/basic_meas, gui/fit_widget based on it
- gui/pulse_monitor
- Buffer and Fit functions for BasicMeasNode (podmr, odmr, spectroscopy, hbt)
- Block/Blocks types for pulse generators
- odmr: background (differential) measurement
- file io: HDF5 (h5) file format
- settings of dev. tools: lint (flake8) and formatter (black)

Changed
^^^^^^^

- project name: meow2 to mahos
- gui: Switched from PyQt5 to PyQt6
- gui: Switched from QDarkStyle to BreezeStyleSheet
- podmr: almost rewrite the pulse generator and gui

Fixed
^^^^^

- inst/tdc: bug of set_sweep_preset
- inst/tdc: load_ctl is now load_config
- meas/odmr_fitter: avoid zero division
- gui/hbt,podmr: bug of disrupting UI

[0.1.0] - 2023-03-01
--------------------

Initial release
