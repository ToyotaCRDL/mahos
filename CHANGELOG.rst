=========
Changelog
=========

[Unreleased]
------------

Added
^^^^^

- meas: added new measurement Qdyne.
- inst/tdc: added raw event recording features for above.
- meas,gui: added new generic node Tweaker.
- inst: new module lockin and LI5640 class in it.
- inst/overlay: new module amplified_pd.

Changed
^^^^^^^

- make config key names consistent for several Instruments or measurement Workers.
  Suffix key name with '_dir' for directory path configs.

  - inst/piezo/E727_3: limits_um -> limit_um, range -> range_um
  - inst/camera/ThorlabsCamera: dll_path -> dll_dir
  - inst/dtg/DTG: root_local -> local_dir, root_remote -> remote_dir
  - inst/pd/LUCI10,LUCI_OE200: dll_path -> dll_dir
  - inst/tdc/MCS: home -> mcs_dir, home_raw_events -> raw_events_dir
  - meas/qdyne_worker -> home_raw_events -> raw_events_dir

- inst/InstrumentServer, Instrument, InstrumentOverlay

  - overlay can reference other overlays.
  - new standard API functions get_param_dict() and get_param_dict_labels().
    Instrument can also use ParamDicts with this.
  - signature of configure(): new kwargs label and group to make it usable with get_param_dict().

Removed
^^^^^^^

- gui/oe200: replaced with much more generic tweaker gui.

[0.2.2] - 2023-07-26
--------------------

Fixed README and metadata. No code is changed since 0.2.1.

[0.2.1] - 2023-07-26
--------------------

0.2.1 is the first release after open-sourcing this project.

Added
^^^^^

- inst/daq: AnalogIn and CounterDivider classes
- inst/pd: LUCI10 and OE200 classes

Changed
^^^^^^^

- APDCounter has been moved from inst.daq to inst.pd.
- meas/confocal,odmr: can be used not only with APDCounter, but also analog-PDs like OE200.
- meas/confocal_io: image export puts the c-unit label (like cps) at colorbar, not as title.
- docs: directory structure has been changed for github-pages publication.

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
