=========
Changelog
=========

[Unreleased]
------------

Added
^^^^^

- meas: new generic measurement Recorder.
- meas: new generic measurement PosTweaker.
- inst: new module dmm (for Digital Multi Meter).
- inst: new module power_sensor (for RF / MW power sensor).
- inst: new module positioner.
- inst: new module filter_wheel.
- inst.tdc: new instrument TimeTagger (Swabian Instruments).
- inst.fg: new instrument SIGLENT_SDG2000X.
- inst.pg: new instrument SpinCore_PulseBlasterESR_PRO.
- inst.spectrometer: new instrument Andor_Spectrometer.

Changed
^^^^^^^

- Python version: drop support of 3.8 and 3.9, and add support of 3.12.
- inst,meas: argument "group" has been removed from get_param_dict_labels(), get_param_dict(),
  and configure() in meas / inst APIs.
- inst: argument "label" has been added to start(), stop(), reset(), set(), and get() inst API.
- inst.fg: class name DG2000 -> RIGOL_DG2000.
- inst.spectrometer: class name LightField -> Princeton_LightField.

[0.2.3] - 2023-11-06
--------------------

Added
^^^^^

- meas: new measurement Qdyne.
- meas: new measurement SPODMR (Pulse ODMR with Slow detectors).
- inst.tdc: raw event recording features for above.
- meas,gui: new generic node Tweaker (for manual instrument parameter tuning).
- msgs.data_msgs.Data: new reserved attr '_inst_params' for parameter injection by Tweaker.
- inst: new module lockin and LI5640 class in it.
- inst.pd: new class LockinAnalogPD.
- meas.confocal,odmr: accept complex-value data (of LockinAnalogPD).
- inst.overlay: new module amplified_pd (to use AnalogPDs with configurable amplifiers).
- inst.fg: ParamDict-based interface.
- meas.odmr: new parameters delay and sg_modulation.
- util: new modules graph and conf.
- cli: auto-start of log (LogBroker) in mahos log and auto-exclude of log in mahos launch.
- meas.podmr: changed definition of plot mode 'normalize' (difference / average).
- node: custom serialization of the message (other than default of pickle).

Changed
^^^^^^^

- (breaking config) make config key names consistent for several Instruments or measurement Workers.
  Suffix key name with '_dir' for directory path configs.

  - inst.piezo.E727_3: limits_um -> limit_um, range -> range_um.
  - inst.camera.ThorlabsCamera: dll_path -> dll_dir.
  - inst.dtg.DTG: root_local -> local_dir, root_remote -> remote_dir.
  - inst.pd.LUCI10,LUCI_OE200: dll_path -> dll_dir.
  - inst.tdc.MCS: home -> mcs_dir, home_raw_events -> raw_events_dir.
  - meas.qdyne_worker -> home_raw_events -> raw_events_dir.

- (breaking config) node class and preferred name changes (46a58d8).

  - LogBroker's preferred node name: 'log_broker' -> 'log'.
    Replace '[localhost.log_broker]' / 'localhost::log_broker' with '[localhost.log]' / 'localhost::log'.
  - Module and class: mahos.node.param_server.ParamServer -> mahos.node.global_params.GlobalParams.
    Config target key and preferred name: param_server -> gparams.
    Replace '[localhost.param_server]' with '[localhost.gparams]'.
    Replace '[localhost.somenode.target] param_server = "localhost::param_server"' with '[localhost.somenode.target] gparams = "localhost::gparams"'.

- inst.InstrumentServer, Instrument, InstrumentOverlay.

  - overlay can reference other overlays as well as raw Instrument.
  - add new standard API methods get_param_dict() and get_param_dict_labels().
    Instrument can also use ParamDicts with this.
  - signature of configure(): add new kwargs label and group to make it usable with get_param_dict().

- (breaking config) DTG classes in inst.dtg module has been moved to inst.pg module.

- common_msgs: Resp class has been renamed to Reply.

- Common directories has been moved from ~/.config/mahos or ~/.cache/mahos/log to ~/.mahos/config ~/.mahos/log .

  - The home directory (~/.mahos) has become configurable with environment var "MAHOS_HOME".

Removed
^^^^^^^

- gui.oe200: use much more generic tweaker gui instead.

[0.2.2] - 2023-07-26
--------------------

Fixed README and metadata. No code is changed since 0.2.1.

[0.2.1] - 2023-07-26
--------------------

0.2.1 is the first release after open-sourcing this project.

Added
^^^^^

- inst.daq: AnalogIn and CounterDivider classes.
- inst.pd: LUCI10 and OE200 classes.

Changed
^^^^^^^

- APDCounter has been moved from inst.daq to inst.pd.
- meas.confocal,odmr: can be used not only with APDCounter, but also analog-PDs like OE200.
- meas.confocal_io: image export puts the c-unit label (like cps) at colorbar, not as title.
- docs: directory structure has been changed for github-pages publication.

[0.2.0] - 2023-07-10
--------------------

On 0.2.0, we have got prepared for open-sourcing this project by following.

- The project is renamed from "meow2" to "mahos", in order to avoid potential trademark infringement.
- The git repository has been re-initialized to avoid exposing internal information of the company.

Added
^^^^^

- ParamDict type: provides functions like unwrap, flattened view, isclose comparison
- gui.basic_meas, gui.fit_widget based on it
- gui.pulse_monitor
- Buffer and Fit functions for BasicMeasNode (podmr, odmr, spectroscopy, hbt)
- Block and Blocks types for pulse generators
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

- inst.tdc: bug of set_sweep_preset
- inst.tdc: load_ctl is now load_config
- meas.odmr_fitter: avoid zero division
- gui.hbt,podmr: bug of disrupting UI

[0.1.0] - 2023-03-01
--------------------

Initial release
