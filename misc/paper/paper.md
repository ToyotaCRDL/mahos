---
title: 'MAHOS: Measurement Automation Handling and Orchestration System'
tags:
  - Python
  - instrumentation
  - measurement automation
  - laboratory automation
  - distributed systems
  - solid-state color center
authors:
  - name: Kosuke Tahara
    orcid: 0000-0002-9474-8970
    affiliation: 1
affiliations:
 - name: Toyota Central R&D Labs., Inc., Japan
   index: 1
date: 27 July 2023
bibliography: paper.bib
---

# Summary

The experimental researchers often face challenges building up the systems
to automate their measurement procedures
in physical science & engineering, or related areas.
While simple automation could be done by a single script,
systematic orchestration is required to handle complex cases.
The measurement automation typically requires several pieces of programs in diverse layers:
1) low-level drivers to communicate with the instruments,
2) high-level measurement logics or analysis algorithms, and
3) graphical user interfaces (GUIs) or top-level automation scripts.
The system may also involve multiple computers.
These demands suggest that a distributed system can serve as
a framework to help to write testable and maintainable programs.

# Statement of need

`MAHOS` is a Python package for distributed measurement automation.
Python has been a major language of choice for laboratory automation.
The low-level binding libraries such as PyVISA [@pyvisa] and PyDAQmx [@pydaqmx] are available for Python.
There are also several projects aiming to provide a comprehensive software stack
ranging from the drivers to the GUIs [@pymeasure; @qudi].
The target of `MAHOS` is similar to these libraries.
A key feature of `MAHOS` is that it provides a simple framework to construct distributed systems,
where the pieces of the programs are run on different processes (or threads), and talk to each other.
Due to this architecture, it is quite straightforward to set up multi-computer systems.
Moving from GUI operation to scripted automation is made seamless as well;
the GUI and top-level script (3rd layer) are seen as equivalent clients from a measurement logic (2nd layer).

Several concepts are inspired by the ROS project [@ros].
Each piece of the program is called a `node`.
Two patterns of communication methods are provided: Request-Reply (synchronous remote procedure call)
and Publish-Subscribe (asynchronous data distribution).
However, contrary to the ROS, the `MAHOS` library is kept rather small and simple.
Creating a new `node` is as easy as defining a Python class and no build process is required.
The system is considered `static` reflecting the purpose of measurement automation,
i.e., we assume all the possible nodes and communications beforehand.
This assumption enables the transparent configuration of the whole system; it can be written in a single TOML file.

Along with the base library above, `MAHOS` currently comes with a confocal microscope and
optically detected magnetic resonance (ODMR) measurement implementations
for research of solid-state color centers such as the Nitrogen-Vacancy (NV) center in diamond [@gruber1997; @jelezko2004; @wrachtrup2006].
The covered functionalities are similar to that of the qudi project [@qudi].

# References
