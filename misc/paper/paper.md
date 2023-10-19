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
in physical science and engineering, or related areas.
While simple automation could be done by a single script,
systematic orchestration is required to handle complex cases.
The measurement automation typically requires several pieces of programs in diverse layers:
1) low-level drivers to communicate with the instruments,
2) high-level measurement logics or analysis algorithms, and
3) graphical user interfaces (GUIs) or top-level automation scripts.
The system may also involve multiple computers.
It would become very difficult to write debuggable, testable, and maintainable programs
on a monolithic framework.
To solve this kind of problem in measurement automation,
`MAHOS` provides a flexible framework to build distributed systems consisting of small independent programs.

# Statement of need

`MAHOS` is a Python framework for building distributed measurement systems.
Python has been a major language of choice for laboratory automation.
The low-level binding libraries for instruments, such as PyVISA [@pyvisa] and PyDAQmx [@pydaqmx], are available for Python.
There are also several projects aiming to provide a comprehensive software framework
ranging from the instrument drivers to the GUIs [@pymeasure; @qudi; @nspyre].

While we believe that a distributed framework would result in a more maintainable and flexible system,
the existing libraries provide rather monolithic ones.
That is why we have built `MAHOS` to provide a flexible framework for constructing distributed systems.
On the `MAHOS`-based system, several pieces of the programs are run on different processes (or threads)
and talk to each other.
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

# Acknowledgements

We acknowledge Prof. Mutsuko Hatano and Prof. Takayuki Iwasaki for their support in developing the predecessor project at Tokyo Institute of Technology.
We are also grateful to Wataru Naruki, Kosuke Mizuno, and Ryota Kitagawa for contributing to and maintaining it.
We thank Katsuhiro Kutsuki and Taishi Kimura for the opportunity and support to start up this project.

# References
