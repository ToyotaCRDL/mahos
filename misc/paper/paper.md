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
to automate their measurement procedures in physical science and engineering, or related areas.
While simple automation could be done by a single script, systematic orchestration is required to handle complex cases.
The measurement automation typically requires several pieces of programs in diverse layers:
1) low-level drivers to communicate with the instruments,
2) high-level measurement logics or analysis algorithms, and
3) graphical user interfaces (GUIs) or top-level automation scripts.

A `modular` framework is necessary to write maintainable programs for this purpose.
On such a framework, the pieces of codes are organized within high-level abstraction mechanisms (module, class, etc.)
and the interfaces are cleanly defined.

The `distributed` messaging is one of the modern approaches for building extensible computing or control system.
In this approach, individual programs run on multiple computers and they exchange the data by using messaging mechanism.
Each program works in concert with the other to realize the system's functions.

Although there are several `modular` frameworks for laboratory automation,
the `distributed` messaging approach is not widely recognized and tested in this field.
`MAHOS` tries to bring these two concepts together and
provide a `modular` and `distributed` framework for measurement automation.

# Statement of need

`MAHOS` is a `modular` and `distributed` framework for building measurement systems in Python.
Python has been a major language of choice for laboratory automation.
The low-level binding libraries for instruments, such as PyVISA [@pyvisa] and PyDAQmx [@pydaqmx], are available for Python.
There are also several projects aiming to provide the comprehensive `modular` framework
ranging from the instrument drivers to the GUIs [@pymeasure; @qudi; @nspyre].
While we believe that the `distributed` messaging could result in more extensible and flexible systems,
the existing libraries take rather centralized approaches.
That is why we have built `MAHOS` to provide both `modular` and `distributed` framework for measurement automation.

On the `MAHOS`, several programs run as different processes
and exchange the data by using a distributed messaging system.
Two distributed system concepts are inspired by the ROS project [@ros].
First, each piece of the program is called a `node`.
Second, two patterns of communication methods are provided for the nodes: Request-Reply (synchronous remote procedure call)
and Publish-Subscribe (asynchronous data distribution).
The ZeroMQ library is currently utilized as the messaging backend [@zeromq].

Contrary to the ROS, the `MAHOS` library is kept rather small and simple.
Creating a new node is as easy as defining a Python class and no build process is required.
The system is considered `static` reflecting the purpose of measurement automation,
i.e., we assume to know all the possible nodes and messages beforehand.
This assumption enables the transparent configuration of the whole system; it can be written in a single TOML configuration file.

Since the messages can be delivered across computers even with different platforms (operating systems),
we can build up multi-computer system easily and flexibly.
For example, if an instrument has the driver only for Windows but we have to use Linux for rest of the system,
we can run a driver node on the Windows host and access it from the other nodes on the Linux.
All it takes to move a node from one host to another is to change a few lines in the configuration file.

The data transparency is the best effect brought to `MAHOS` by adopting the distributed messaging approach.
Notable examples can be listed as below.

- The data published by measurement logics (nodes in 2nd layer defined in Summary) can be A) visualized in the GUI (3rd layer) node
and B) processed and analyzed in interactive console such as IPython or Jupyter, simultaneously.
- It is straightforward to conduct an initial measurement using GUI node, and then write ad-hoc script to run many measurements
with different parameters because the GUI node and the script are seen as equivalent clients from the measurement logic node.
The GUI will visualize the data even when the measurement is started by the script.
- We can inspect the instrument's status or perform ad-hoc operations on the instrument at run time without shutting down any nodes.
This is because the operation requests from measurement logic nodes and ad-hoc ones are equivalent from the perspective of instrument driver (1st layer) node.

The transportation overhead could be the downside of networked messaging
if one deals with very large data such as high resolution images produced at high rate.
However, this overhead can be reduced significantly by running the relevant nodes as threads in single process,
and use intra-process transportation.
This switching can be performed by only modifying and adding several lines in the configuration file
thanks to the ZeroMQ providing both networked (e.g. TCP) and intra-process transportations.

Along with the base framework above, `MAHOS` currently comes with a confocal microscope and
optically detected magnetic resonance (ODMR) measurement implementations
for research of solid-state color centers such as the Nitrogen-Vacancy (NV) center in diamond [@gruber1997; @jelezko2004; @wrachtrup2006].
The covered functionalities are similar to that of the qudi project [@qudi].

# Acknowledgements

We acknowledge Prof. Mutsuko Hatano and Prof. Takayuki Iwasaki for their support in developing the predecessor project at Tokyo Institute of Technology.
We are also grateful to Wataru Naruki, Kosuke Mizuno, and Ryota Kitagawa for contributing to and maintaining it.
We thank Katsuhiro Kutsuki and Taishi Kimura for the opportunity and support to start up this project.

# References
