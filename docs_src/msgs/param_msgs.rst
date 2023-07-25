mahos.msgs.param\_msgs
======================

.. automodule:: mahos.msgs.param_msgs

.. currentmodule:: mahos.msgs.param_msgs

Messages
--------

.. autoclass:: GetParamDictReq

.. _param-dict:

ParamDict
---------

Measurement parameter can be reported by using ParamDict.

.. autoclass:: ParamDict
.. autoclass:: PDValue
.. autoclass:: RawPDValue

Contents of ParamDict is instances of Param.
The Param is mutable object; we can get / set the value.

.. autoclass:: Param

   .. automethod:: Param.value
   .. automethod:: Param.set

Param has subclasses according to the types.

NumberParam
^^^^^^^^^^^

NumberParam has bounds: minimum and maximum.

.. autoclass:: NumberParam

   .. automethod:: NumberParam.minimum
   .. automethod:: NumberParam.maximum

.. autoclass:: IntParam

.. autoclass:: FloatParam

ChoiceParam
^^^^^^^^^^^^^^

.. autoclass:: ChoiceParam

   .. automethod:: ChoiceParam.options

.. autoclass:: BoolParam

.. autoclass:: EnumParam

.. autoclass:: StrChoiceParam

.. autoclass:: IntChoiceParam

Other Param
^^^^^^^^^^^

.. autoclass:: StrParam

.. autoclass:: UUIDParam

Functions
^^^^^^^^^

.. autofunction:: unwrap
.. autofunction:: isclose
