Glossary
========

.. currentmodule:: mahos

.. glossary::

   node
      Node is a program which is a part of distributed system. Implemented as a subclass of :class:`Node <node.node.Node>` or :class:`GUINode <gui.gui_node.GUINode>`.

   client
   node client
      Interface to :term:`node`'s functions. Implemented as a subclass of :class:`NodeClient <node.client.NodeClient>`.

   conf
      Configuration (dict) for something. Difference between params: conf is considered static (won't change run time).

   params
      Parameters (dict) for something. Difference between conf: params is considered dynamic (will change run time).

   gparams
   global params
      A dict with global parameters which is handled by :class:`GlobalParams <node.global_params.GlobalParams>`.

   Req-Rep
      Request-Reply communication pattern. The client sends a request and the server sends back a reply.

   RPC
      abbreviation of Remote Procedure Call. Req-Rep pattern is a typical RPC.

   Pub-Sub
      Publish-Subscribe communication pattern. One-to-many data distribution.

   status
      Messages expressing the node's status. Implemented as a subclass of :class:`Status <mahos.msgs.common_msgs.Status>`. Usually published as topic `status`.

   state
      Node can have explicit state. If so, it is implemented as a subclass of :class:`State <mahos.msgs.common_msgs.State>` and usually contained as a attribute of :term:`status`.
