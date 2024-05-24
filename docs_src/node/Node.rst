mahos.node.node.Node
====================

.. currentmodule:: mahos.node.node

.. autoclass:: Node

   .. rubric:: Methods to be overridden

   .. automethod:: Node.main
   .. automethod:: Node.close_resources

   .. rubric:: Important methods

   .. automethod:: Node.add_rep
   .. automethod:: Node.add_pub
   .. automethod:: Node.add_clients

   .. rubric:: Other methods

   .. autosummary::

      ~Node.__init__
      ~Node.close
      ~Node.fail_with
      ~Node.main_event
      ~Node.main_interrupt
      ~Node.poll
      ~Node.set_shutdown
      ~Node.wait

   .. rubric:: Attributes

   .. autosummary::

      ~Node.CLIENT
      ~Node.TOPIC_TYPES
