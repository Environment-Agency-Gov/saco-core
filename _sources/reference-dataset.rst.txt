Dataset
=======

A *Dataset* class is used to group WRGIS-like data tables together and perform common
read/write/modify operations. The calculator and optimiser components of the tool take
a *Dataset* as their main input.

.. currentmodule:: saco

.. autoclass:: Dataset
   :members: wbs, qnat, asbs, swabs, gwabs, dis, sup, asb_percs, efi, sfac, mt

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated

      ~Dataset.find_outlet_waterbodies
      ~Dataset.identify_upstream_waterbodies
      ~Dataset.infer_mean_abstraction
      ~Dataset.load_data
      ~Dataset.set_flow_targets
      ~Dataset.set_optimise_flag
      ~Dataset.set_tables
      ~Dataset.write_tables

   .. rubric:: Attributes

   .. autosummary::

      ~Dataset.asbs
      ~Dataset.asb_percs
      ~Dataset.dis
      ~Dataset.refs
      ~Dataset.gwabs
      ~Dataset.mt
      ~Dataset.qnat
      ~Dataset.sfac
      ~Dataset.sup
      ~Dataset.swabs
      ~Dataset.wbfx
      ~Dataset.wbs
