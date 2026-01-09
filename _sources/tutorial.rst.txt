Tutorial
========

This section introduces the code in the package and how it might be used. It is best to
read the preceding sections of the documentation first to get a more detailed
introduction to the tool components. See the Jupyter notebooks in the examples folder
of the repository for further code examples.

Imports
-------

The first step in using SACO is importing it as a package into a script/notebook. The
most important functionality becomes available with::

    from saco import Dataset, Calculator, Optimiser

From this we can see that the three most important components that a user might
interact with are:

    - ``Dataset``: provides a "container" to group and manipulate multiple data tables.
    - ``Calculator``: takes a ``Dataset`` as input, calculates "scenario flows" and
      assesses their compliance against environmental flow targets.
    - ``Optimiser``: takes a ``Dataset`` as input then formulates and solves an
      optimisation problem to find abstraction impact reductions needed to meet flow
      targets.

We provide an initial guide to these objects/components below. See the "Reference"
sections of the documentation for further details, including information about additional
helper functions.

.. note::

    The backgrounds to the Calculator and Optimiser components are explained in more
    detail in the :doc:`calculator` and :doc:`optimiser` sections of the documentation.

Dataset
-------

A ``Dataset`` is primarily used to store and group together the relevant data tables
(i.e. primarily WRGIS tables). One way to get going with a ``Dataset`` (and the tool
in general) is to load a folder of data table files::

    ds = Dataset(data_folder='/path/to/my/data-folder')
    ds.load_data()

Here we have created a ``Dataset`` object (as ``ds``) and loaded data into memory. As a
``Dataset`` object is the main input to the Calculator and Optimiser components of the
tool, we could actually now go ahead and run those components. But first let us look a
bit more into what a ``Dataset`` consists of.

Tables
~~~~~~

A Dataset object has individual data tables as its most important attributes. For example,
``ds.swabs`` provides access to the SWABS_NBB table of surface water abstractions from
WRGIS (with "swabs" being the "short name" for SWABS_NBB). Similar attributes exist for
the other key WRGIS tables listed below:

    - ``asbs``: AbsSensBands_NBB (waterbody abstraction sensitivity bands)
    - ``asb_percs``: ASBPercentages (fractional deviations defining the reference flows
      - typically environmental flow indicator, EFI)
    - ``dis``: Discharges_NBB (point surface water discharges)
    - ``gwabs``: GWABs_NBB (point groundwater abstractions)
    - ``qnat``: QNaturalFlows_NBB (waterbody natural flows)
    - ``refs``: REFS_NBB (reference flows - typically EFI)
    - ``sup``: SupResGW_NBB (point "complex impacts")
    - ``swabs``: SWABS_NBB (point surface water abstractions)
    - ``wbs``: IntegratedWBs_NBB (waterbody metadata)

An additional table that is not in WRGIS is derived and included in a ``Dataset``:

    - ``mt``: Master (waterbody summary table - water balance terms, compliance, etc)

The Master table is intended to be the key waterbody-level table that brings together
the water balance components with information on surplus/deficit and compliance
classifications.

The tables have various properties, but most importantly each table class possesses a
``data`` attribute, which is a pandas.DataFrame. Therefore, to access the dataframe of
surface water abstractions, we can use ``ds.swabs.data``. We can then query or
manipulate the data table as we would any pandas.DataFrame. A description of the column
indexes and fields is given in :doc:`fields`.

Changing Numbers
~~~~~~~~~~~~~~~~

A user may wish to changes numbers in a ``Dataset`` to improve the data or to test the
implications of known, planned, hypothetical or other types of prescribed changes. Two
ways to change the numbers in a ``Dataset`` are:

    - Modify the numbers in data table files on disk and then load the ``Dataset``
      using syntax like the ``load_data`` example above.
    - Modify a dataframe directly using a table's ``data`` attribute, as described in
      the preceding paragraph.

An example of the latter approach to change a surface water abstraction impact would be::

    ds.swabs.data.loc[ds.swabs.data.index == 'swab-unique-id', 'SWQ95FLWR'] = 1.23

This would change the abstraction impact under the fully licensed (FL) artificial
influences scenario at the 95th (natural) flow percentile to 1.23 (Ml/d). But the
dataframe could be queried or manipulated in different ways, including through pandas
merge/join operations etc.

.. note::

    The Master (``mt``) table and its data should not be set or edited directly by a
    typical user (in general). See below about (1) using the Calculator to obtain an
    updated Master table and (2) using specific methods to ensure that a Dataset and
    its Master table are ready to go into the Optimiser.

.. note::

    If a user changes a surface or groundwater abstraction impact in a ``Dataset``
    under a given artificial influences scenario and at a given (natural) flow
    percentile, any long-term average abstraction columns in the relevant table are
    *not* automatically updated (currently). See :doc:`reference-dataset` for
    explanation of the available options to make this calculation if needed.

Other Functionality
~~~~~~~~~~~~~~~~~~~

The ``Dataset`` possesses additional functionality to help set table values, write
tables to output files, work with the "network" of waterbodies, and prepare for input
to the Optimiser component. This functionality (the "methods" of ``Dataset``) are
described in :doc:`reference-dataset`.

Calculator
----------

Once a ``Dataset`` has been loaded or constructed (potentially with modifications
relative to the "base" WRGIS), it can be supplied as input to the ``Calculator``. As
demonstrated below, the ``run`` method of the ``Calculator`` can then be executed to
calculate scenario flows, surpluses/deficits and compliance bands based on the input
``Dataset``::

    calculator = Calculator(ds)
    output_dataset = calculator.run()

In the example above we rely on default arguments, which will run the ``Calculator``
for all scenario/percentile combinations and for the whole domain in the input
``Dataset`` (i.e. all waterbodies present). It will also run with some default
methodological choices (see below for more on this).

By default, the ``run`` method returns a complete ``Dataset`` with an updated Master
table (i.e. one that is consistent with all the other tables in the ``Dataset``). If we
want to save this ``Dataset`` (i.e. all of its component tables) at this point, we could
do so as follows (see :doc:`reference-dataset` for guidance on output options)::

    output_dataset.write_tables(output_folder='/my/output/folder')

However, if we want to customise the execution of the ``Calculator``, we can provide
optional arguments, as described in :doc:`reference-calculate`. One such argument
defined on initialisation of the ``Calculator`` is named ``capping_method``. This
argument controls the approach to "unfeasible" impacts - prescribed abstraction impacts
that cannot be satisfied. See :doc:`calculator` for a more precise explanation of this
point. By default, the Calculator takes a WRGIS-like approach to this issue. Using the
:doc:`reference-calculate` documentation to understand the available options, we could
override the default and take WBAT-like approach as follows::

    calculator = Calculator(ds, capping_method='simple')

We might then for example ask the ``Calculator.run`` method to return only an updated
Master table as a pandas.DataFrame::

    updated_master_table = calculator.run(master_only=True)

These are just a couple of examples of customisation via optional arguments - see
:doc:`reference-calculate` for more options and details.

.. note::

    If capping has been applied to avoid propagation of unfeasible impacts, scenario
    flows output from the Calculator may be larger than initially expected from
    performing a simple "ups" water balance calculation for some waterbodies. This is
    because capping reduced net impacts upstream to retain physical plausibility. The
    Master table gives the "target" artificial influences components, which are not
    capped individually.

.. note::

    Any long-term average fields in a ``Dataset`` passed to the ``Calculator`` are *not*
    changed by the execution of the ``Calculator.run`` method currently (i.e. they are
    unchanged in the ``Dataset`` or Master table returned).

Optimiser
---------

The role of the Optimiser is to suggest how impacts could best be adjusted to meet flow
targets, given some objective(s) and constraints. The solution to this problem is
obtained via mixed integer (binary) linear programming.

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

The starting point for the ``Optimiser`` is again a ``Dataset``. However, in this case
we need to ensure that certain columns are present in some tables - columns that are not
necessarily relevant to the ``Calculator``. The relevant tables and columns are
(currently):

    - Master table: requires a flow target column(s) (optional for the Calculator).
    - GWABs_NBB table: requires a flag column to indicate whether a given impact (row)
      should be available for change in the optimisation (1) or not (0).
    - SWABS_NBB table: as per GWABs_NBB table.
    - SupResGW_NBB table: requires a flag column - we return to this below.

See :doc:`fields` for a guide to the naming conventions for these columns.

The relevant columns can be added or set using the methods in the example snippet below
(assuming still that we have a ``Dataset`` instance as ``ds``)::

    ds.set_flow_targets()
    ds.set_optimise_flag()

Called in this way, both of these methods will use their default settings, which are
described in :doc:`reference-dataset`. Both methods have optional arguments that can be
used to customise flow targets and flag which abstraction impacts will be
included/excluded in optimisation.

.. note::

    If any further manipulation of the inclusion/exclusion flag is needed it could be
    achieved by working with the relevant dataframes (i.e. ``ds.swabs.data``,
    ``ds.gwabs.data`` and ``ds.sup.data``).

.. note::

    For users interested in Environmental Destination (ED) modelling, it should be noted
    that the default behaviour of the ``Dataset.set_optimise_flag`` method does not
    mimic exactly the configuration used in ED modelling for the second National
    Framework for Water Resources. It is important for a user to check that they are
    happy with the flag setup.

.. note::

    Waterbody targets can be (optionally) specified via a separate *Fix_Flags* table
    that sits within a ``Dataset`` (accessible via the short name ``wbfx``). If present
    in a ``Dataset``, the flags in this table will be used to customise flow targets
    (i.e. permitting further relaxation beyond the flow in REFS_NBB). See :doc:`fields`
    for more details.

Optional Arguments
~~~~~~~~~~~~~~~~~~

Once we are happy that a ``Dataset`` is ready for the ``Optimiser``, we could invoke the
run method of the ``Optimiser`` as below::

    optimiser = Optimiser(ds)
    output_dataset = optimiser.run()
    output_dataset.write_tables(output_folder='/my/output/folder')

However, the :doc:`reference-optimise` section provides information on options that we
may want to customise when setting up the ``Optimiser`` (i.e. before execution). One
such option concerns the geographical domain considered. By default, the code above
will run the ``Optimiser`` for the whole domain contained in the input ``Dataset``. The
following lines provide an example of how to run for only part of the domain (referring
to the most downstream waterbody of interest as an "outlet")::

    outlet_waterbody = 'outlet-waterbody-id'  # could be a list of outlet waterbodies

    selected_waterbodies = ds.identify_upstream_waterbodies(outlet_waterbody)

    optimiser = Optimiser(ds, domain=selected_waterbodies)
    output_dataset = optimiser.run()

Other options can be specified too, such as which artificial influences scenario(s) and
flow percentile(s) should be considered. Options also exist concerning the objectives
of the optimisation and whether any "relaxation" should be applied when attempting to
solve for a secondary objective - see :doc:`optimiser`.

Outputs
~~~~~~~

The contents of the output from ``Optimiser.run`` are similar to a normal ``Dataset``,
apart from (keeping complex impacts to one side for the moment):

    - The SWABS_NBB and GWABs_NBB tables now contain abstraction impacts as they are
      the optimisation has been completed (i.e. the impacts that remain after the
      “fix”).
    - Similarly, the Master table summarises the water balance and compliance etc for
      the solution formulated by the ``Optimiser``.
    - Additional tables are present: SWABS_Changes and GWABS_Changes (accessible via
      the output ``Dataset``'s attributes ``swabs_chg`` and ``gwab_chg``, respectively.
      These tables contain the impact reductions (Ml/d) required relative to a
      “reference” ``Dataset`` - see :doc:`reference-optimise`.

.. note::

    Long-term average abstraction is recalculated after optimisation under the
    assumption that the relative impact profile across the FDC remains constant.
    However, SWABS with hands-off flow (HOF) conditions are omitted from the
    recalculation at present to avoid introducing a conservative bias into the
    estimates. See :doc:`reference-dataset` and :doc:`reference-optimise` for more
    details.

Complex Impacts
~~~~~~~~~~~~~~~

Exploratory and optional functionality has been included to allow specific types of
complex impacts to be included in optimisation. As noted above, the SupResGW_NBB table
thus requires a flag column to indicate whether a given complex impact (row) should be:

    - 0: Excluded from optimisation
    - 1: Included as a reservoir compensation flow increase
    - 2: Included as a complex abstraction

The default is for all complex impacts to be excluded from optimisation. Thus, the
``Dataset.set_optimise_flag`` method will insert a flag column of zeros, if a column
has not already been added to the table.

If we wish to include specific complex impacts in optimisation, we need to manually
adjust the flag column for the appropriate rows. This can be achieved in memory using
dataframe operations, for example::

    ds.sup.data.loc[ds.sup.data.index == 'complex-impact-id', 'Optimise_Flag'] = 1

If we specify that a reservoir compensation flow increase is allowed in optimisation,
we also need to indicate that maximum increase permitted. This should be done for the
relevant scenarios and percentiles under consideration. For example, to permit a maximum
reservoir compensation flow increase of 5.0 Ml/d, we could do the following::

    ds.sup.data.loc[ds.sup.data.index == 'complex-impact-id', 'SUPFLQ95_MAX_INCREASE'] = 5.0

Once the flag field and any maximum increase fields have been added to the relevant
``Dataset`` table, the ``Optimiser`` can be run in the way described above. Outputs are
as detailed above, with the addition of a SupResGW_Changes table that gives any changes
to complex impacts. Note that positive numbers in this table indicate increases to
compensation flows or *reductions in complex abstraction impacts*. The latter sign
convention is opposite to that for SWABS_Changes and GWABS_Changes. It arises because
complex impacts can be positive or negative.

Some important points to note about the behaviour of complex impact functionality in the
``Optimiser`` are:

    - Use of this functionality will typically require local knowledge of specific
      impacts *and* how they have been represented in CAMS ledgers and WRGIS.
    - In its current implementation, the Optimiser seeks to minimise increases to
      reservoir compensation flows in its solution for several reasons. It will only
      increase a compensation flow if it helps to meet flow targets that would have
      otherwise been missed. Other approaches may be explored in future.
    - Complex abstractions (flag = 2) are treated in the same way as normal surface and
      groundwater abstractions. The Optimiser identifies required impact reductions for
      the scenario/percentile combination in question, rather than changes to complex
      licence conditions.
    - A user has the option to explore the implications of different prescribed changes
      to complex impacts by editing the relevant rows in the data table and excluding
      them from optimisation.
