Data
====

As noted in the :doc:`overview`, the SACO tool is based around the Water Resources GIS
(WRGIS) conceptualisation and tables. Here we provide some further background on WRGIS
and its use in the SACO tool.

.. note::

    WRGIS data *may* be made available under licence following a data request to the
    Environment Agency.

WRGIS
-----

WRGIS summarises information on natural flows and the impacts of artificial influences
on these natural flows. WRGIS provides this information for selected percentiles of the
flow duration curve (FDC) for all Water Framework Directive (WFD) surface waterbodies
in England. These percentiles are (lowest flow to highest flow): 95, 70, 50, 30.
The WRGIS database does not contain time series information.

Impacts are provided for three artificial influence scenarios, which may be defined
approximately as:

    - Recent actual: mean impacts over a recent ~6-year period
    - Fully licensed: hypothetical impacts if all abstractions were to take place at
      their maximum rates (according to licence conditions)
    - Future predicted: recent actual impacts multiplied by a growth factor to estimate
      potential changes

Artificial influences are primarily surface water abstractions, groundwater
abstractions, and (surface water) discharges. However, a fourth class of "complex"
influences is also delineated in WRGIS. One example of a complex impact might be the
impounding effect of a reservoir on a natural flow regime. This impact does not
necessarily fall easily into the simpler categories of abstractions and discharges.
More discussion on this class of influences is given below.

WRGIS is constructed by a set of tools that process information from the Catchment
Management Strategy (CAMS) "ledgers". These ledgers describe the impacts of artificial
influences on the FDC at so-called CAMS Assessment Points (APs). The WRGIS tools
translate the information on natural flows and artificial influences from the ledgers
to the WFD waterbody scale. This processing results in a (geo)database. When referring
to WRGIS throughout the SACO package/documentation, we are referring to the database
rather than the processing toolset.

Tables
~~~~~~

The data tables in the WRGIS database that are relevant to the SACO tool are listed in
the summary table below:

========================    ==========================================================
Table Name                  Notes
========================    ==========================================================
IntegratedWBs_NBB           Defines waterbody network and waterbody properties
QNaturalFlows_NBB           Natural flows per waterbody
AbsSensBands_NBB            Abstraction sensitivity bands (ASBs) per waterbody
ASBPercentages              Permitted deviations (fractional) from natural flow by ASB
                            and flow percentile
SWABS_NBB                   Surface water abstractions
GWABs_NBB                   Groundwater abstractions
Discharges_NBB              Discharges
SupResGW_NBB                Complex impacts
========================    ==========================================================

This summary indicates that the SACO tool uses three tables that are indexed by
waterbody (i.e. one row per waterbody): IntegratedWBs_NBB, QNaturalFlows_NBB and
AbsSensBands_NBB. ASBPercentages is essentially a metadata table that facilitates
calculation of the environmental flow indicator (EFI) for each waterbody. The remaining
four tables of artificial influences are indexed by "point". For example, a given
abstraction might have multiple point locations associated with it. The impact of each
of these points is entered on one row in the relevant table.

All tables except ASBPercentages have a "wide" format, i.e. they may contain multiple
value columns and no "factor" columns. For example, the abstractions tables contain a
separate value column for the impact of each combination of scenario and percentile.

.. note::

    An abstraction licence may also include multiple "purpose codes" that define the
    permitted reason for abstraction. The abstraction tables are therefore technically
    given on a licence-point-purpose basis.

.. note::

    Some artificial influences are highly complex, so care is needed when interpreting
    the various tables in WRGIS.

Conceptualisation
~~~~~~~~~~~~~~~~~

Here we summarise a few relevant aspects of the WRGIS conceptual model:

    - Individual surface water abstraction points, (surface water) discharge points
      and complex influence points can impact one waterbody.
    - Discharges do not vary as a function of percentile or scenario (RA and FL only)
      in general. Values tend to represent dry weather flows or consented discharges,
      but this may vary between discharges and areas.
    - Complex impacts may be positive (increasing flow) or negative (decreasing flow).
    - Groundwater abstractions may impact between one and five waterbodies according to
      a fixed set of proportions (summing to 100%). These proportions are constant
      across flow percentiles and abstraction scenarios.
    - Total artificial influence impacts are disaggregated across the FDC (selected
      percentiles) with reference to known/presumed seasonality.
    - Scenario flows are defined as the flows resulting after artificial influences
      have been applied to natural flows. These are effectively denaturalised flows.
    - Scenario flows are calculated through a simple water balance for a given
      combination of artificial influences scenario (RA, FL or FP) and percentile
      (95, 70, 50, 30).
    - This is equivalent to an instantaneous propagation of impacts through the network
      of waterbodies. I.e. there are no time-dependent considerations in the "routing"
      of flows and impacts. Implicitly, all waterbodies thus experience the same flow
      percentile at the same time.

Limitations
~~~~~~~~~~~

As noted in the :doc:`overview`, the SACO tool operates in "WRGIS world". Indeed, the
tool explicitly seeks to preserve the "rules" of this world. Yet, as a national data
product, WRGIS encodes a number of simplifications and assumptions, including those
outlined in the previous section. These features are inherited by the SACO tool. It is
also bound by limitations of the data, i.e. the uncertainties, approximations and
errors involved in translating observed or estimated quantities into the WRGIS database.

It is important to note that the SACO tool is "downstream" of WRGIS. This means that it
takes WRGIS tables as inputs. One implication of this is that the SACO tool is
"downstream" of the process that assigns an impact to each artificial influence at each
flow percentile to create the "base" WRGIS. (As noted above, this assignment is
undertaken by the WRGIS toolset using information from the CAMS ledgers.) The SACO tool
can explore the implications of changing specific numbers in the tables (via the
Calculator and Optimiser components outlined in the :doc:`overview`), but it does not
contain the logic/data by which the WRGIS toolset assigns the "base" impacts during
construction of the WRGIS database.

Processing
----------

Functions for processing the raw data tables from the WRGIS database (currently in
Access format) into the format required by SACO are not included within the package
currently. To summarise this functionality, the main processing steps are:

    - Extract tables and perform basic checks of indexes and important columns. Columns
      not required in the SACO tool are filtered out.
    - Convert waterbody relationships into a directed graph (``networkx.DiGraph``).
      This provides useful helper functions/methods for working with the network.
    - Calculate other derived quantities for convenience, including environmental flow
      indicators (EFIs) for each waterbody and flow percentile (as a function of
      abstraction sensitivity band, which defines a permitted fractional deviation from
      the natural flow).

The processing code writes a set of output files to a specified directory. The data
tables are generally saved in parquet format, the network graph is saved in graphml
format, and a numpy helper array (routing matrix) is saved in compressed npz format. A
directory containing these files forms the main input to the SACO tool.

.. note::

    EFIs are calculated as per WRGIS. This is consistent with the CAMS ledgers at the
    95th flow percentile, but some (typically small) divergence is possible at higher
    flow percentiles. The calculations in the CAMS ledgers use additional data that are
    not as readily available at the waterbody scale. However, work is underway to
    harmonise the WRGIS/SACO method of EFI calculation with the CAMS ledger method for
    full consistency above the 95th percentile.

Further Details
---------------

See the :doc:`fields` and :doc:`tutorial` pages for more explanation of each of the
tables involved in SACO input/output. Synthetic examples of the tables/data (with no
relationship to any real waterbodies or artificial influences) are available in the
repository (under tests/data). These examples can be read in and viewed using the
notebook in the examples folder of the repository.
