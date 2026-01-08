Table Fields
============

Here we summarise the table fields/columns typically used in SACO inputs and outputs.
Most of the tables are from WRGIS and follow its formatting/naming conventions. The
small number of non-WRGIS tables used as input/output for the tool are also included
here. See :doc:`data` for a more general overview of the tables.

Several WRGIS tables use a "wide" table structure, in which multiple value columns are
present (rather than say a single value column but multiple "factor" columns). We use
the following substitutions in the field names presented below:

    - {P}: flow percentile (natural) {95, 70, 50, 30}
    - {S}: artificial influences scenario {RA, FL, FP}
    - {agg}: type of aggregation for a waterbody-scale quantity {sub, ups}

.. note::

    In WRGIS, "sub" and "ups" suffixes are used in relation to some waterbody-scale
    quantities. Using flow as an example, an "ups" number indicates total flow at the
    waterbody outlet, i.e. including all upstream contributing area. A "sub" number
    refers only to flow generated within the waterbody in question (so something like
    locally generated runoff). Artificial influence impacts can be given as either
    "ups" or "sub" numbers too.

Sign Conventions
----------------

Note the following sign conventions used in the various tables:

    - Abstraction impacts (i.e. acting to decrease flows) are entered as positive in
      SWABS_NBB and GWABs_NBB (and in the groundwater and surface water abstraction
      columns of the Master table).
    - Discharge impacts (i.e. acting to increase flows) are entered as positive in
      Discharges_NBB (and in the discharge columns of the Master table).
    - Complex impacts in SupResGW_NBB are positive if representing a discharge (or
      other impact increasing flows) and negative if representing an abstraction (or
      other impact decreasing flows). This is also the case for the net complex impact
      terms in the Master table.
    - Required abstraction impact changes in SWABS_Changes or GWABS_Changes (an
      Optimiser output) are given as negative to indicate that an impact needs to be
      reduced relative to its reference / starting point. So -10 in these tables means
      reduce an impact by 10 Ml/d to get to the solution found by the Optimiser.
    - Required impact changes in SupResGW_Changes are positive numbers. These numbers
      indicate either an increase in reservoir compensation flow or a required
      *reduction* in a complex abstraction, depending on the nature of the table row.
      The sign convention for complex abstraction changes is thus different to
      that in SWABS_Changes and GWABS_Changes.

WRGIS Table Fields
------------------

Where relevant, units are indicated in square brackets. "[-]" indicates a dimensionless
or fractional quantity. Columns that are used as dataframe indexes in SACO are also
indicated.

AbsSensBands_NBB
    - EA_WB_ID: *Waterbody ID (index column)*
    - ASBFinal: *Abstraction sensitivity band (per waterbody)*

ASBPercentages
    - QFLOW: *Flow percentile (natural) (index column)*
    - ASB: *Abstraction sensitivity band (index column)*
    - PERCENT\_: *Fractional permitted deviation from natural flow [-]*

Discharges_NBB
    - UNID: *Unique ID (index column)*
    - CONSHOLDER: *Consent holder name*
    - DISCH{S}: *Discharge impact under scenario {S} [Ml/d]*
    - DISNUMBER: *Consent number*
    - EA_WB_ID: *Impacted waterbody ID*
    - SITENAME: *Site name*

GWABs_NBB
    - UNIQUEID: *Unique ID (index column)*
    - EA_WB_ID: *ID of waterbody containing impact point*
    - GWLTA{S}NR: *Long-term average abstraction (no local returns) under scenario {S}
      [Ml/d]*
    - GWPROPCONS: *Abstraction consumptiveness [-]*
    - GWQ{P}{S}WR: *Total abstraction impact at flow percentile {P} under scenario {S}
      (accounting for local consumptiveness) [Ml/d]*
    - IMPFAC: *Factor/flag used in seasonal apportionment of impacts*
    - LICHOLDER: *Licence holder*
    - LICNUMBER: *Licence number*
    - LICN_EXPD: *Licence expiry date or flag (D for deregulated)*
    - PURPCODE: *Abstraction purpose code*
    - SITENAME: *Site name*
    - WB_1ST: *First waterbody receiving impact*
    - WB_2ND: *Second waterbody receiving impact*
    - WB_3RD: *Third waterbody receiving impact*
    - WB_4TH: *Fourth waterbody receiving impact*
    - WB_5TH: *Fifth waterbody receiving impact*
    - WB_1ST_PRO: *Percentage of total impact received by WB_1ST [%]*
    - WB_2ND_PRO: *Percentage of total impact received by WB_2ND [%]*
    - WB_3RD_PRO: *Percentage of total impact received by WB_3RD [%]*
    - WB_4TH_PRO: *Percentage of total impact received by WB_4TH [%]*
    - WB_5TH_PRO: *Percentage of total impact received by WB_5TH [%]*

IntegratedWBs_NBB
    - EA_WB_ID: *Waterbody ID (index column)*
    - AREA_NAME: *EA area name*
    - CATCHMENT: *Catchment (WRGIS boundaries/definitions)*
    - DSTREAM_WB: *ID of next waterbody (downstream)*
    - Ledger_Are: *EA ledger area*
    - Outflowx: *Waterbody outflow point x-coordinate (BNG) [m]*
    - OutflowY: *Waterbody outflow point y-coordinate (BNG) [m]*
    - RBD_NAME: *River basin district name*
    - Type_IWB: *Waterbody type*
    - UpsArea_m2: *Area upstream of waterbody outlet [m^2]*
    - XCent: *Waterbody centroid x-coordinate (BNG) [m]*
    - Ycent: *Waterbody centroid y-coordinate (BNG) [m]*

QNaturalFlows_NBB
    - EA_WB_ID: *Waterbody ID (index column)*
    - QN{P}sub: *Natural flow (sub) at percentile {P} [Ml/d]*
    - QN{P}ups: *Natural flow (ups) at percentile {P} [Ml/d]*

REFS_NBB
    - EA_WB_ID: *Waterbody ID (index column)*
    - REFSQ{P}: *Reference flow (typically EFI = environmental flow indicator) at flow
      percentile {P} [Ml/d]*

Seasonal_Lookup
    - LOOKUP: *Identifier to look up percentile impact factor for SWABS according to
      abstraction start and end months (index column)*
    - SFAC{P}: *SWABS impact factor for percentile {P} [-]*

SupResGW_NBB
    - UNID: *Unique ID (index column)*
    - EA_WB_ID: *Impacted waterbody ID*
    - NAME: *Impact name*
    - OPERATOR: *Operator name*
    - PURPOSE: *Impact purpose*
    - SUP{S}Q{P}: *Impact at flow percentile {P} under scenario {S} [Ml/d]*
    - TYPE_SUPRESGW: *Complex impact type*

SWABS_NBB
    - UNIQUEID: *Unique ID (index column)*
    - EA_WB_ID: *Impacted waterbody ID*
    - ENDMON: *End month for abstraction profile*
    - HOFMLD: *Hands-off flow (in HOFWBID) at which abstraction impact ceases [Ml/d]*
    - HOFWBID: *ID of waterbody that defines hands-off flow for this abstraction*
    - LICHOLDER: *Licence holder name*
    - LICNUMBER: *Licence number*
    - LICN_EXPD: *Licence expiry date or flag (D for deregulated)*
    - PURPCODE: *Abstraction purpose code*
    - RESRVRFLAG: *Flag indicating whether abstraction associated with reservoir*
    - SITENAME: *Site name*
    - STARTMON: *Start month for abstraction profile*
    - SWLTA{S}NR: *Long-term average abstraction (no local returns) under scenario {S}
      [Ml/d]*
    - SWPROPCONS: *Abstraction consumptiveness [-]*
    - SWQ{P}{S}WR: *Total abstraction impact at flow percentile {P} under scenario {S}
      (accounting for local consumptiveness) [Ml/d]*
    - SW_LAKE{i}: *Flag indicating whether abstraction associated with lake {i} (refers to
      ledger numbering, with {i} = {1, 2, 3, 4, 5})*
    - SW_LDMU_NO: *Flag indicating whether abstraction associated with level-dependent
      management unit*

Optional Table Fields
---------------------

Fix_Flags
    - EA_WB_ID: *Waterbody ID (index column)*
    - Fix_Flag: *Flag indicating level of "fix" to aim for in optimisation (3 =
      compliant, 0 = no deterioration, -1 = no fix required)*

.. note::

    The Fix_Flags table is intended to align with the "fix" target options and flag
    conventions used in the Environmental Destination modelling for the second
    National Framework for Water Resources. This table is not essential for the
    Optimiser, but if available it will be used to guide how flow targets are set.

Derived Table Fields
--------------------

Master
    - EA_WB_ID: *Waterbody ID (index column)*
    - COMP{S}Q{P}: *Compliance band (0 = compliant, 1/2/3 = band 1/2/3, -999 = unassessed
      due to type)*
    - DISCH{S}{agg}: *Discharge impacts [Ml/d]*
    - GW{S}Q{P}{agg}: *Groundwater abstraction impacts (accounting for local
      consumptiveness) [Ml/d]*
    - QN{P}{agg}: *Natural flow [Ml/d]*
    - QT{S}Q{P}: *Flow target (which may differ from EFI) [Ml/d]*
    - REFSQ{P}: *Reference flow (typically EFI) [Ml/d]*
    - SCEN{S}Q{P}{agg}: *Scenario flow (i.e. impacted/denaturalised) [Ml/d]*
    - SD{S}Q{P}: *Surplus/deficit relative to reference flow [Ml/d]*
    - SDT{S}Q{P}: *Surplus/deficit relative to target flow (i.e. QT{S}Q{P}) [Ml/d]*
    - SUP{S}Q{P}{agg}: *Complex impacts [Ml/d]*
    - SW{S}Q{P}{agg}: *Surface water abstraction impacts (accounting for local
      consumptiveness) [Ml/d]*

.. note::

    The Optimiser output tables SWABS_Changes, GWABS_Changes and SupResGW_Changes list
    the impact changes required relative to a reference (typically the input / starting
    point). These tables follow the format of SWABS_NBB, GWABs_NBB and SupResGW_NBB,
    respectively, except their value columns represent impact changes, rather than
    impacts themselves. See above for sign conventions.

Optimiser Required Fields
-------------------------

As noted in the :doc:`tutorial`, some extra columns are needed in certain ``Dataset``
tables before the Optimiser can be run:

    - The Master table requires a flow target column(s) of the form QT{S}Q{P}
      (for a scenario {S} and a percentile {P}). This is typically set by
      ``Dataset.set_flow_targets``.
    - SWABS_NBB, GWABs_NBB and SupResGW_NBB require an additional field called
      Optimise_Flag . For the former two tables, this field indicates whether a given
      abstraction should be included (1) or excluded (0) from the optimisation process.
      Things are slightly more complicated for SupResGW_NBB, as explained in the
      :doc:`tutorial`. Defaults for the flag field can be set with the
      ``Dataset.set_optimise_flag`` method.
    - If any reservoir compensation flow increases are to be considered in the
      optimisation, an additional field specifying the maximum permitted increase is
      required. This should take the form SUP{S}Q{P}_MAX_INCREASE .
