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
    - Required abstraction impact changes (an Optimiser output) are given as negative
      to indicate that an impact needs to be reduced relative to its reference / starting
      point. So -10 in SWABS_Changes or GWABS_Changes means reduce an impact by 10 Ml/d
      to get to the solution found by the Optimiser.

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
    - DISCHRA: *Discharge impact under RA scenario [Ml/d]*
    - DISCHFL: *Discharge impact under FL scenario [Ml/d]*
    - DISCHFP: *Discharge impact under FP scenario [Ml/d]*
    - EA_WB_ID: *Impacted waterbody ID*
    - DISNUMBER: *Consent number*
    - SITENAME: *Site name*
    - CONSHOLDER: *Consent holder name*

GWABs_NBB
    - UNIQUEID: *Unique ID (index column)*
    - EA_WB_ID: *ID of waterbody containing impact point*
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
    - GWQ{P}{S}WR: *Total abstraction impact at flow percentile {P} under scenario {S}
      [Ml/d]*
    - LICNUMBER: *Licence number*
    - SITENAME: *Site name*
    - PURPCODE: *Abstraction purpose code*
    - LICHOLDER: *Licence holder*
    - FLPTPANQM3: *Point-purpose annual quantity under FL scenario [m^3]*
    - RAPTPANQM3: *Point-purpose annual quantity under RA scenario [m^3]*
    - GWPROPCONS: *Abstraction consumptiveness [-]*
    - IMPFAC: *Factor/flag used in seasonal apportionment of impacts*

IntegratedWBs_NBB
    - EA_WB_ID: *Waterbody ID (index column)*
    - DSTREAM_WB: *ID of next waterbody (downstream)*
    - Type_IWB: *Waterbody type*
    - RBD_NAME: *River basin district name*

QNaturalFlows_NBB
    - EA_WB_ID: *Waterbody ID (index column)*
    - QN{P}sub: *Natural flow (sub) at percentile {P} [Ml/d]*
    - QN{P}ups: *Natural flow (ups) at percentile {P} [Ml/d]*

SupResGW_NBB
    - UNID: *Unique ID (index column)*
    - PURPOSE: *Impact purpose*
    - EA_WB_ID: *Impacted waterbody ID*
    - SUP{S}Q{P}: *Impact at flow percentile {P} under scenario {S} [Ml/d]*
    - NAME: *Impact name*
    - OPERATOR: *Operator name*
    - TYPE_SUPRESGW: *Complex impact type*

SWABS_NBB
    - UNIQUEID: *Unique ID (index column)*
    - EA_WB_ID: *Impacted waterbody ID*
    - SWQ{P}{S}WR: *Total abstraction impact at flow percentile {P} under scenario {S}
      [Ml/d]*
    - HOFMLD: *Hands-off flow (in HOFWBID) at which abstraction impact ceases [Ml/d]*
    - HOFWBID: *ID of waterbody that defines hands-off flow for this abstraction*
    - LICNUMBER: *Licence number*
    - SITENAME: *Site name*
    - PURPCODE: *Abstraction purpose code*
    - LICHOLDER: *Licence holder name*
    - FLPTPANQM3: *Point-purpose annual quantity under FL scenario [m^3]*
    - RAPTPANQM3: *Point-purpose annual quantity under RA scenario [m^3]*
    - SWPROPCONS: *Abstraction consumptiveness [-]*
    - RESRVRFLAG: *Flag indicating whether abstraction associated with reservoir*
    - SW_LDMU_NO: *Flag indicating whether abstraction associated with level-dependent
      management unit*
    - SW_LAKE1: *Flag indicating whether abstraction associated with lake 1 (refers to
      ledger numbering)*
    - SW_LAKE2: *As SW_LAKE1 but for lake 2*
    - SW_LAKE3: *As SW_LAKE1 but for lake 3*
    - SW_LAKE4: *As SW_LAKE1 but for lake 4*
    - SW_LAKE5: *As SW_LAKE1 but for lake 5*

Derived Table Fields
--------------------

EFI
    - EA_WB_ID: *Waterbody ID (index column)*
    - EFIQ{P}: *Environmental flow indicator at flow percentile {P} [Ml/d]*

Master
    - COMP{S}Q{P}: *Compliance band (0 = compliant, 1/2/3 = band 1/2/3, -999 = unassessed
      due to type)*
    - DISCH{S}{agg}: *Discharge impacts [Ml/d]*
    - EFIQ{P}: *Environmental flow indicator [Ml/d]*
    - GW{S}Q{P}{agg}: *Groundwater abstraction impacts [Ml/d]*
    - QN{P}{agg}: *Natural flow[Ml/d]*
    - QT{S}Q{P}: *Flow target (which may differ from EFI) [Ml/d]*
    - SCEN{S}Q{P}{agg}: *Scenario flow (i.e. impacted/denaturalised) [Ml/d]*
    - SD{S}Q{P}: *Surplus/deficit relative to EFI [Ml/d]*
    - SDT{S}Q{P}: *Surplus/deficit relative to target flow (i.e. QT{S}Q{P}) [Ml/d]*
    - SUP{S}Q{P}{agg}: *Complex impacts [Ml/d]*
    - SW{S}Q{P}{agg}: *Surface water abstraction impacts [Ml/d]*

.. note::

    The Optimiser output tables SWABS_Changes and GWABS_Changes list the impact
    reductions required relative to a reference (typically the input / starting point).
    These tables follow the format of SWABS_NBB and GWABs_NBB, respectively, except
    their value columns represent impact changes, rather than impacts themselves.
    Negative values in these tables of changes indicate that an impact is reduced.

Optimiser Required Fields
-------------------------

As noted in :doc:`tutorial`, some extra columns are needed in certain ``Dataset`` tables
before the Optimiser can be run:

    - SWABS_NBB and GWABs_NBB require an additional (boolean) field called
      Optimise_Flag . This field indicates whether a given abstraction should be
      included (1) or excluded (0) from the optimisation process.
    - The Master table requires a flow target column(s) of the form QT{S}Q{P}
      (for a scenario {S} and a percentile {P}). This is typically set by
      ``Dataset.set_flow_targets``.
