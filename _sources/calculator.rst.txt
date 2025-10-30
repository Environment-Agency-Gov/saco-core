Calculator
===========

The Calculator finds the impact of a given set of artificial influences on scenario
flows. These scenario flows are then used to identify surpluses/deficits and compliance
relative to flow targets.

The set of artificial influences used in the Calculator could take a couple of forms:

    - Artificial influences could be taken directly from the "base" WRGIS.
    - A modified, alternative or new set of artificial influences could be used. These
      influences might include user-prescribed changes/overrides compared with the
      "base" WRGIS.

It is anticipated that the Calculator will primarily be used to evaluate the latter of
these two cases. Scenario flows, surpluses/deficits and compliance classifications in
the former case are an output from the "upstream" WRGIS toolset and available in the
WRGIS database.

The Calculator is based on a programmatic implementation of the "Waterbody Abstraction
Tool" (WBAT) Excel workbook. The WBAT is a faster way to examine the effects of
artificial influence changes compared with rerunning the full WRGIS toolset code. See
below for further details of the relationship between the core logic of the WRGIS
toolset, the WBAT and the SACO Calculator.

.. note::

    The Calculator uses WRGIS abstraction impact fields that already account for (local)
    consumptiveness. This should be remembered if abstraction impacts are modified.

Terminology
-----------

We follow WRGIS use of "sub" and "ups" terminology. "sub" refers to just the current
waterbody, whereas "ups" refers to sums for the current waterbody and all waterbodies
upstream of it. So, taking natural flow as an example, "sub" represents runoff
generated within the waterbody itself. Conversely, "ups" represents the "actual"
outflow from the waterbody (including all of the area upstream of the waterbody
outlet).

Logic
-----

The starting point for the Calculator is a dataset that contains the tables described
in :doc:`data`. As noted above, the artificial influences tables may contain changes
relative to the "base" WRGIS tables.

From this point, the Calculator follows a few steps to arrive at scenario flows,
surpluses/deficits and compliance classifications:

    1. Aggregate the updated artificial influences (point) tables to waterbody-level
       (first "sub" and then "ups", using the routing matrix).
    2. Apply the "ups" impacts to the natural ("ups") flows to derive updated
       scenario flows (also "ups") for each waterbody. The tool also identifies the
       consistent "sub" scenario flows for reference.
    3. Calculate updated surplus/deficit numbers for each waterbody as the new scenario
       flow minus the EFI. If an alternative flow target table/column has been
       specified then surplus/deficit relative to this target can also be calculated at
       the same time.
    4. Identify the compliance band associated with the new scenario flow. The band is
       obtained by dividing the surplus/deficit (relative to the EFI) by the natural
       flow ("ups") and then comparing this quantity with the band definitions in the
       table below.

*Compliance band definitions based on deficit (D) and natural flow (Qn)*

=========   ===============================
Band        Definition
=========   ===============================
Compliant   D / Qn >= 0
1           -0.25 <= D / Qn < 0
2           -0.5 <= D / Qn < -0.25
3           D / Qn < -0.5
=========   ===============================

.. note::

    Waterbodies with natural flows of zero are set as "compliant". Selected waterbody
    types are not evaluated for compliance: 'Seaward Transitional' and 'Saline Lagoon'.

The core of the Calculator thus follows the simple routed (instantaneous) water balance
approach of WRGIS noted in :doc:`data`. I.e. scenario flows are essentially the result
of subtracting abstractions and adding discharges to natural flows, while ensuring that
impacts upstream are propagated to downstream waterbodies. The primary output from the
Calculator is a "Master" table - indexed by waterbody - that provides all water balance
terms, surpluses/deficits and compliance bands.

Capping
-------

One question relevant to the WRGIS toolset, the WBAT and the SACO Calculator concerns
the impacts on flows downstream of a waterbody in which the "prescribed" impact
(according to the artificial influences tables) cannot be met. For example, say a
waterbody has a scenario flow of 10 Ml/d coming in from upstream (i.e. after all
artificial influences *upstream* of the waterbody are taken into account). Say then that
the impact of artificial influences inside that waterbody ("sub") in a given scenario
is prescribed as -15 Ml/d. Say also that the waterbody does not generate any runoff
inside its area. The net impact (abstraction) therefore exceeds the available flow by
5 Ml/d.

In this example, the WBAT would effectively take the available 10 Ml/d, giving a
scenario flow of zero for the waterbody. However, it would also then propagate a -15
Ml/d impact to downstream waterbodies, which would be subtracted from available flows
if possible (i.e. up to the limit of the available flow in the downstream waterbodies).
In contrast, the WRGIS toolset would limit the net impact propagated to downstream
waterbodies to -10 Ml/d. I.e. the impact would be "capped" to what can actually be
taken in the waterbody in question.

The WBAT works in this way partly because "capping" would be harder to implement in
Excel. It might be expected that the WBAT provides slightly more conservative results
than WRGIS where cases of "capping" arise in WRGIS. This is a situation that does
occur with non-negligible frequency in the base WRGIS (and higher frequency in the
fully licensed abstraction scenario).

In the SACO Calculator we have provided options to follow both the WBAT approach and a
WRGIS-like approach to this situation. The latter approach is the default and involves
capping net impacts at the waterbody scale, rather than adjusting individual artificial
influences at the point scale. An upstream-to-downstream loop adjusts net impacts by
the minimum amount needed to ensure that scenario flows do not become negative. We
adopt this approach to emulate the WRGIS toolset calculations as far as possible, but
an alternative could involve using a network flow model to solve for all flows and
feasible impacts directly.

.. note::

    If capping has been applied to avoid propagation of unfeasible impacts, scenario
    flows output from the Calculator may be larger than initially expected from
    performing a simple "ups" water balance calculation for some waterbodies. This is
    because capping reduced net impacts upstream to retain physical plausibility.
