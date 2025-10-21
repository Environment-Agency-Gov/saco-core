Overview
========

SACO stands for the Sustainable Abstraction Calculator and Optimiser tool. It aims to
help identify potential ways to meet environmental flow targets in Water Framework
Directive (WFD) surface waterbodies.

Introduction
------------

The tool is based around the Water Resources GIS (WRGIS) conceptualisation and tables.
For all surface waterbodies in England (and parts of Wales), WRGIS summarises
information on natural flows and the impacts of artificial influences scenarios on these
natural flows. WRGIS provides this information for selected percentiles of the flow
duration curve (FDC), rather than as time series.

.. note::

    We define artificial influences primarily as abstractions (from surface water
    and groundwater) and discharges - see :doc:`data` page for further details.

For some waterbodies, abstraction impacts may reduce flows below environmental flow
targets. At a high level, the role of the SACO tool is to help indicate potential ways
in which flows could be restored to meet targets in these cases.

The SACO tool can currently help explore two types of questions:

    1. What happens if we change a given set of artificial influence impacts in a
       certain way?
    2. How could we best change artificial influence impacts to meet flow targets?

When answering the first question, we bring our own impact changes to the tool
(hypothetical or otherwise). For example, we might want to see what happens to flows
relative to their targets if we were to reduce abstraction X's impact by 5%.

When answering the second question we formulate and solve an optimisation problem to
propose solutions. In this case, we are asking the tool to suggest a set of impact
changes, given an objective(s) and some constraints that we set.

It is also  possible to address questions that are some combination of (1) and (2).
For example, if we change a set of abstractions in a certain way (that we prescribe)
but some flow targets are still not met, what else could be done? Here we would like
the tool to suggest further changes to us to solve the remaining part of the problem
(i.e. after our user-prescribed impact changes have been taken into account).

.. note::

    The SACO tool operates in "WRGIS world". As a national data product, WRGIS involves
    a number of simplifications, assumptions and limitations. This must be borne in
    mind when using WRGIS/SACO and interpreting results.

Components
----------

Aligning with the two key questions outlined above, the tool currently has two core
components: a Calculator and an Optimiser.

The Calculator finds the impact of a given set of artificial influences on scenario
flows. Note that the set of artificial influences could be taken directly from WRGIS or
or it could be a modified/alternative set of influences, which might include
user-prescribed changes/overrides compared with the "base" WRGIS. The Calculator uses
scenario flows to identify surpluses/deficits and compliance relative to flow targets.

.. note::

    Scenario flows are defined as the flows that result from applying artificial
    influence impacts to natural flows. They might also be referred to as impacted or
    denaturalised flows.

The Optimiser identifies potential artificial influence changes that could achieve flow
targets by solving an optimisation problem. Again, the starting point could be the
"base" WRGIS tables or a modified/alternative version of "WRGIS-like" tables. Outputs
from this component include updated tables of how the artificial influence impacts were
changed by the solver to help meet flow targets, given an objective(s) and constraints.

The key distinction between the components is thus whether the user specifies if/how an
artificial influence impact should change (Calculator) or whether the model/solver
suggests how an artificial influence impact could be changed (Optimiser). As noted
above, it is possible to use the two components separately and/or in sequence. Further
explanations are given in the relevant sections of the documentation.
