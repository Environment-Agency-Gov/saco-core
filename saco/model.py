"""
Data/array preparation and model to solve a single optimisation problem.

"""
from copy import deepcopy
from typing import Union, List, Dict, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import scipy.sparse as sp
import pandas as pd
import cvxpy as cp

from .config import Constants
from .tables import Table
from .dataset import Dataset, subset_dataset_on_wbs, subset_dataset_on_columns
from .calculate import Calculator


class Flows(Table):
    """
    Inflows to network (natural flows adjusted for static impacts).

    """
    def set_schema(self, *args):
        pass

    @staticmethod
    def get_qmod_column(value_type):
        return f'QM{value_type}'

    @property
    def name(self) -> str:
        return 'Flows'

    @property
    def short_name(self) -> str:
        return 'flows'

    @property
    def index_name(self) -> Union[str, List[str]]:
        return self.constants.waterbody_id_column

    @property
    def qt_column(self):
        return 'QT'

    @property
    def arc_index_column(self):
        return self.constants.arc_index_column


class GWABS(Table):
    """
    Input groundwater impacts (demands) (row per GWAB + impacted waterbody combinations).

    """
    def set_schema(self, *args):
        pass

    @property
    def name(self) -> str:
        return 'GWABS'

    @property
    def short_name(self) -> str:
        return 'gwabs'

    @property
    def index_name(self) -> Union[str, List[str]]:
        return ['UNIQUEID', 'IMPACT_NUMBER']

    @property
    def waterbody_id_column(self) -> str:
        return self.constants.waterbody_id_column

    @property
    def proportion_column(self):
        return 'PROPORTION'

    @property
    def impact_column(self):
        return 'IMPACT'

    @property
    def arc_index_column(self):
        return self.constants.arc_index_column


class SWABS(Table):
    """
    Input surface water abstraction impacts (demands).

    """
    def set_schema(self, *args):
        pass

    @property
    def name(self) -> str:
        return 'SWABS'

    @property
    def short_name(self) -> str:
        return 'swabs'

    @property
    def index_name(self) -> Union[str, List[str]]:
        return 'UNIQUEID'

    @property
    def waterbody_id_column(self) -> str:
        return self.constants.waterbody_id_column

    @property
    def impact_column(self):
        return 'IMPACT'

    @property
    def hof_value_column(self) -> str:
        return 'HOFMLD'

    @property
    def hof_waterbody_column(self) -> str:
        return 'HOFWBID'

    @property
    def arc_index_column(self):
        return self.constants.arc_index_column


class DataPreparer:
    """
    Production of formatted inflows and abstraction impacts (demands) for model.

    Args:
        ds: Input Dataset.
        scenario: Name/abbreviation of artificial influences scenario.
        percentile: Flow percentile (natural).
        waterbodies: Waterbodies in domain to be used.
        constants: By default instance of config.Constants.

    """
    def __init__(
            self, ds: Dataset, scenario: str, percentile: int, waterbodies: List[str],
            constants: Constants = None,
    ):
        self._input_ds = ds
        self.scenario = scenario
        self.percentile = percentile
        self.waterbodies = waterbodies

        if constants is None:
            constants = Constants()
        self.constants = constants

        # Trim dataset down to relevant waterbodies and value columns
        _ds = subset_dataset_on_wbs(self._input_ds, self.waterbodies)
        self.ds = subset_dataset_on_columns(_ds, [scenario], [percentile])
        self.ds_opt = None  # will include only swabs and gwabs to be optimised

        self.flows = Flows()
        self.swabs = SWABS()
        self.gwabs = GWABS()

        self.subdomain_indices = {}  # to help evaluate equality on subdomains

        self.formatted_data = None  # becomes dictionary returned by run method

    def filter_supresgw(self):
        """Filter out complex impact purposes that are excluded in WBAT."""

        df = self.ds.sup.data.copy()
        df = df.loc[
            ~df[self.ds.sup.purpose_column].isin(self.ds.sup.purposes_to_exclude)
        ]
        self.ds.sup.set_data(df)

    def prepare_flows(self):
        """Derive model inflows by modifying natural flow with static impacts."""

        # Set any swabs/gwabs to be optimised to zero - i.e. only retain abs that should
        # be held constant when modifying natural flows. This gives a set of static
        # swabs/gwabs that should modify the natural flow (alongside any discharges and
        # complex impacts that are held constant).
        df1 = self.ds.swabs.data.copy()
        value_col = self.ds.swabs.get_value_column(self.scenario, self.percentile)
        df1.loc[df1[self.ds.swabs.optimise_flag_column] == 1, value_col] = 0.0

        df2 = self.ds.gwabs.data.copy()
        value_col = self.ds.gwabs.get_value_column(self.scenario, self.percentile)
        df2.loc[df2[self.ds.gwabs.optimise_flag_column] == 1, value_col] = 0.0

        ds1 = deepcopy(self.ds)
        ds1.swabs.set_data(df1)
        ds1.gwabs.set_data(df2)

        # Setting capping_method to None would give potentially negative flows if
        # abstractions held constant exceed natural flows + discharges. Instead obtain
        # "feasible" impacts via capping
        calculator = Calculator(
            input_dataset=ds1, scenarios=[self.scenario],
            percentiles=[self.percentile], capping_method='cap-net-impacts',
        )
        ds2 = calculator.run()

        scen_col = self.ds.mt.get_scen_column(
            self.scenario, self.percentile, self.constants.ups_abb,
        )
        if np.any(ds2.mt.data[scen_col] < 0.0):
            raise NotImplementedError(
                'Constant impacts (negative) exceed natural flow + discharges '
                'in one or more waterbodies (ups) - should not occur.'
            )

        # Relax flow target if flow target now unattainable
        qt_col = self.ds.mt.get_qt_column(self.scenario, self.percentile)
        if np.any(ds2.mt.data[scen_col] < ds2.mt.data[qt_col]):
            _wbs = ds2.mt.data.loc[ds2.mt.data[scen_col] < ds2.mt.data[qt_col]].index.tolist()
            _n_wbs = len(_wbs)
            warnings.warn(
                f'Some flow targets ({_n_wbs}) cannot be met (likely due to static '
                f'impacts). These targets have been dropped - check outputs and/or set '
                f'lower targets if desired: {_wbs}'
            )
            ds2.mt.data[qt_col] = np.where(
                ds2.mt.data[scen_col] < ds2.mt.data[qt_col],
                0.0,
                ds2.mt.data[qt_col]
            )

        df = self.format_flows(ds2.mt)
        self.flows.set_data(df)

    def format_flows(self, mt):
        """Helper method to format inflows for Flows table."""

        scen_sub_col = mt.get_scen_column(
            self.scenario, self.percentile, self.constants.sub_abb,
        )
        scen_ups_col = mt.get_scen_column(
            self.scenario, self.percentile, self.constants.ups_abb,
        )
        qt_col = mt.get_qt_column(self.scenario, self.percentile)

        df = mt.data[[scen_sub_col, scen_ups_col, qt_col]].copy()
        df = df.rename(columns={
            scen_sub_col: self.flows.get_qmod_column(self.constants.sub_abb),
            scen_ups_col: self.flows.get_qmod_column(self.constants.ups_abb),
            qt_col: self.flows.qt_column,
        })

        return df

    def filter_abstractions(self):
        """Filter abstractions tables to just those rows to be optimised."""

        ds1 = self.ds

        # Need to retain metadata columns and just the (one) relevant value column
        # - greater than zero check should avoid trying to optimise pseudo-discharges
        df1 = ds1.swabs.data.loc[ds1.swabs.data[ds1.swabs.optimise_flag_column] == 1]
        value_col = ds1.swabs.get_value_column(self.scenario, self.percentile)
        cols_to_omit = [c for c in ds1.swabs.value_columns if c != value_col]
        df1 = df1.loc[df1[value_col] > 0.0, ~df1.columns.isin(cols_to_omit)].copy()

        df2 = ds1.gwabs.data.loc[ds1.gwabs.data[ds1.gwabs.optimise_flag_column] == 1]
        value_col = ds1.gwabs.get_value_column(self.scenario, self.percentile)
        cols_to_omit = [c for c in ds1.gwabs.value_columns if c != value_col]
        df2 = df2.loc[df2[value_col] > 0.0, ~df2.columns.isin(cols_to_omit)].copy()

        if (df1.shape[0] == 0) and (df2.shape[0] == 0):
            raise ValueError('No abstraction impacts to optimise.')

        self.ds_opt = deepcopy(self.ds)

        self.ds_opt.swabs.set_data(df1)
        self.ds_opt.gwabs.set_data(df2)

    def prepare_gwabs(self):
        """
        Form GWABS impacts (demands) table.

        We take the overall impact of a GWAB and its proportional split then calculate
        the impact on each waterbody. So get to one row per GWAB + impacted waterbody
        combination.

        """
        index_col = self.gwabs.index_name[0]
        impact_number_col = self.gwabs.index_name[1]
        waterbody_col = self.gwabs.waterbody_id_column
        proportion_col = self.gwabs.proportion_column
        impact_col = self.gwabs.impact_column

        dc = {
            index_col: [],
            impact_number_col: [],
            waterbody_col: [],
            proportion_col: [],
            impact_col: [],
        }

        impacted_wb_cols = self.ds_opt.gwabs.impacted_waterbody_columns
        impact_prop_cols = self.ds_opt.gwabs.impact_proportion_columns
        value_col = self.ds_opt.gwabs.get_value_column(self.scenario, self.percentile)

        for idx, row in self.ds_opt.gwabs.data.iterrows():
            for i in range(len(impacted_wb_cols)):
                impacted_wb_col = impacted_wb_cols[i]
                impact_prop_col = impact_prop_cols[i]

                if (
                        (row[impacted_wb_col] is None)
                        or (row[impacted_wb_col].strip() == '')
                ):
                    break
                elif (
                        row[impacted_wb_col].startswith('AP')
                        or row[impacted_wb_col].startswith('GB')
                ):
                    if row[impacted_wb_col] in self.waterbodies:
                        if row[impact_prop_col] > 0.0:
                            dc[index_col].append(idx)
                            dc[waterbody_col].append(row[impacted_wb_col])
                            dc[impact_number_col].append(i + 1)
                            dc[proportion_col].append(row[impact_prop_col] / 100.0)
                            dc[impact_col].append(
                                row[value_col] * row[impact_prop_col] / 100.0
                            )
                else:
                    wb = row[impacted_wb_col]
                    raise ValueError(
                        f'Unknown waterbody naming convention (expect AP... or GB...): '
                        f'{wb}'
                    )

        df = pd.DataFrame(dc)
        df = df.set_index(self.gwabs.index_name)

        self.gwabs.set_data(df)

    def prepare_swabs(self):
        """Form SWABS impacts (demands) table."""

        waterbody_col = self.ds_opt.swabs.waterbody_id_column
        value_col = self.ds_opt.swabs.get_value_column(self.scenario, self.percentile)
        hof_value_col = self.ds_opt.swabs.hof_value_column
        hof_waterbody_col = self.ds_opt.swabs.hof_waterbody_column

        df = self.ds_opt.swabs.data.loc[:, [
            waterbody_col,
            value_col,
            hof_value_col,
            hof_waterbody_col,
        ]].copy()

        df = df.rename(columns={
            waterbody_col: self.swabs.waterbody_id_column,
            value_col: self.swabs.impact_column,
            hof_value_col: self.swabs.hof_value_column,
            hof_waterbody_col: self.swabs.hof_waterbody_column,
        })
        df.index.name = self.swabs.index_name

        # If the flow target for a waterbody exceeds a HOF flow defined on that
        # waterbody then by default the HOF will be respected under hard flow
        # constraints (i.e. flow has to remain above the target, which is greater than
        # the HOF flow). So in these cases HOFs do not need to be explicitly modelled
        df = df.merge(
            self.flows.data[[self.flows.qt_column]], how='left',
            left_on=self.swabs.hof_waterbody_column,right_index=True,
        )
        df.loc[
            df[self.flows.qt_column] >= df[self.swabs.hof_value_column],
            self.swabs.hof_value_column
        ] = 0.0
        df = df.drop(columns=self.flows.qt_column)

        self.swabs.set_data(df)

    def define_index(self):
        """Define master index of all model elements (flows and abstractions)."""

        index_col = self.constants.arc_index_column

        n_swabs = self.swabs.data.shape[0]
        n_gwabs = self.gwabs.data.shape[0]
        n_flows = self.flows.data.shape[0]

        i0 = 0
        self.swabs.data[index_col] = np.arange(n_swabs, dtype=int)
        i0 += n_swabs
        self.gwabs.data[index_col] = np.arange(i0, i0 + n_gwabs, dtype=int)
        i0 += n_gwabs
        self.flows.data[index_col] = np.arange(i0, i0 + n_flows, dtype=int)

    def identify_subdomains(self):
        """
        Identify subdomains and their indices in the "master" index.

        Sets *subdomain_indices*  dictionary attribute with:

            - All abstractions in a subdomain.
            - Just those abstractions needed to evaluate equality (point-scale
              initially) - i.e. to avoid duplication of GWABS impacting multiple
              waterbodies.
            - List of outlet waterbodies used as identifiers for each subdomain.

        First two entries in list above are identified by subdomain (identified by its
        outlet waterbody ID).

        Note that only subdomains with non-zero demand are included here - i.e. any with
        zero demand are just ignored.

        """
        outlet_wbs = self.ds.find_outlet_waterbodies()
        outlet_wbs = [wb for wb in outlet_wbs if wb in self.ds.wbs.data.index]

        subdomain_wbs = []  # will contain outlet waterbody ID for each subdomain
        for wb in outlet_wbs:
            wbs = self.ds.identify_upstream_waterbodies(wb)

            # All abstractions per subdomain
            swabs = self.swabs.data.loc[
                self.swabs.data[self.swabs.waterbody_id_column].isin(wbs)
            ]
            gwabs = self.gwabs.data.loc[
                self.gwabs.data[self.gwabs.waterbody_id_column].isin(wbs)
            ]
            indices = swabs[self.swabs.arc_index_column].tolist()
            indices.extend(gwabs[self.gwabs.arc_index_column].tolist())

            if len(indices) > 0:
                subdomain_wbs.append(wb)
                self.subdomain_indices[('all-abstractions', wb)] = indices

                # Abstractions needed to evaluate equality
                gwabs_indexes = gwabs.groupby(
                    gwabs.index.get_level_values(0)
                )[self.gwabs.arc_index_column].min().to_numpy()
                if (
                        gwabs.loc[
                            (gwabs[self.gwabs.arc_index_column].isin(gwabs_indexes))
                            & (gwabs[self.gwabs.proportion_column] == 0.0)
                        ].shape[0] > 0
                ):
                    raise NotImplementedError(
                        'Not yet handled the case where the first-impacted waterbody '
                        'of a gwab has zero proportion of the total impact.'
                    )
                gwabs = gwabs.loc[gwabs[self.gwabs.arc_index_column].isin(gwabs_indexes)]

                indices = swabs[self.swabs.arc_index_column].tolist()
                indices.extend(gwabs[self.gwabs.arc_index_column].tolist())
                self.subdomain_indices[('unique-abstractions', wb)] = indices

        self.subdomain_indices['waterbody-list'] = subdomain_wbs

    def run(self) -> Dict:
        """
        Run sequence of data processing steps.

        Returns:
            Dictionary containing the following keys/values:
                - 'subset-dataset': Input Dataset restricted to relevant domain and
                  containing all impacts, not just those to be optimised.
                - 'flows-table': Formatted inflows (natural flows adjusted for static
                  impacts) as an instance of Flows table.
                - 'swabs-table': Formatted surface water abstraction impacts (demand)
                  as an instance of SWABS table.
                - 'gwabs-table': Formatted grounwater abstraction impacts (demand) as
                  an instance of SWABS table.
                - 'subdomain-dicts': Helper for calculating equality metrics per
                  subomain.

        """
        self.filter_supresgw()
        self.prepare_flows()
        self.filter_abstractions()
        self.prepare_gwabs()
        self.prepare_swabs()

        # Check again that something to optimise - possible for gwabs to pass through
        # filter_abstractions but then end up with no relevant impacts left after
        # waterbody splits considered in prepare_gwabs
        if (self.swabs.data.shape[0] == 0) and (self.gwabs.data.shape[0] == 0):
            raise ValueError('No abstraction impacts to optimise.')

        self.define_index()
        self.identify_subdomains()

        self.formatted_data = {
            'subset-dataset': self.ds,  # includes all swabs/gwabs, not just those to be optimised
            'flows-table': self.flows,
            'swabs-table': self.swabs,
            'gwabs-table': self.gwabs,
            'subdomain-dicts': self.subdomain_indices,
        }

        return self.formatted_data


class ArrayBuilder:
    """
    Take prepared data and build arrays to help form constraints/objectives in model.

    Args:
        flows: Instance of Flows table containing inflows (natural flows adjusted for
            static impacts).
        swabs: Instance of SWABS table containing surface water abstractions (demands).
        gwabs: Instance of SWABS table containing surface water abstractions (demands).
        graph: Directed graph of waterbody network.
        subdomain_indices: Helper for calculating equality metrics for subdomains.
        raise_external_hof_error: Whether to raise error if waterbody on which a HOF
            condition is set sits outside model domain. Default False (i.e. assume
            that HOF is respected - covered elsewhere in documentation.
        constants: By default instance of config.Constants.

    """
    def __init__(
            self, flows: Flows, swabs: SWABS, gwabs: GWABS, graph,
            subdomain_indices: Dict, raise_external_hof_error: bool,
            constants: Constants = None,
    ):
        self.flows = flows
        self.swabs = swabs
        self.gwabs = gwabs
        self.graph = graph
        self.subdomain_indices = subdomain_indices
        self.raise_external_hof_error = raise_external_hof_error

        if constants is None:
            constants = Constants()
        self.constants = constants

        # Arrays needed by model
        self.A_1 = None  # arc flow bounds (lhs) (2D) [flow targets and abstraction limits]
        self.A_2 = None  # mass balance (lhs) (2D)
        self.A_3 = None  # gwabs proportional splits (lhs) (2D)
        self.b_1 = None  # arc flow bounds (rhs) (1D) [flow targets and abstraction limits]
        self.b_2 = None  # mass balance (rhs) (1D)
        self.b_3 = None  # gwabs proportional splits (rhs) (1D)
        self.c = None  # cost (1D)
        self.m = None  # upper bounds of arc flows (1D)
        self.h = None  # flows defining hofs (1D)
        self.p = None  # indexes of arcs that are swabs with hofs (1D)
        self.q = None  # indexes of arcs defining hofs (1D)
        self.r = None  # domain-level unique abstractions (for equality) (1D)
        self.s = None  # all abstractions in (each) subdomain (2D)
        self.t = None  # subdomain-level unique abstractions (for equality) (2D)

        # Helper counts needed by model (for convenience)
        self.n_swabs = self.swabs.data.shape[0]
        self.n_gwabs = self.gwabs.data.shape[0]
        self.n_flows = self.flows.data.shape[0]
        self.n_abs = self.n_swabs + self.n_gwabs
        self.n_arcs = self.n_abs + self.n_flows
        self.n_subdomains = len(self.subdomain_indices['waterbody-list'])

        # Dictionaries returned by run method
        self.arrays = None
        self.counts = None

    def construct_arc_flow_bounds(self):
        """
        Prepare arrays used to define arc flow upper/lower bounds.

        A_1 is a square matrix with either +1 or -1 on diagonal (off-diagonal entries are
        zeros). This is used to modulate the sign of the arc flow constraints, which are
        upper bounds for abstraction arcs and lower bounds (i.e. targets) for waterbody
        outflow arcs. Bound values are in b_1, whose entries are positive for abstractions
        and negative for flow targets.

        """
        self.A_1 = sp.eye(self.n_arcs, format='csr')
        self.A_1.data[self.n_abs:] = -1

        self.b_1 = np.zeros(self.n_arcs)
        self.b_1[:self.n_swabs] = self.swabs.data[self.swabs.impact_column].to_numpy()
        self.b_1[self.n_swabs:self.n_abs] = self.gwabs.data[
            self.gwabs.impact_column
        ].to_numpy()
        self.b_1[-self.n_flows:] = -self.flows.data[self.flows.qt_column].to_numpy()

    def construct_mass_balance_arrays(self):
        """
        Construct arrays needed to apply mass balance constraints (A_2 and b_2).

        Using sign convention of positive for outflows (including abstractions) from
        waterbody and negative for inflows (arcs coming in from upstream).

        """
        arc_index_col = self.swabs.arc_index_column  # common across tables

        self.A_2 = np.zeros((self.n_flows, self.n_arcs))

        # Helper to find the index of a flow arc just relative to all flow arcs, rather
        # than all arcs (as df_flows may not be "ordered")
        flow_indexes = np.arange(self.n_flows)

        i = 0
        for waterbody, row in self.flows.data.iterrows():
            waterbody_col = self.swabs.waterbody_id_column
            swabs = self.swabs.data.loc[self.swabs.data[waterbody_col] == waterbody]
            for j in swabs[arc_index_col]:
                self.A_2[i, j] = 1

            gwabs = self.gwabs.data.loc[self.gwabs.data[waterbody_col] == waterbody]
            for j in gwabs[arc_index_col]:
                self.A_2[i, j] = 1

            # Outflow occurs at the arc_index of the waterbody itself
            self.A_2[i, int(row[arc_index_col])] = 1

            # Then need successor to know it has an inflow from an upstream waterbody
            successors = list(self.graph.successors(waterbody))
            if len(successors) > 0:
                next_waterbody = successors[0]
                k = flow_indexes[self.flows.data.index == next_waterbody][0]
                self.A_2[k, int(row[arc_index_col])] = -1

            i += 1

        self.A_2 = sp.csr_matrix(self.A_2)

        self.b_2 = self.flows.data[
            self.flows.get_qmod_column(self.constants.sub_abb)
        ].to_numpy()

    def construct_gwabs_proportion_arrays(self):
        """
        Construct arrays needed to apply gwabs proportions constraints (A_3 and b_3).

        Using the first impacted waterbody as a reference for defining ratios (factors).
        Then can check that, after multiplication by (signed) factors, sum is zero.

        """
        id_col, impact_num_col = self.gwabs.index_name

        # Subset on only gwabs that impact multiple waterbodies
        df0 = self.gwabs.data.reset_index()
        df0 = df0.groupby(df0[id_col])[impact_num_col].size().reset_index()
        ids = df0.loc[df0[impact_num_col] > 1, id_col].unique()
        gwabs = self.gwabs.data.loc[
            self.gwabs.data.index.get_level_values(id_col).isin(ids)
        ]

        # For a given gwab that impacts multiple waterbodies, the number of impacted
        # waterbodies minus one gives the number of constraints, as one component of impact
        # is used as reference to define ratios. From this we can figure out the total
        # number of constraints
        n_constraints = gwabs.shape[0] - len(ids)

        self.A_3 = np.zeros((n_constraints, self.n_arcs))

        proportion_col = self.gwabs.proportion_column

        j = 0  # constraint number incremented below
        unique_ids = gwabs.index.unique(0)
        for unique_id in unique_ids:
            tmp = gwabs.loc[gwabs.index.get_level_values(0) == unique_id]

            if tmp[proportion_col].iloc[0] == 0.0:
                raise NotImplementedError(
                    'Currently using the first waterbody impacted by a gwab as a '
                    'reference for forming proportional split ratios: failed due to '
                    'proportion for first impacted waterbody being zero for gwab '
                    f'{unique_id}'
                )

            # Start with second impacted waterbody for forming constraints, as ratios
            # (factors) are defined relative to first waterbody, which therefore does not
            # need a constraint here
            for i in range(1, tmp.shape[0]):

                # Handle case where a waterbody is "impacted" with zero proportion (seen in
                # WRGIS data)
                if tmp[proportion_col].iloc[i] > 0.0:
                    factor = (
                        1.0 / (tmp[proportion_col].iloc[i] / tmp[proportion_col].iloc[0])
                    )
                else:
                    factor = 0.0

                # Place factor into array - so waterbody i will come through as factor * x
                # and waterbody 0 will come through as -1 * x (and sum should be zero)
                k0 = tmp[self.constants.arc_index_column].iloc[i]
                k1 = tmp[self.constants.arc_index_column].iloc[0]
                self.A_3[j, k0] = factor
                self.A_3[j, k1] = -1.0

                j += 1

        self.b_3 = np.zeros(n_constraints)

        if self.A_3.shape[0] == 0:
            self.A_3 = np.zeros((1, self.n_arcs))
            self.b_3 = np.zeros(1)

        self.A_3 = sp.csr_matrix(self.A_3)

    def construct_costs(self):
        """
        Construct the costs vector.

        Initially costs are equal (-1) for abstraction arcs and zero for flow arcs, as
        model solves a minimisation problem. Flow targets are implemented as (hard)
        constraints so flows not part of objective currently.

        """
        self.c = np.zeros(self.n_arcs)
        self.c[:self.n_abs] = -1

    def construct_hof_arrays(self):
        """
        Build arrays needed to implement explicit HOF constraints.

        Constructs and sets the following attributes:

            - *h*: Flows conditions that define a HOF (for SWABS with HOFs).
            - *p*: Indexes of arcs with HOFs (i.e. indicating arcs that are SWABS with
              HOFs).
            - *q*: Indexes of arcs whose flows determine whether a HOF is being
              respected.

        So comparing the conditions in *h* with the flows in arcs with indexes *q*
        helps the model ensure HOFs are respected.

        """
        swabs = self.swabs.data.loc[
            (self.swabs.data[self.swabs.hof_value_column] > 0.0)
            & (self.swabs.data[self.swabs.impact_column] > 0.0)
        ]

        if swabs.shape[0] > 0:
            df = swabs.merge(
                self.flows.data[[self.constants.arc_index_column]].rename(
                    columns={self.constants.arc_index_column: 'flow_arc_index'}
                ),
                how='left', left_on=self.swabs.hof_waterbody_column, right_index=True,
            )

            required_waterbodies = set(
                swabs[self.swabs.hof_waterbody_column].unique().tolist()
            )
            available_waterbodies = set(self.flows.data.index.unique().tolist())
            if not required_waterbodies.issubset(available_waterbodies):
                unavailable_waterbodies = list(
                    set(required_waterbodies) - set(available_waterbodies)
                )
                if self.raise_external_hof_error:
                    raise ValueError(
                        f'A waterbody(s) that defines a HOF is outside of the domain '
                        f'being modelled. Set raise_external_hof_error argument to '
                        f'False to avoid this error (i.e. assuming that HOF condition '
                        f'continues to be met, as it must have been in the input data).'
                        f'\nThe waterbody(s) outside the domain are: {unavailable_waterbodies}'
                    )
                else:
                    df = df.loc[
                        ~df[self.swabs.hof_waterbody_column].isin(unavailable_waterbodies)
                    ]

            # Indicator for whether an arc is a swab with a hof
            # - 1 for arcs that are swabs with hofs, 0 elsewhere
            # - note this is the opposite definition to p used in earlier code
            self.p = np.zeros(self.n_arcs, dtype=int)
            self.p[df[self.constants.arc_index_column]] = 1

            # Indices of waterbody outflow arcs that are used for hof conditions
            # - one element per swab with a hof
            # - value indicates index of z to look in to check flow vs hof condition
            self.q = df['flow_arc_index'].to_numpy(dtype=int)

            # HOF flow thresholds
            # - one element per swab with a hof
            self.h = df[self.swabs.hof_value_column].to_numpy(dtype=int)

        else:
            self.p = np.array([], dtype=int)
            self.q = np.array([], dtype=int)
            self.h = np.array([])

    def construct_arc_upper_bounds(self):
        """
        Find upper bounds of arc flows.

        This is the target abstraction for abstraction arcs and the (modified) natural
        "ups" flow for waterbody outflow arcs (i.e. modified for discharges, complex
        impacts and any gwabs/swabs held constant.)

        """
        self.m = np.zeros(self.n_arcs)

        i = 0
        j = self.n_swabs
        self.m[i:j] = self.swabs.data[self.swabs.impact_column].to_numpy()

        i = j
        j = self.n_swabs + self.n_gwabs
        self.m[i:j] = self.gwabs.data[self.gwabs.impact_column].to_numpy()

        self.m[j:] = self.flows.data[
            self.flows.get_qmod_column(self.constants.ups_abb)
        ].to_numpy()  # ups flow

    def find_indexes_of_unique_abstractions(self):
        """
        Find indexes of unique abstraction arcs to include in equality metric.

        This is initially to make sure that we avoid double-counting gwabs that impact
        multiple waterbodies (i.e. presumption that each licence-point-purpose level
        impact should be represented once in determining an equality metric). This
        works because the gwabs will be kept in proportion to each other, so the
        proportion fulfilled will be the same for each component impact.

        The method sets the attribute *r*, which is used to support domain-wide
        evaluation of proportional equality. Additional arrays (*s* and *t*) are now
        used to help evaluate subdomain-level proportional equality.

        Currently depends on self.m being available (set in construct_arc_upper_bounds).
        This is for filtering out abstractions whose rate is zero (assuming *r* is
        only used for equality calcs...).

        """
        gwabs_indexes = self.gwabs.data.groupby(
            self.gwabs.data.index.get_level_values(0)
        )[self.constants.arc_index_column].min().to_numpy()
        if (
                self.gwabs.data.loc[
                    (self.gwabs.data[self.constants.arc_index_column].isin(gwabs_indexes))
                    & (self.gwabs.data[self.gwabs.proportion_column] == 0.0)
                ].shape[0] > 0
        ):
            raise NotImplementedError(
                'Not yet handled the case where the first-impacted waterbody of a gwab '
                'has zero proportion of the total impact.'
            )
        self.r = np.concatenate([
            self.swabs.data[self.constants.arc_index_column].to_numpy(), gwabs_indexes
        ])

    def construct_abstraction_subdomain_array(self):
        """
        Build array to indicate abstraction elements impacting each subdomain.

        """
        self.s = np.zeros((self.n_subdomains, self.n_arcs), dtype=int)
        i = 0
        for wb in self.subdomain_indices['waterbody-list']:
            indices = self.subdomain_indices[('all-abstractions', wb)]
            self.s[i, indices] = 1
            i += 1

    def construct_unique_abstraction_subdomain_array(self):
        """
        Build array to indicate "unique" abstraction elements impacting each subdomain.

        Unique here refers to just those abstraction impacts to be used in equality
        calculations (i.e. avoiding double-counting gwabs that impact muliple
        waterbodies).

        """
        self.t = np.zeros((self.n_subdomains, self.n_arcs), dtype=int)
        i = 0
        for wb in self.subdomain_indices['waterbody-list']:
            indices = self.subdomain_indices[('unique-abstractions', wb)]
            self.t[i, indices] = 1
            i += 1

    def run(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """
        Run processing sequence to construct arrays needed for model.

        Returns:
            Tuple of dictionaries containing:
                - Arrays used to help construct constraints/objective(s) in model. Keys
                  are the array names used by this class and Model class: A_1, A_2, A_3,
                  b_1, b_2, b_3, c, m, h, p, q, r, s, t. Values are arrays.
                - Counts of model elements used to help in problem formulation (just
                  convenience things). Keys are count labels: n_swabs, n_gwabs, n_flows,
                  n_abs, n_arcs. Values are integer counts.

        """
        self.construct_arc_flow_bounds()
        self.construct_mass_balance_arrays()
        self.construct_gwabs_proportion_arrays()
        self.construct_costs()
        self.construct_hof_arrays()
        self.construct_arc_upper_bounds()
        self.find_indexes_of_unique_abstractions()
        self.construct_unique_abstraction_subdomain_array()
        self.construct_abstraction_subdomain_array()

        self.arrays = {
            'A_1': self.A_1, 'A_2': self.A_2, 'A_3': self.A_3,
            'b_1': self.b_1, 'b_2': self.b_2, 'b_3': self.b_3,
            'c': self.c, 'm': self.m, 'h': self.h, 'p': self.p, 'q': self.q,
            'r': self.r, 's': self.s, 't': self.t,
        }

        self.counts = {
            'n_swabs': self.n_swabs, 'n_gwabs': self.n_gwabs, 'n_flows': self.n_flows,
            'n_abs': self.n_abs, 'n_arcs': self.n_arcs,
        }

        return self.arrays, self.counts


@dataclass
class Metrics:
    """
    Model performance metrics.

    In comments below, using the following terminology:

        - "MAD" (or "mad") - mean absolute deviation.
        - "Point" - abstraction point/element.
        - "Proportion fulfilled" - proportion of target abstraction impact (demand).
          fulfilled.

    """
    domain_total_abstraction: float = None

    # Proportion of total domain target abstraction fulfilled in solution
    domain_proportion_fulfilled: float = None

    # MAD of each point's proportion fulfilled vs domain proportion fulfilled
    domain_point_mad: float = None

    # Total abstraction per subdomain
    subdomain_total_abstractions: np.ndarray = None

    # Proportion of total subdomain target abstraction fulfilled in solution (i.e. per
    # subdomain)
    subdomain_proportions_fulfilled: np.ndarray = None

    # MAD of each subdomain's proportion fulfilled vs domain proportion fulfilled
    subdomain_mad: float = None

    # For each subdomain, MAD of each point's proportion fulfilled vs subdomain
    # proportion fulfilled
    subdomain_point_mad: np.ndarray = None

    # Number of points changed in overall domain
    domain_n_changes: int = None


@dataclass
class AuxiliaryInfo:
    """
    Subset of model metrics and results needed to support sequential objectives.

    """
    domain_total_abstraction: float = None
    subdomain_proportions_fulfilled: np.ndarray = None
    y: np.ndarray = None  # binary variable (for HOFs)
    z: np.ndarray = None  # "final" solution


class Model:
    """
    Model to formulate and solve one optimisation problem.

    One problem is defined as a combination of: domain + scenario + percentile +
    objective.

    Currently available objectives are:

        - Maximise total abstraction impact (equivalent to minimising total abstraction
          impact reductions).
        - Maximise equality of proportional fulfillment of target abstractions
          (~demands). Equivalent to maximising equality of proportional reductions.

    If a secondary objective is being used (typically maximum equality of proportional
    fulfillment/reductions), a primary objective will have already been determined. This
    can become a constraint on the solution for the secondary objective (optionally with
    some relaxation applied). I.e. it nudges the model towards finding the most equal
    solution given that total abstraction impact should be some (now) known amount. The
    special_constraints argument indicates whether a primary objective should be used as
    a constraint for the given objective_type.

    Args:
        objective_type: Either 'max-abstraction' or 'max-point-equality'. See above.
        special_constraints: If objective_type is 'max-point-equality', this argument
            will typically be 'max-abstraction' to indicate that a previously run
            objective should now become a constraint on the solution.
        auxiliary_info: Subset of model metrics and results needed to support sequential
            objectives.
        arrays: Output from ArrayBuilder.run.
        counts: Output from ArrayBuilder.run.
        solver: Name used by cvxpy to indicate the solver that it should use.

    """
    def __init__(
            self, objective_type: str, special_constraints: List[str],
            auxiliary_info: AuxiliaryInfo, arrays: Dict, counts: Dict,
            solver: str = cp.SCIPY,
    ):
        self.objective_type = objective_type
        self.special_constraints = special_constraints
        self.aux = auxiliary_info
        self.solver = solver

        # Arrays needed by model
        self.A_1 = arrays['A_1']  # arc flow bounds (lhs) (2D) [flow targets and abstraction limits]
        self.A_2 = arrays['A_2']  # mass balance (lhs) (2D)
        self.A_3 = arrays['A_3']  # gwabs proportional splits (lhs) (2D)
        self.b_1 = arrays['b_1']  # arc flow bounds (rhs) (1D) [flow targets and abstraction limits]
        self.b_2 = arrays['b_2']  # mass balance (rhs) (1D)
        self.b_3 = arrays['b_3']  # gwabs proportional splits (rhs) (1D)
        self.c = arrays['c']  # cost (1D)
        self.m = arrays['m']  # upper bounds of arc flows (1D)
        self.h = arrays['h']  # flows defining hofs (1D)
        self.p = arrays['p']  # indexes of arcs that are swabs with hofs (1D)
        self.q = arrays['q']  # indexes of arcs defining hofs (1D)
        self.r = arrays['r']  # domain-level unique abstractions (for equality) (1D)
        self.s = arrays['s']  # all abstractions in (each) subdomain (2D)
        self.t = arrays['t']  # subdomain-level unique abstractions (for equality) (2D)

        # Helper counts
        self.n_swabs = counts['n_swabs']
        self.n_gwabs = counts['n_gwabs']
        self.n_flows = counts['n_flows']
        self.n_abs = counts['n_abs']
        self.n_arcs = counts['n_arcs']

        # cvxpy "variables"
        self.w = cp.Variable(self.n_abs)  # helper for vectorised equality objective
        self.y = cp.Variable(self.n_arcs, boolean=True)  # for hofs
        self.z = cp.Variable(self.n_arcs)  # "final" decision vector (all flows, abstractions)

        # Helpers for vectorised equality objective
        self.T = None
        self.Ti = None
        self.p_per_node = None
        self.inv_m = None
        self.z_norm = None
        self.inv_row_counts = None
        self.sum_w_per_sub = None
        self.mean_w_per_sub = None

        self.objective = None  # becomes a cvxpy expression
        self.constraints = None  # becomes a list of cvxpy-style constraints
        self.problem = None  # becomes a cvxpy problem object

        self.metrics = Metrics()

        # ---
        epsilon = 1e-2
        self.v = 1.0 / (np.abs(self.m[:self.n_abs]) + epsilon)
        # ---

    def initialise_constraints(self):
        """
        Construct "core" constraints of problem.

        These constraints are largely shared regardless of the objective_type. However,
        we do currently take the binary variable y (indicating whether a given SWAB can
        be "on" if it has a HOF condition") as an input if solving for max-point-equality.
        Testing suggested that this objective can be harder to obtain a solution to, so
        for now we simplify by removing the mixed integer (binary) part of the problem
        when solving for the secondary (max-point-equality) objective only.

        """
        # Core constraints
        self.constraints = [
            self.z >= 0,  # no negative arc flows
            self.A_1 @ self.z <= self.b_1,  # flow bounds (flow targets and abstraction limits)
            self.A_2 @ self.z == self.b_2,  # mass balance
            self.A_3 @ self.z == self.b_3,  # gwabs proportional splits
        ]

        if self.objective_type == 'max-abstraction':
            # In general limit flows/abstraction to their maximum possible rates, which
            # are known upfront for a case of only reducing abstraction impacts. (They
            # are not necessarily known if changing discharges...) Use y to enforce HOF
            # condition - i.e. if y is zero it limits z to zero in the respective SWABS
            # arc.
            self.constraints.append(self.z <= cp.multiply(self.m, self.y))

            # Further HOF constraints
            if np.sum(self.m[self.p == 1]) > 0.0:
                self.constraints.extend([
                    # y should be one for elements without HOFs (i.e. no decision)
                    self.y[self.p == 0] == 1,

                    # Ensure HOF is respected if swab is "on" (i.e. y == 1)
                    self.z[self.q] - cp.multiply(self.y[self.p == 1], self.h) >= 0,
                ])

        elif self.objective_type == 'max-point-equality':
            # Technically no need to place the same upper bound limit on arc flows if
            # not modelling HOFs explicitly under this objective (i.e. taking y as
            # known from solving max-abstraction objective before trying to solve for
            # the point equality objective.

            # HOF constraints
            # - now y is known so any swabs that should be off can be switched off
            # - modify hof thresholds for this set of swabs so that they do not form a
            #   constraint on the solution
            # - where swabs with hofs are still on, we know that the relevant waterbody
            #   outflows should be >= the hof thresholds now - we still need to enforce
            #   this in the solution (hence the final constraint below)
            if np.sum(self.m[self.p == 1]) > 0.0:
                if np.sum(self.aux.y == 0) > 0:
                    self.constraints.append(self.z[self.aux.y == 0] == 0)
                    self.h[self.aux.y[self.p == 1] == 0] = 0
                self.constraints.append(self.z[self.q] - self.h >= 0)

        # ---
        elif self.objective_type == 'min-n-changes':
            if np.sum(self.m[self.p == 1]) > 0.0:
                if np.sum(self.aux.y == 0) > 0:
                    self.constraints.append(self.z[self.aux.y == 0] == 0)
                    self.h[self.aux.y[self.p == 1] == 0] = 0
                self.constraints.append(self.z[self.q] - self.h >= 0)
        # ---

        else:
            raise ValueError(f'Unknown objective type: {self.objective_type}')

    def augment_constraints(self):
        """
        Add in any "special" constraints (see class doctring).

        """
        if 'max-abstraction' in self.special_constraints:
            self.constraints.append(
                cp.sum(self.z[:self.n_abs]) == self.aux.domain_total_abstraction
            )
        if 'max-point-equality' in self.special_constraints:
            # Under the current sequential two-objective setup, only the first objective
            # could become a constraint for the second objective. It may not make sense
            # for max-point-equality to be the first objective and so potentially become
            # a constraint on a second objective. Refer to the formulation for the
            # objective if this changes though.
            raise NotImplementedError

    def set_objective(self):
        """
        Define expression that should become the cvxpy objective.

        The 'max-point-equality' objective is evaluated per subdomain, i.e. for the
        "catchment" upstream of and including an outlet waterbody. If the overall domain
        only has one outlet this is equivalent to a domain-wide evaluation. If it has
        multiple outlets then the basic metric (mean absolute deviation from a target
        proportion) is evaluated per subdomain and then aggregated to form the final
        objective. Note that the target proportion varies per subdomain.

        """
        if self.objective_type == 'max-abstraction':
            self.objective = cp.Minimize(self.c.T @ self.z)

        elif self.objective_type == 'max-point-equality':
            self.T = sp.csr_matrix(self.t[:, :self.n_abs])
            self.Ti = self.T.indices
            g = np.asarray(self.t[:, :self.n_abs].argmax(axis=0)).ravel()
            self.p_per_node = self.aux.subdomain_proportions_fulfilled[g]

            self.inv_m = 1.0 / self.m[:self.n_abs]
            self.z_norm = cp.multiply(self.inv_m, self.z[:self.n_abs])

            self.constraints += [self.w >= self.z_norm - self.p_per_node]
            self.constraints += [self.w >= -(self.z_norm - self.p_per_node)]
            self.constraints += [self.w >= 0, self.w <= 1]

            row_counts = np.asarray(self.T.sum(axis=1)).ravel()
            self.inv_row_counts = 1.0 / row_counts

            self.sum_w_per_sub = self.T @ self.w
            self.mean_w_per_sub = cp.multiply(self.inv_row_counts, self.sum_w_per_sub)

            self.objective = cp.Minimize(cp.sum(self.mean_w_per_sub) / self.T.shape[0])

        # ---
        elif self.objective_type == 'min-n-changes':
            self.objective = cp.Minimize(
                cp.sum(
                    cp.multiply(
                        self.v, cp.abs(self.z[:self.n_abs] - self.m[:self.n_abs])
                    )
                )
            )
        # ---

        else:
            raise ValueError(f'Unknown objective type: {self.objective_type}')

    def set_problem(self):
        """Construct cvxpy problem object (but do not solve yet)."""
        self.initialise_constraints()
        self.augment_constraints()
        self.set_objective()
        self.problem = cp.Problem(
            objective=self.objective, constraints=self.constraints,
        )

    def run(self):
        """Run solver to find solution to problem and set metrics attribute."""
        self.problem.solve(solver=self.solver)
        if self.z.value is not None:
            self.set_metrics()

    def set_metrics(self):
        """Evaluate metrics and assign to metrics (dataclass) attribute."""
        self.metrics.domain_total_abstraction = np.sum(self.z.value[:self.n_abs])
        self.metrics.domain_proportion_fulfilled = (
                self.metrics.domain_total_abstraction / np.sum(self.m[:self.n_abs])
        )
        self.metrics.domain_point_mad = np.mean(np.abs(
            self.z.value[self.r] / self.m[self.r]
            - self.metrics.domain_proportion_fulfilled
        ))
        self.metrics.subdomain_total_abstractions = self.s @ self.z.value
        self.metrics.subdomain_proportions_fulfilled = (
            self.metrics.subdomain_total_abstractions / (self.s @ self.m)
        )
        self.metrics.subdomain_mad = np.mean(np.abs(
            self.metrics.subdomain_proportions_fulfilled
            - self.metrics.domain_proportion_fulfilled
        ))

        z = self.z.value[:self.n_abs]
        y = self.m[:self.n_abs]
        p = self.metrics.subdomain_proportions_fulfilled
        t = self.t[:, :self.n_abs]
        self.metrics.subdomain_point_mad = (
            np.abs((z / y - p[:, None]) * t).sum(axis=1) / t.sum(axis=1)
        )

        # Number of lpp-level (point) changes (i.e. if a gwab impacts multiple
        # waterbodies only count it as one change)
        self.metrics.domain_n_changes = np.sum(
            np.abs(self.z.value[self.r] - self.m[self.r]) > 1e-9
        )
