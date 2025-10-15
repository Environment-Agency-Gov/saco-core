"""
Calculation and classification of flows for prescribed artificial influences scenarios.

Notes:
    - Use LPP (or lpp) or "point" throughout to stand for licence-point-purpose (and
      the equivalent level of (point) granularity for discharges and complex impacts).
    - Following WRGIS use of sub and ups terminology. sub refers to just the current
      waterbody (e.g. for an abstraction impact or naturalised flow), whereas ups
      refers to sums for the current waterbody and all waterbodies upstream of it (e.g.
      total abstraction impact on flows at the outlet of the current water body).

"""
import copy
import itertools
from typing import List, Dict

import numpy as np
import pandas as pd
import networkx as nx

from .tables import DataTable
from .dataset import Dataset, subset_dataset_on_wbs
from .config import Constants


class Calculator:
    """
    For calculating scenario flows and assessing compliance against targets.

    In most use cases, the steps needed to use the Calculator are (code example below):

        - Prepare/load an input ``Dataset``
        - Create an instance of this (the ``Calculator``) class
        - Execute the calculations using the ``run`` method.

    The ``run`` method produces an updated "Master" dataframe, which can be returned
    on its own or as part of a full ``Dataset`` (the latter option being the current
    default).

    Args:
        input_dataset: Instance of Dataset for which to run calculations.
        scenarios: Names/abbreviations of artificial influences scenarios for which
            calculations should be performed. If None (default) then taken from
            input_dataset.
        percentiles: Flow percentiles (natural) for which calculations should be
            performed. If None (default) then taken from input_dataset.
        domain: List of waterbody IDs indicating domain/catchment to be optimised. If
            None (default) then all waterbodies in input_dataset will be included in
            the domain.
        capping_method: Indicates method to use to deal with potential for "negative
            flows" if ups impacts exceed ups natural flow in a waterbody. Options are
            'cap-net-impacts', 'simple' or None. See notes below.
        unassessed_waterbodies: Optional list of waterbodies for which compliance
            should not be assessed. If None (default) then any waterbodies of types
            'Seaward Transitional' or 'Saline Lagoon' will not be assessed.
        check_order: Confirm that order of dataframe(s) matches order of nodes in
            *input_dataset.graph* before carrying out calculations (small overhead).
        bin_edges: Bin edges defining compliance band boundaries. By default, bin edges
            are taken from config.Constants (see :doc:`calculator`).
        na_value: Value to use for compliance in unassessed waterbodies.
        tolerance_dp: Number of decimal places of tolerance to use in rounding deficit
            divided by natural flow in compliance assessment (i.e. to avoid unfairly
            classing a very small negative difference as non-compliant).
        constants: Instance of config.Constants (default) or similar.

    Notes:
        A starting point for the input_dataset might be a Dataset comprising the WRGIS
        tables. Make any desired modifications to this (or some other) "base" Dataset
        before creating an instance of ``Calculator``. See :doc:`reference-dataset`
        documentation and examples.

        As discussed in :doc:`calculator`, there is a potential for "negative flows" if
        ups impacts exceed ups natural flow in a waterbody. This is because abstraction
        impacts are large under some scenarios and because the calculator works as a
        set of elementary arithmetic/array operations, rather than as a network
        flow-type model. A few options are provided to deal with this issue:

            - WRGIS-like approach (capping_method='cap-net-impacts'): loop through
              topological generations of waterbodies (upstream to downstream) and adjust
              net impacts on flows until no negatives. This is the default approach.
            - WBAT-like approach (capping_method='simple'): set potential negative flows to
              zero. This approach can be slightly more conservative (resulting in lower
              flows) than the WRGIS-like approach. It allows for potential propagation of
              some impacts downstream even if they have been implicitly "capped" in a
              waterbody if its flow has been increased to zero.
            - Do nothing (capping_method=None): allow negative scenario flows in output.

    Examples:
        >>> from saco import Dataset, Calculator
        >>>
        >>> ds = Dataset(data_folder='/path/to/data/files')
        >>> ds.load_data()
        >>>
        >>> calculator = Calculator(ds)
        >>> output_dataset = calculator.run()

    """
    def __init__(
            self,
            input_dataset: Dataset,
            scenarios: List[str] = None,
            percentiles: List[int] = None,
            domain: List[str] = None,
            capping_method: str | None = 'cap-net-impacts',
            unassessed_waterbodies: List[str] = None,
            check_order: bool = True,
            bin_edges: List[float] = None,
            na_value: int = -999,
            tolerance_dp: int = 3,
            constants: Constants = None,
    ):
        if domain is None:
            self.ds = copy.deepcopy(input_dataset)
        else:
            self.ds = subset_dataset_on_wbs(input_dataset, domain)

        if unassessed_waterbodies is None:
            unassessed_waterbodies = self.identify_unassessed_waterbodies()
        self.unassessed_waterbodies = unassessed_waterbodies

        if scenarios is None:
            scenarios = self.ds.scenarios
        if percentiles is None:
            percentiles = self.ds.percentiles
        self.scenarios = scenarios
        self.percentiles = percentiles

        self.capping_method = capping_method
        self.check_order = check_order

        self.na_value = na_value
        self.tolerance_dp = tolerance_dp

        if constants is None:
            self.constants = Constants()

        if bin_edges is None:
            bin_edges = self.constants.compliance_bin_edges
        self.bin_edges = bin_edges

    def run(self, master_only: bool = False) -> Dataset | pd.DataFrame:
        """
        Run data preparation and all calculations.

        Args:
            master_only: If True return the calculated "Master" table only as a
                dataframe. If False (default) return a full Dataset that includes an
                updated Master table.

        Returns:
            Either a full Dataset (default) with an updated "Master" table or a dataframe
            of just the updated master table.

        """
        self.prepare_data()
        self.calculate_scenario_flow()
        self.calculate_deficit()
        self.assess_compliance()
        self.calculate_target_deficit()

        if master_only:
            return self.ds.mt.data.copy()
        else:
            return self.ds

    def identify_unassessed_waterbodies(self):
        """Identify waterbodies that are not typically assessed due to their type."""
        unassessed_waterbodies = self.ds.wbs.data.loc[
            self.ds.wbs.data[self.ds.wbs.waterbody_type_column].isin(
                self.ds.wbs.unassessed_types
            )
        ].index.tolist()
        return unassessed_waterbodies

    def prepare_data(self):
        """
        Initialise master data table.

        The first step is to aggregate point-scale artificial influences to
        waterbody-scale (sub and ups). Then we merge in natural flows, EFI and any
        available flow target columns.

        """
        dfs = []
        for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
            ai_sub = self.aggregate_lpp_to_sub(scenario, percentile)

            # AI ups
            new_column_names = {}
            for variable_abb in self.ds.ai_variable_abbs:
                sub_name = self.ds.mt.get_value_column(
                    variable_abb, scenario, percentile, self.constants.sub_abb,
                )
                ups_name = self.ds.mt.get_value_column(
                    variable_abb, scenario, percentile, self.constants.ups_abb,
                )
                new_column_names[sub_name] = ups_name
            ai_ups = self.propagate_quantities(ai_sub, ai_sub.columns, new_column_names)

            # Initial master dataframe - qnat + AI columns
            df = pd.concat([
                self.ds.qnat.data[
                    self.ds.qnat.get_value_column(percentile, self.constants.sub_abb)
                ],
                self.ds.qnat.data[
                    self.ds.qnat.get_value_column(percentile, self.constants.ups_abb)
                ],
                ai_sub,
                ai_ups,
            ], axis=1)
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # Natural flows and discharges may be duplicated, as they do not have both
        # scenario or percentile as factors
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Incorporate EFI and flow targets (the latter only if they have been set
        # previously on the input dataset [its master table])
        df = pd.merge(df, self.ds.efi.data, left_index=True, right_index=True)
        if self.ds.mt.data is not None:
            qt_cols = []
            for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
                qt_col = self.ds.mt.get_qt_column(scenario, percentile)
                qt_cols.append(qt_col)
            qt_cols = [col for col in qt_cols if col in self.ds.mt.data.columns]
            df = pd.merge(df, self.ds.mt.data[qt_cols], left_index=True, right_index=True)

        self.ds.mt.set_data(df, validate=False)

    def aggregate_lpp_to_sub(self, scenario: str, percentile: int) -> pd.DataFrame:
        """
        Aggregate (all) point artificial influences to waterbody sub level.

        Complex purposes filtered down in _aggregate_lpp method (following WBAT).

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
            Dataframe of aggregated (sub) artificial influences compatible with
            master table (i.e. following master column naming conventions).

        """
        dfs = []
        for table in self.ds.ai_tables:
            if table.variable_abb == self.constants.dis_abb:
                value_column = table.get_value_column(scenario)
            else:
                value_column = table.get_value_column(scenario, percentile)

            new_column_name = self.ds.mt.get_value_column(
                table.variable_abb, scenario, percentile, self.constants.sub_abb,
            )

            if table.variable_abb == self.constants.gwabs_abb:
                df = self._aggregate_lpp_gwabs(value_column, new_column_name)
            else:
                df = self._aggregate_lpp(table, value_column, new_column_name)

            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        return df

    def propagate_quantities(
            self, df: pd.DataFrame, value_columns: List[str] | str,
            new_column_names: Dict[str, str] = None, replace_columns: bool = True,
    ):
        """
        Propagate quantities (flows or impacts) through the waterbody network.

        This method is for sub to ups conversions.

        Args:
            df: Input dataframe.
            value_columns: Value columns that should be propagated (sub numbers).
            new_column_names: Mapping of value column names in df to those required in
                the output dataframe (i.e. output column names might indicate ups
                aggregation). Only used is replace_columns is True.
            replace_columns: Whether to use new_column_names.

        Returns:
            Dataframe of value columns with ups aggregation.

        """
        if isinstance(value_columns, str):
            value_columns = [value_columns]

        if new_column_names is None:
            new_column_names = {c: c for c in value_columns}

        if self.check_order:
            if not np.all(df.index == np.array(self.ds.graph.nodes)):
                if list(set(df.index.tolist())) == list(set(self.ds.graph.nodes)):
                    df = pd.merge(
                        df, pd.DataFrame({df.index.name: self.ds.graph.nodes}),
                        how='right', left_index=True, right_on=df.index.name,
                    )
                else:
                    raise ValueError(
                        'Mismatch between dataframe index and graph nodes: they must '
                        'match.'
                    )

        df1 = df.copy()
        if replace_columns:
            df1 = df1.rename(columns=new_column_names)

        for value_column in value_columns:
            new_column_name = new_column_names[value_column]
            y = propagate_quantity(self.ds.routing_matrix, df[value_column])
            df1.loc[:, new_column_name] = y

        return df1

    def calculate_scenario_flow(self):
        """
        Calculate scenario flow (updating master table).

        First we derive the ups scenario flow (no capping). Then we cap using specified
        capping_method (if relevant). Then we derive sub to be consistent with ups.

        """
        for scenario, percentile in itertools.product(
                self.scenarios, self.percentiles,
        ):
            qnat_col = self.ds.mt.get_qnat_column(percentile, self.constants.ups_abb)
            dis_col = self.ds.mt.get_dis_column(scenario, self.constants.ups_abb)
            sup_col = self.ds.mt.get_sup_column(
                scenario, percentile, self.constants.ups_abb
            )
            swabs_col = self.ds.mt.get_swabs_column(
                scenario, percentile, self.constants.ups_abb
            )
            gwabs_col = self.ds.mt.get_gwabs_column(
                scenario, percentile, self.constants.ups_abb
            )
            scen_col = self.ds.mt.get_scen_column(
                scenario, percentile, self.constants.ups_abb
            )

            self.ds.mt.data[scen_col] = (
                    self.ds.mt.data[qnat_col] + self.ds.mt.data[dis_col]
                    + self.ds.mt.data[sup_col] - self.ds.mt.data[swabs_col]
                    - self.ds.mt.data[gwabs_col]
            )

            if self.capping_method == 'simple':  # following WBAT
                scen_col = self.ds.mt.get_scen_column(
                    scenario, percentile, self.constants.ups_abb,
                )
                self.ds.mt.data[scen_col] = np.maximum(
                    self.ds.mt.data[scen_col], 0.0
                )
            elif self.capping_method == 'cap-net-impacts':
                self._cap_impacts(
                    qnat_col, dis_col, sup_col, swabs_col, gwabs_col, scen_col,
                )
            else:
                pass  # if capping_method is None

            # Convert ups to sub
            scen_col_sub = self.ds.mt.get_scen_column(
                scenario, percentile, self.constants.sub_abb
            )
            self.ds.mt.data[scen_col_sub] = unpropagate_quantity(
                self.ds.adjacency_matrix, self.ds.mt.data[scen_col].to_numpy()
            )

    def calculate_deficit(self):
        """
        Calculate deficit relative to EFI (updating master table).

        Note that EFI might not be flow target in Dataset.

        """
        for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
            scen_col = self.ds.mt.get_scen_column(scenario, percentile, self.constants.ups_abb)
            efi_col = self.ds.mt.get_efi_column(percentile)
            sd_col = self.ds.mt.get_sd_column(scenario, percentile)
            self.ds.mt.data[sd_col] = self.ds.mt.data[scen_col] - self.ds.mt.data[efi_col]

    def assess_compliance(self):
        """Assess compliance band (updating master table."""
        for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
            nat_col = self.ds.mt.get_qnat_column(percentile, self.constants.ups_abb)
            sd_col = self.ds.mt.get_sd_column(scenario, percentile)
            comp_col = self.ds.mt.get_comp_column(scenario, percentile)

            # Find compliance bands
            # - rounding the deficit/natural term helps to avoid issues due to float
            #   precision (specifically avoiding cases where a waterbody receives a
            #   lower classification than it "should")
            # - np.digitize returns indices of bin_edges that need to be converted to
            #   classes that make sense (0 = compliant, 1 = band-1, ...)
            f = np.round(
                self.ds.mt.data[sd_col] / self.ds.mt.data[nat_col], self.tolerance_dp
            )
            bin_idx = np.digitize(f, np.array(self.bin_edges), right=False)
            self.ds.mt.data[comp_col] = -1 * (bin_idx - 4)
            self.ds.mt.data[comp_col] = np.where(
                self.ds.mt.data[nat_col] == 0.0, 0, self.ds.mt.data[comp_col]
            )

            # Assign NA value for waterbody types that are not assessed
            if self.unassessed_waterbodies is not None:
                self.ds.mt.data.loc[
                    self.ds.mt.data.index.isin(self.unassessed_waterbodies), comp_col
                ] = self.na_value

    def calculate_target_deficit(self):
        """Calculate deficit relative to flow target (which may not be EFI)."""
        for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
            scen_col = self.ds.mt.get_scen_column(
                scenario, percentile, self.constants.ups_abb
            )
            qt_col = self.ds.mt.get_qt_column(scenario, percentile)
            sdt_col = self.ds.mt.get_sdt_column(scenario, percentile)

            if qt_col in self.ds.mt.data.columns:
                self.ds.mt.data[sdt_col] = (
                    self.ds.mt.data[scen_col] - self.ds.mt.data[qt_col]
                )

    def _cap_impacts(
            self, qnat_col: str, dis_col: str, sup_col: str, swabs_col: str,
            gwabs_col: str, scen_col: str,
    ):
        """
        Cap net impacts using WRGIS-like approach (uses and updates master table).

        Approach is to loop through "generations" of waterbodies (effectively
        waterbodies of the same stream network order) and identify the adjustment
        needed to keep ups flows >= zero (after propagation through network).

        Args:
            qnat_col: Natural flow column name.
            dis_col: Discharge column name.
            sup_col: Complex impacts column name.
            swabs_col: Surface water abstractions column name.
            gwabs_col: Groundwater abstractions column name.
            scen_col: Scenario flow column name.

        """
        df = self.ds.mt.data

        # Impact adjustment (sub)column will contain amount that needs to be subtracted
        # from net impacts to get plausible flows (once propagated to ups)
        net_impact_col = '__NET_IMPACT_TEMP'
        impact_adj_sub_col = '__IMPACT_ADJ_SUB_TEMP'
        impact_adj_ups_col = '__IMPACT_ADJ_UPS_TEMP'
        df[net_impact_col] = df[dis_col] + df[sup_col] - df[swabs_col] - df[gwabs_col]
        df[impact_adj_sub_col] = 0.0

        df[scen_col] = df[qnat_col] + df[net_impact_col]  # ok to be negative at this point

        # Generations gives list of waterbodies of same "order" (in upstream to
        # downstream loop)
        capping_applied = False
        for waterbodies_subset in nx.topological_generations(self.ds.graph):
            if np.all(df[scen_col] >= 0.0):
                break
            else:
                capping_applied = True
                df[impact_adj_sub_col] = np.where(
                    (df.index.isin(waterbodies_subset)) & (df[scen_col] < 0.0),
                    df[scen_col] * -1,
                    0.0
                )
                df = self.propagate_quantities(
                    df, impact_adj_sub_col, new_column_names={
                        impact_adj_sub_col: impact_adj_ups_col
                    }, replace_columns=False,
                )
                df[scen_col] += df[impact_adj_ups_col]

        # Whether capping was applied affects columns in df
        if capping_applied:
            df = df.drop(columns=[net_impact_col, impact_adj_sub_col, impact_adj_ups_col])
        else:
            df = df.drop(columns=[net_impact_col, impact_adj_sub_col])

        self.ds.mt.set_data(df)

    def _aggregate_lpp(
            self, table: DataTable, value_column: str, new_column_name: str,
    ) -> pd.DataFrame:
        """
        Aggregate point (artificial influences) data to waterbody (sub) level.

        Not suitable for groundwater abstractions. A couple of impact purposes are
        excluded when aggregating complex impacts to the waterbody level:
        'Trib Summary Impacts' and 'LDMU'.

        Args:
            table: Input table instance.
            value_column: Column to aggregate.
            new_column_name: New name for column

        Returns:
            Dataframe containing aggregated (sub) values for specified column.

        """
        df = table.data

        if table.name == 'SupResGW_NBB':
            df = df.loc[
                ~df[table.purpose_column].isin(table.purposes_to_exclude)
            ]

        if df.shape[0] == 0:
            df2 = self.ds.wbs.data[[]].copy()
            df2[new_column_name] = 0.0

        else:

            # Need to merge with a list of all waterbodies, as not every waterbody has all
            # types of impacts - this is required for propagate_quantity()
            df1 = pd.merge(
                df, self.ds.wbs.data[[]], how='right',
                left_on=self.constants.waterbody_id_column, right_index=True,
            )

            # Following merge with complete list of waterbodies nans may be present - just
            # indicates no relevant impacts in a waterbody, i.e. infill with zeros
            df1.loc[df1[value_column].isna(), value_column] = 0.0

            # Aggregate from LPP-level impacts to waterbody-level sub impacts
            df2 = df1.groupby(self.constants.waterbody_id_column)[[value_column]].sum()
            df2 = df2.rename(columns={value_column: new_column_name})

            # Waterbody order not necessarily preserved in aggregation, so ensure here
            df2 = pd.merge(
                df2, self.ds.wbs.data[[]], how='right', left_index=True, right_index=True,
            )

        return df2

    def _aggregate_lpp_gwabs(self, value_column: str, new_column_name: str):
        """
        Aggregate point groundwater abstractions data to waterbody (sub) level.

        Uses *data* attribute of groundwater abstractions (gwabs) table, rather than an
        input table/dataframe.

        Args:
            value_column: Column to aggregate.
            new_column_name: New name for column

        Returns:
            Dataframe containing aggregated (sub) values for specified column.

        """
        df = self.ds.gwabs.data.copy()

        if df.shape[0] == 0:
            df1 = self.ds.wbs.data[[]].copy()
            df1[new_column_name] = 0.0

        else:

            # Loop is to calculate impact of each gwab on all relevant waterbodies (then
            # can sum per waterbody in subsequent step)
            df0s = []
            for i in range(1, 6):
                impacted_wb_col = self.ds.gwabs.impacted_waterbody_columns[i - 1]
                proportion_col = self.ds.gwabs.impact_proportion_columns[i - 1]

                # Temporary column will contain (absolute) impact of a gwab on a given wb
                temp_column = f'{value_column}__WB{i}__TEMP'
                if temp_column in df.columns:
                    raise ValueError(f'Temporary column already exists: {temp_column}')

                df[temp_column] = df[value_column] * df[proportion_col] / 100.0
                df0 = df.groupby([impacted_wb_col])[temp_column].sum().reset_index()
                df0 = df0.rename(columns={impacted_wb_col: self.constants.waterbody_id_column})
                df0 = df0.set_index(self.constants.waterbody_id_column)
                df0s.append(df0)

            # This concat may result in nans in the temp columns where a gwab does not
            # impact on all waterbodies (e.g. if it only impacts one wb, then wbs 2-5 will
            # have nans) - OK to set these nans to zero
            df1 = pd.concat(df0s, axis=1)
            df1[value_column] = 0.0

            # Add up the gwab impacts per waterbody
            columns_to_drop = []  # temporary columns
            for i in range(1, 6):
                temp_column = f'{value_column}__WB{i}__TEMP'
                # see above - nans indicate no impact, i.e. treat as zero/omit
                df1.loc[~df1[temp_column].isna(), value_column] += (
                    df1.loc[~df1[temp_column].isna(), temp_column]
                )
                columns_to_drop.append(temp_column)
            df1 = df1[[value_column]].copy()
            df1 = df1.rename(columns={value_column: new_column_name})

            # Ensure all waterbodies are included (e.g. what if not gwab impact on a wb?)
            df1 = pd.merge(
                df1, self.ds.wbs.data[[]], how='right',
                # left_on=self.constants.waterbody_id_column, right_index=True,
                left_index=True, right_index=True,
            )
            # df1 = df1.set_index(self.constants.waterbody_id_column)
            df1.loc[df1[new_column_name].isna(), new_column_name] = 0.0

        return df1


def propagate_quantity(
        routing_matrix: np.ndarray, sub_values: np.ndarray,
) -> np.ndarray:
    """
    Convert a quantity from "sub" to "ups" by propagating through a waterbody network.

    Notes:
        - All waterbodies must be present in sub_values even if values are zero.
        - The order of waterbodies implied in sub_values needs to match the order
          of nodes in the nx.DiGraph used to create the routing_matrix.

    Args:
        routing_matrix: 2D array that represents a routing matrix obtained by passing a
            nx.DiGraph to the dataset.construct_routing_matrix function or similar.
        sub_values: 1D array that represents the sub quantity to be propagated. Size
            should fit with routing_matrix, i.e. ``sub_values.shape[0]``,
            ``routing_matrix.shape[0]`` and ``routing_matrix.shape[1]`` are all equal.

    Returns:
        Propagated quantity (ups value) with order matching input arrays.

    """
    # Broadcast sub values to 2d array (same value across each row)
    a = np.broadcast_to(sub_values, routing_matrix.shape).T

    # Summing (by column) the element-wise product of modified adjacency matrix and sub
    # values gives ups values
    b = routing_matrix * a
    y = np.sum(b, axis=0)

    return y


def unpropagate_quantity(
        adjacency_matrix: np.ndarray,  ups_values: np.ndarray,
) -> np.ndarray:
    """
    Convert an array of ups values to one of sub values.

    User responsible for ensuring that the orders of adjacency_matrix and ups_values
    are consistent.

    Args:
        adjacency_matrix: Adjacency matrix (available from nx.DiGraph).
        ups_values: 1D array of ups values to convert.

    Returns:
        1D array of sub values of order consistent with input arrays.

    """
    sub_values = (
        ups_values - np.sum(adjacency_matrix * ups_values[:, None], axis=0)
    )
    return sub_values
