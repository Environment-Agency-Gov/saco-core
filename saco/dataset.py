"""
Dataset class to group together core tables plus dataset helper functions.

"""
import os
import itertools
from typing import List, Union, Dict, Iterable
from pathlib import Path
import zipfile
from copy import deepcopy
import warnings
import datetime

import numpy as np
import pandas as pd
import networkx as nx

from .tables import (
    Table, DataTable, IntegratedWBs_NBB, QNaturalFlows_NBB, AbsSensBands_NBB,
    SWABS_NBB, GWABs_NBB, Discharges_NBB, SupResGW_NBB, ASBPercentages, REFS_NBB,
    Seasonal_Lookup, Fix_Flags, Master
)
from .config import Constants
from .utils import check_if_output_path_exists


class Dataset:
    """
    Container to group core data tables for use in calculator/optimiser components.

    Args:
        data_folder: Path to folder containing all input files.
        scenarios: Names of artificial influences scenarios.
        percentiles: Flow percentiles (natural).
        value_types: Indicator of aggregation types ('sub' or 'ups').
        constants: Global constants (by default instance of config.Constants).

    Examples:
        >>> from saco import Dataset
        >>> ds = Dataset(data_folder='/path/to/data/files')
        >>> ds.load_data()

    """
    def __init__(
            self,
            data_folder: Union[str, Path] = None,
            scenarios: List[str] = None,
            percentiles: List[int] = None,
            value_types: List[str] = None,
            constants: Constants = None,
    ):
        self.data_folder = data_folder

        if constants is None:
            self.constants = Constants()

        if scenarios is None:
            scenarios = self.constants.valid_scenarios
        if percentiles is None:
            percentiles = self.constants.valid_percentiles
        if value_types is None:
            value_types = self.constants.valid_value_types

        self.scenarios = scenarios
        self.percentiles = percentiles
        self.value_types = value_types

        #: tables.IntegratedWBs_NBB: Instance of waterbody metadata table.
        self.wbs = IntegratedWBs_NBB()

        #: tables.QNaturalFlows_NBB: Instance of natural flows table.
        self.qnat = QNaturalFlows_NBB()

        #: tables.AbsSensBands_NBB: Instance of abstraction sensitivity bands (ASBs) table.
        self.asbs = AbsSensBands_NBB()

        #: tables.SWABS_NBB: Instance of surface water abstractions table.
        self.swabs = SWABS_NBB()

        #: tables.GWABs_NBB: Instance of groundwater abstractions table.
        self.gwabs = GWABs_NBB()

        #: tables.Discharges_NBB: Instance of discharges table.
        self.dis = Discharges_NBB()

        #: tables.SupResGW_NBB: Instance of complex impacts table.
        self.sup = SupResGW_NBB()

        #: tables.ASBPercentages: Instance of ASB percentage definitions table.
        self.asb_percs = ASBPercentages()

        #: tables.REFS_NBB: Instance of reference flow (EFI+) table.
        self.refs = REFS_NBB()

        #: tables.Seasonal_Lookup: Instance of SWABS disaggregation factors table.
        self.sfac = Seasonal_Lookup()

        #: tables.Fix_Flags: Instance of (optional) waterbody fix flags table.
        self.wbfx = Fix_Flags()

        #: tables.Master: Instance of waterbody level "master" table.
        self.mt = Master(self.scenarios, self.percentiles)

        self._graph = None
        self._routing_matrix = None
        self._adjacency_matrix = None

    def load_tables(
            self, skip_tables: List[str] = None, set_index: bool = False,
            validate: bool = False, optional_tables: List[str] = None,
    ):
        """
        Load data tables from parquet files (sets *data* attributes of tables).

        Args:
            skip_tables: Tables to skip (using *name* attributes of tables, rather than
                *short_name* attributes - i.e. generally WRGIS names).
            set_index: Whether to set index explicitly on read (not typically required
                for files written by data preparation routine).
            validate: Whether to validate data tables against their schemas (again not
                typically required after data have been prepared).
            optional_tables: Tables to be loaded if available in data_folder but skipped
                if unavailable. The default (None) indicates that the Fix_Flags table
                is the only optional table.

        """
        if skip_tables is None:
            skip_tables = []
        if optional_tables is None:
            optional_tables = ['Fix_Flags']

        for table in self.tables:
            if table.name not in skip_tables:
                try:
                    table.load_data(self.data_folder, set_index, validate)
                except FileNotFoundError:
                    if table.name not in optional_tables:
                        raise FileNotFoundError(
                            f'File for table {table.name} not found in {self.data_folder}'
                        )

    def load_graph(self):
        """Load directed graph defining waterbody network (sets *graph* attribute)."""
        graph = nx.read_graphml(os.path.join(self.data_folder, self.graph_file_name))
        self.set_graph(graph)

    def load_routing_matrix(self):
        """Load array used to propagate quantities (sets *routing_matrix* attribute)."""
        arrays = np.load(os.path.join(self.data_folder, self.routing_matrix_file_name))
        self.set_routing_matrix(arrays['routing_matrix'])

    def load_data(
            self, skip_tables: List[str] = None, set_index: bool = False,
            validate: bool = False, optional_tables: List[str] = None,
    ):
        """
        Load data tables, directed graph (waterbody network) and routing matrix.

        Sets *data*, *graph* and *routing_matrix* attributes. Load methods use
        *data_folder* attribute of Dataset and *file_name* attributes of tables. See
        "set" methods to set attributes using in-memory objects.

        Args:
            skip_tables: Tables to skip (using *name* attributes of tables, rather than
                *short_name* attributes - i.e. generally WRGIS names).
            set_index: Whether to set index explicitly on read (not typically required
                for files written by data preparation routine).
            validate: Whether to validate data tables against their schemas (again not
                typically required after data have been prepared).
            optional_tables: Tables to be loaded if available in data_folder but skipped
                if unavailable. The default (None) indicates that the Fix_Flags table
                is the only optional table.

        """
        self.load_tables(skip_tables, set_index, validate, optional_tables)
        self.load_graph()
        self.load_routing_matrix()

    def set_tables(self, tables: Dict[str, pd.DataFrame]):
        """
        Set *data* attributes of tables.

        This method can be used to set the *data* attribute of a table for the first
        time or it can be used to override the existing *data* attribute. Dataframe
        indexes must have been set before being passed in via the tables argument.

        Args:
            tables: Keys are table *short_name* attributes and values are dataframes.

        Notes:
            Valid dictionary keys for the tables argument are (alongside the relevant
            full table names):

                - 'swabs': SWABS_NBB (point surface water abstractions)
                - 'gwabs': GWABs_NBB (point groundwater abstractions)
                - 'dis': Discharges_NBB (point surface water discharges)
                - 'sup': SupResGW_NBB (point "complex impacts")
                - 'qnat': QNaturalFlows_NBB (waterbody natural flows)
                - 'wbs': IntegratedWBs_NBB (waterbody metadata)
                - 'asbs' AbsSensBands_NBB (waterbody abstraction sensitivity bands)
                - 'asb_percs': ASBPercentages (fractional deviations defining the EFI)
                - 'efi': EFI (waterbody environmental flow indicator)

            The *data* attribute of each table can also be set directly (if preferred),
            for example: ``Dataset.swabs.data = pd.DataFrame(...)``.

        """
        for short_name, df in tables.items():
            getattr(self, short_name).set_data(df)

    def set_graph(self, graph: nx.DiGraph = None):
        """
        Set *graph* attribute (directed graph representing waterbody network).

        Args:
            graph: Directed graph representing waterbody network - if not passed it
                will be derived from *wbs* table (IntegratedWBs_NBB).

        """
        if graph is None:
            graph = construct_waterbody_network(
                self.wbs.data.index,
                self.wbs.data[self.wbs.downstream_waterbody_column]
            )
        self._graph = graph
        self._adjacency_matrix = nx.adjacency_matrix(graph).todense()

    def set_routing_matrix(self, routing_matrix: np.ndarray = None):
        """
        Set *routing_matrix* attribute.

        Args:
            routing_matrix: Routing matrix - if not provided it will be constructed
                based on *graph* attribute.

        """
        if routing_matrix is None:
            routing_matrix = construct_routing_matrix(self.graph)
        self._routing_matrix = routing_matrix

    def write_tables(
            self, output_folder: Union[Path, str], overwrite: bool = False,
            table_names: List[str] = None, output_format: str = 'parquet',
            zip_name: str = None, decimal_places: float = None,
    ):
        """
        Write data tables to parquet or (zipped) csv files.

        Args:
            output_folder: Path to output folder where files should be saved.
            overwrite: Whether to overwrite any existing files in output_folder.
            table_names: Tables to write (using *name* attribute of tables, rather
                than *short_name* attribute).
            output_format: Either 'parquet' or 'zip-csv' (currently).
            zip_name: Optional name of zip archive if output_format is 'zip-csv'. If
                not provided zip archive name will default to 'dataset.zip'.
            decimal_places: Number of decimal places to use in rounding. If None
                (default) then no rounding is applied.

        """
        if table_names is None:
            tables = self.tables
        else:
            tables = [table for table in self.tables if table.name in table_names]

        tables = [table for table in tables if table.data is not None]

        if output_format == 'parquet':
            for table in tables:
                output_path = os.path.join(output_folder, table.file_name)
                check_if_output_path_exists(output_path, overwrite)

                cols = sorted(table.data.columns)

                if decimal_places is None:
                    df = table.data
                else:
                    df = table.data.round(decimal_places)

                df[cols].to_parquet(output_path, index=True)

        elif output_format == 'zip-csv':
            if zip_name is None:
                zip_name = 'dataset.zip'
            else:
                if not zip_name.endswith('.zip'):
                    zip_name = zip_name + '.zip'

            output_path = os.path.join(output_folder, zip_name)

            with zipfile.ZipFile(
                    output_path, mode='w', compression=zipfile.ZIP_DEFLATED,
            ) as zf:
                for table in tables:
                    cols = sorted(table.data.columns)

                    if decimal_places is None:
                        df = table.data
                    else:
                        df = table.data.round(decimal_places)

                    # Use ZipInfo object to get sensible file creation timestamp
                    now = datetime.datetime.now()
                    date_time = (
                        now.year, now.month, now.day, now.hour, now.minute, now.second
                    )
                    zi = zipfile.ZipInfo(f'{table.name}.csv', date_time)
                    zi.compress_type = zipfile.ZIP_DEFLATED

                    output_data = df[cols].to_csv().encode('utf-8')
                    zf.writestr(zi, output_data)

        else:
            raise ValueError(f'Unknown output_format: {output_format}')

    def write_graph(
            self, output_folder: Union[Path, str], overwrite: bool = False,
    ):
        """
        Write directed graph of waterbody network in graphml format.

        Args:
            output_folder: Path to output folder where files should be saved.
            overwrite: Whether to overwrite any existing files in output_folder.

        """
        output_path = os.path.join(output_folder, self.graph_file_name)
        check_if_output_path_exists(output_path, overwrite)
        nx.write_graphml(self.graph, output_path)

    def write_routing_matrix(
            self, output_folder: Union[Path, str], overwrite: bool = False,
    ):
        """
        Write routing matrix in compressed numpy (npz) format.

        Args:
            output_folder: Path to output folder where files should be saved.
            overwrite: Whether to overwrite any existing files in output_folder.

        """
        output_path = os.path.join(output_folder, self.routing_matrix_file_name)
        check_if_output_path_exists(output_path, overwrite)
        np.savez_compressed(
            output_path, allow_pickle=False, **{'routing_matrix': self.routing_matrix},
        )

    def get_table(self, name: str) -> Table:
        """
        Return table with a given *name* attribute.

        Args:
            name: Table name (i.e. using *name* attribute).

        Returns:
            Table object.

        """
        table = None
        for _, attr_value in self.__dict__.items():
            if isinstance(attr_value, Table):
                if attr_value.name == name:
                    table = attr_value
        if table is None:
            raise ValueError(f'Table {name} not present in dataset')
        return table

    def find_outlet_waterbodies(self, assessed_outlets_only: bool = True) -> List[str]:
        """
        Find IDs of "outlet" waterbodies (i.e. no *relevant* downstream waterbodies).

        The role of this method is to help find the most downstream waterbodies. This
        can then be used to help identify all waterbodies upstream and so form a domain
        for input to the calculator or optimiser tools.

        If assessed_outlets_only is False, then "outlet" waterbodies are those at the
        very (outer) margins of the waterbody network stored in the *graph* attribute.
        If it is True (the default) then the method identifies only the outermost
        waterbodies of types that are assessed for compliance (as some waterbody types
        near the coast are not assessed for compliance).

        Args:
            assessed_outlets_only: Whether "outlet" waterbodies should only comprise
                waterbodies of types that are assessed for compliance.

        Returns:
            List of outlet waterbody IDs.

        """
        # First identify outlet waterbodies without reference to waterbody type - this
        # is a first-pass if assessed_outlets_only is True, but a final-pass if not
        outlet_waterbodies = []
        for waterbody in list(self.graph.nodes):
            downstream_waterbodies = list(nx.descendants(self.graph, waterbody))
            if len(downstream_waterbodies) == 0:
                outlet_waterbodies.append(waterbody)

        if assessed_outlets_only:
            # For any given outlet, work back upstream and stop at the first assessed
            # waterbody type. Working from downstream to upstream in the way the graph
            # is "pruned" should avoid any issues where e.g. an assessed waterbody is
            # downstream of an unassessed waterbody (if this ever occurs in the data).
            # All impacts relevant to any waterbody in the domain will still be included
            # in calculations/optimisation.

            _outlet_wbs = outlet_waterbodies
            _graph = deepcopy(self.graph)

            all_types_ok = False
            while not all_types_ok:
                _wbs_to_remove = self.wbs.data.loc[
                    (self.wbs.data.index.isin(_outlet_wbs))
                    & (self.wbs.data[self.wbs.waterbody_type_column].isin(
                        self.wbs.unassessed_types
                    ))
                ].index.tolist()

                if len(_wbs_to_remove) > 0:
                    _graph.remove_nodes_from(_wbs_to_remove)

                    _new_outlet_wbs = []
                    for wb in list(_graph.nodes):
                        downstream_waterbodies = list(nx.descendants(_graph, wb))
                        if len(downstream_waterbodies) == 0:
                            _new_outlet_wbs.append(wb)
                    _outlet_wbs = _new_outlet_wbs

                else:
                    all_types_ok = True
                    outlet_waterbodies = _outlet_wbs

        return outlet_waterbodies

    def identify_upstream_waterbodies(
            self, outlet_waterbodies: Union[str, List[str]],
    ) -> List[str]:
        """
        Identify all waterbodies upstream of (and including) a waterbody(s).

        This method can be used to define a domain for the calculator or optimiser
        tools. Just one list is returned, no matter how many outlet_waterbodies are
        provided as an argument to the method. The outlet_waterbodies supplied do not
        have to be at the margins of the waterbody network (i.e. the method can identify
        catchments upstream of any waterbody(s)).

        Args:
            outlet_waterbodies: One or more waterbodies.

        Returns:
            List of waterbodies upstream of (all) outlet_waterbodies (list includes the
            outlet_waterbodies themselves).

        """
        if isinstance(outlet_waterbodies, str):
            outlet_waterbodies = [outlet_waterbodies]

        upstream_waterbodies = []
        for outlet_waterbody in outlet_waterbodies:
            wbs = list(nx.ancestors(self.graph, outlet_waterbody))
            upstream_waterbodies.extend(wbs)
            upstream_waterbodies.append(outlet_waterbody)

        upstream_waterbodies = list(set(upstream_waterbodies))

        return upstream_waterbodies

    def set_flow_targets(
            self, overall_target: str = 'compliant', use_fix_flags_table: bool = True,
            custom_targets: Dict = None, overwrite_existing: bool = False,
            df: pd.DataFrame = None,
    ):
        """
        Set flow target columns in Master table (in *Dataset.mt.data* attribute).

        Args:
            overall_target: Overall flow target to use for all waterbodies (which can
                be overridden by custom_targets and/or df arguments. See notes below for
                valid arguments.
            use_fix_flags_table: Whether to use Fix_Flags table if available in the
                Dataset.
            custom_targets: For overriding overall_target for specific waterbodies. See
                discussion/example below for a guide to formulating this dictionary.
            overwrite_existing: Whether to overwrite any flow target columns that
                already exist in the dataframe. If False (default) then overall_target
                and custom_targets will not be used to overwrite existing flow target
                columns. However, if df is not None, it will be used regardless of
                overwrite_existing value.
            df: Dataframe indexed by waterbody and containing any flow target columns
                that should be given priority (Ml/d). It does not need to contain all
                waterbodies in the domain/dataset. Used regardless of
                overwrite_existing.

        Notes:

            Valid arguments for overall_target are: 'compliant', 'band-1', 'band-2',
            'band-3', 'none' and 'no-det'. The first four options refer to compliance
            bands (relative to the EFI unless a waterbody has a different target, which
            will be indicated by its ASB). In each case, the target is the minimum flow
            needed to achieve the band (i.e. the lower bound/edge of the band). 'none'
            and 'band-3' are aliases (i.e. no flow target / trivial target of zero).

            The final option ('no-det') stands for "no deterioration" of compliance
            band between the RA and FL scenarios. Again, this is translated into the
            minimum flow required to meet this condition, i.e. the lower bound of the
            appropriate band. Note that this is not the same as no change between the
            RA and FL scenario flows (which could be implemented by passing an
            appropriate dataframe via the df argument).

            If the ``use_fix_flags_table`` argument is set to True, the Fix_Flags table
            will be used to define initial targets (if present in the Dataset). The
            Fix_Flags table indicates whether the target for each waterbody is
            compliance (3), no deterioration (0) or none/do-not-fix (-1). The table is
            intended to align with targets used in the National Framework 2
            Environmental Destination work programme. Note that the ``custom_targets``
            and ``df`` arguments will override the Fix_Flags table if provided.

            The custom_targets argument can be used to override the overall_target for
            specific waterbodies. It should be provided as a dictionary of dictionaries,
            as per the following example:
            ``{('FL', 95): {'band-1': ['waterbody-id-1', 'waterbody-id-2', ...]}}``.

            Finally, a dataframe of flow targets for one or more waterbodies can also be
            supplied through the df argument. This is given priority and it will also be
            used regardless of the overwrite_existing argument (currently). This
            dataframe should have waterbody ID as its index (with name EA_WB_ID) and
            one or more flow target columns (in Ml/d) whose names follow
            ``f'QT{S}Q{P}'``, where ``S`` is a scenario name and ``P`` is a flow
            percentile.

        """
        valid_targets = ['compliant', 'band-1', 'band-2', 'band-3', 'none', 'no-det']

        if overall_target not in valid_targets:
            raise ValueError(f'Overall flow target not recognised: {overall_target}')

        df0 = self._calculate_flow_targets(
            overall_target, use_fix_flags_table, custom_targets,
        )

        qt_cols = self.mt.get_value_columns(self.constants.qt_abb)
        if not overwrite_existing:
            new_cols = [col for col in qt_cols if col not in self.mt.data.columns]
            df0 = df0[new_cols]

        cols_to_drop = [col for col in df0.columns if col in self.mt.data.columns]
        df1 = self.mt.data.drop(columns=cols_to_drop)

        df1 = pd.merge(df1, df0, how='left', left_index=True, right_index=True)

        # TODO: Consider changing so that this respects overwrite_existing
        if df is not None:
            warnings.warn(
                'If df argument is not None, any flow target columns in df will be used, '
                'regardless of overwrite_existing argument value.'
            )

            new_cols = [col for col in qt_cols if col in df.columns]
            df0 = df[new_cols].copy()
            if len(list(set(qt_cols) - set(df0.columns))) > 0:
                warnings.warn(
                    'Flow target input dataframe only contains some of the '
                    'scenario/percentile combinations in the dataset.'
                )

            df0.columns = [f'{col}__NEW' for col in df0.columns]
            df1 = pd.merge(
                df1, df0, how='left', left_index=True, right_index=True
            )
            for qt_col in new_cols:
                df1[qt_col] = np.where(
                    np.isfinite(df1[f'{qt_col}__NEW']), df1[f'{qt_col}__NEW'], df1[qt_col]
                )
                df1 = df1.drop(columns=f'{qt_col}__NEW')

        self.mt.set_data(df1)

    def set_optimise_flag(
            self, swabs_exclusions: Iterable[str] | None = ('reservoir', 'ldmu', 'lake'),
            purpose_exclusions: Iterable[str] = None, exclude_deregulated: bool = True,
            exclude_below: float | None = 0.1,
            exclude_below_case: tuple | None = ('FL', 95),
    ):
        """
        Set the (at least initial) value of the optimise flag columns in relevant tables.

        The main role of this method is to provide defaults on what is included and
        excluded (i.e. available for change or not) during optimisation.

        Args:
            swabs_exclusions: Types of surface water abstractions to exclude (one or
                more of: 'reservoir', 'ldmu', 'lake'. An empty list or None may also be
                provided.
            purpose_exclusions: Purpose codes to exclude. Note that just the "start" of
                a purpose code can be supplied to act as a wildcard. For example,
                purpose_exclusions=['WPWS330', 'E'] would flag any rows with the specific
                purpose WPWS330 for exclusion from optimisation, as well as any purposes
                beginning E. No exclusions by default.
            exclude_deregulated: Whether to exclude "deregulated licences" from
                optimisation - default is True, as not amenable to change in reality.
            exclude_below: Threshold impact (Ml/d) below which (<=) abstractions should
                be excluded during optimisation. Use None or 0.0 to indicate that there
                should not be any exclusions based on impact magnitude. Based on
                scenario and percentile combination defined by exclude_below_case.
            exclude_below_case: Scenario and percentile combination to use in
                conjunction with exclude_below threshold. By default, impacts less than
                or equal to exclude_below under the fully licensed scenario at Q95 will
                be flagged for exclusion from optimisation. Expects tuple like
                ('FL', 95).

        Notes:

            This method operates on the *data* attributes of the SWABS_NBB, GWABs_NBB
            and SupResGW_NBB (complex) tables. If not already present, it inserts a flag
            column (called 'Optimise_Flag') that is used in the optimiser to
            include/exclude (1/0) particular table rows in the optimisation. By default,
            SWABS and GWABS rows are included (if not ruled out by one of the exlusion
            arguments), whereas complex impacts are excluded. If 'Optimise_Flag' columns
            are already present in the tables, this method will not modify them.

            Options are available to:

                - Flag particular types of surface water abstractions for exclusion from
                  optimisation (in line with the Fix-It tool).
                - Flag particular abstraction purposes for exclusion from optimisation.
                - Flag that deregulated licences should be excluded during optimisation.
                - Flag that impacts below some threshold should be excluded during
                  optimisation.

            Note that exclusions based on purpose are applied to both surface water and
            groundwater abstractions.

            A user also has the option to (1) modify the Optimise_Flag columns created by
            this method or (2) set them manually. For complex impacts (in table
            SupResGW_NBB), manual intervention is necessary (by default these impacts
            are excluded from optimisation). See tutorial in documentation for complex
            flag details.

        """
        if swabs_exclusions is None:
            swabs_exclusions = []
        if purpose_exclusions is None:
            purpose_exclusions = []
        if exclude_below is None:
            exclude_below = 0.0

        if self.swabs.optimise_flag_column in self.swabs.data.columns:
            warnings.warn(
                'Optimise_Flag column already present in SWABS_NBB - it will not be '
                'altered by set_optimise_flag method.'
            )
        else:
            self.swabs.data[self.swabs.optimise_flag_column] = 1

            if 'reservoir' in swabs_exclusions:
                self.swabs.data.loc[
                    self.swabs.data[self.swabs.reservoir_flag_column] > 0,
                    self.swabs.optimise_flag_column
                ] = 0
            if 'ldmu' in swabs_exclusions:
                self.swabs.data.loc[
                    self.swabs.data[self.swabs.ldmu_flag_column] > 0,
                    self.swabs.optimise_flag_column
                ] = 0
            if 'lake' in swabs_exclusions:
                for lake_col in self.swabs.lake_flag_columns:
                    self.swabs.data.loc[
                        self.swabs.data[lake_col] > 0, self.swabs.optimise_flag_column
                    ] = 0

            self._set_common_exclusions(
                    'SWABS_NBB', purpose_exclusions, exclude_deregulated,
                exclude_below, exclude_below_scenario=exclude_below_case[0],
                exclude_below_percentile=exclude_below_case[1],
            )

        if self.gwabs.optimise_flag_column in self.gwabs.data.columns:
            warnings.warn(
                'Optimise_Flag column already present in GWABs_NBB - it will not be '
                'altered by set_optimise_flag method.'
            )
        else:
            self.gwabs.data[self.gwabs.optimise_flag_column] = 1

            self._set_common_exclusions(
                'GWABs_NBB', purpose_exclusions, exclude_deregulated,
                exclude_below, exclude_below_scenario=exclude_below_case[0],
                exclude_below_percentile=exclude_below_case[1],
            )

        if self.sup.optimise_flag_column in self.sup.data.columns:
            warnings.warn(
                'Optimise_Flag column already present in SupResGW_NBB - it will not be '
                'altered by set_optimise_flag method.'
            )
        else:
            self.sup.data[self.sup.optimise_flag_column] = 0

    def infer_mean_abstraction(
            self, scenario: str = 'FL', percentile: int = 95,
            exclude_swabs_with_hofs: bool = True, exclude_gwabs: List[str] = None,
            exclude_swabs: List[str] = None,
    ):
        """
        Infer mean abstraction from impacts under a given scenario and percentile.

        Args:
            scenario: Abbreviation of artificial influences scenario used as basis for
                inferring long-term average abstraction.
            percentile: Flow percentile (natural) used as basis for inferring long-term
                average abstraction.
            exclude_swabs_with_hofs: Whether to exclude SWABS with HOFs from long-term
                average calculations.
            exclude_gwabs: Groundwater abstractions whose long-term average should not
                be inferred. List should contain entries from UNIQUEID in GWABs_NBB.
            exclude_swabs: Surface water abstractions whose long-term average should not
                be inferred. List should contain entries from UNIQUEID in SWABS_NBB.

        Notes:

            WRGIS models the relationships between long-term average abstraction and
            impacts at different flow percentiles. Here we use these relationships to
            estimate long-term average abstraction given impacts at specific (single)
            flow percentile. This is done under an assumption that the relative
            seasonal/FDC profile of impacts remains constant.

            The impact under a given scenario/percentile combination includes the
            effect of local consumptiveness. The long-term average numbers calculated
            using this method *exclude* local consumptiveness. This is reflected by the
            "WR" vs "NR" suffixes in the impact numbers for a specific percentile (WR =
            "water returned") vs the long-term average abstraction numbers (NR = "no
            water returned").

            Lists of abstractions to be excluded from long-term average calculations can
            be supplied via the exclude_gwabs and exclude_swabs arguments. Abstractions
            to be excluded should be specified using their UNIQUEID. If a long-term
            average column is already present before this method is called, the existing
            value will be retained. If not, a NaN will be inserted for these
            abstractions.

            This method operates only for the SWABS_NBB and GWABs_NBB tables. Complex
            abstractions/impacts in the SupResGW_NBB table are not handled. By default,
            SWABS with HOFs are excluded from long-term average calculations (see
            ``exclude_swabs_with_hofs`` argument). This is because the method may not
            yield a reasonable long-term average abstraction if the impact in SWABS_NBB
            at the reference percentile is being constrained by the HOF condition.

            For example, a SWAB might be "off" at Q95 due to a HOF condition. However,
            it might well be unreasonable for its long-term average to become zero.
            Hence it is not appropriate to apply the seasonal disaggregation factors to
            obtain a revised long-term average in this case.

        """
        if exclude_gwabs is None:
            exclude_gwabs = []
        if exclude_swabs is None:
            exclude_swabs = []

        if self.gwabs.data.shape[0] > 0:
            self.gwabs.infer_mean_abstraction(scenario, percentile, exclude_gwabs)

        if self.swabs.data.shape[0] > 0:
            self.swabs.infer_mean_abstraction(
                scenario, percentile, self.sfac, exclude_swabs_with_hofs, exclude_swabs,
            )

    def _calculate_flow_targets(
            self, overall_target: str, use_fix_flags_table: bool = True,
            custom_targets: Dict = None,
    ) -> pd.DataFrame:
        """See docstring for set_flow_targets, for which this method is a helper."""
        col_mapper = {
            'compliant': '__COMPLIANT', 'band-1': '__BAND1', 'band-2': '__BAND2',
            'band-3': '__BAND3', 'none': '__NONE', 'no-det': '__NO_DET',
        }

        df = self.mt.data.copy()
        for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
            refs_col = self.mt.get_refs_column(percentile)
            qt_col = self.mt.get_qt_column(scenario, percentile)
            qnat_col = self.mt.get_qnat_column(percentile, self.constants.ups_abb)
            ra_comp_col = self.mt.get_comp_column(self.constants.ra_abb, percentile)

            df['__COMPLIANT'] = df[refs_col]
            f1 = self.constants.compliance_bin_edges[2]
            df['__BAND1'] = df[refs_col] + f1 * df[qnat_col]
            f2 = self.constants.compliance_bin_edges[1]
            df['__BAND2'] = df[refs_col] + f2 * df[qnat_col]
            df['__BAND3'] = 0.0
            df['__NONE'] = 0.0

            if ra_comp_col in df.columns:
                df['__NO_DET'] = df[refs_col]
                df['__NO_DET'] = np.where(
                    df[ra_comp_col] == 1, df['__BAND1'], df['__NO_DET']
                )
                df['__NO_DET'] = np.where(
                    df[ra_comp_col] == 2, df['__BAND2'], df['__NO_DET']
                )
                df['__NO_DET'] = np.where(
                    df[ra_comp_col] == 3, df['__BAND3'], df['__NO_DET']
                )
            if (overall_target == 'no-det') and (ra_comp_col not in df.columns):
                raise ValueError(
                    'RA compliance column is needed to set no-det target. Run '
                    'Calculator to obtain suitable Dataset.'
                )
            else:
                # TODO: Consider warning if Fix_Flags table not present?
                if use_fix_flags_table and (self.wbfx.data is not None):
                    df[qt_col] = df['__COMPLIANT']
                    df[qt_col] = np.where(
                        df.index.isin(
                            self.wbfx.data.loc[
                                self.wbfx.data[self.wbfx.fix_flag_column] == self.wbfx.targets['no-det']
                            ].index.tolist()
                        ),
                        df['__NO_DET'],
                        df[qt_col]
                    )
                    df[qt_col] = np.where(
                        df.index.isin(
                            self.wbfx.data.loc[
                                self.wbfx.data[self.wbfx.fix_flag_column] == self.wbfx.targets['none']
                            ].index.tolist()
                        ),
                        df['__NONE'],
                        df[qt_col]
                    )
                else:
                    df[qt_col] = df[col_mapper[overall_target]]

            if custom_targets is not None:
                k = (scenario, percentile)
                if k in custom_targets.keys():
                    dc = custom_targets[k]
                    for target, waterbodies in dc.items():
                        if (target == 'no-det') and (ra_comp_col not in df.columns):
                            raise ValueError(
                                'RA compliance column is needed to set no-det target. '
                                'Run Calculator to obtain suitable Dataset.'
                            )
                        df[qt_col] = np.where(
                            df.index.isin(waterbodies), df[col_mapper[target]], df[qt_col]
                        )

            # Ensure no negative targets - only relevant to unassessed waterbodies that
            # are assigned an EFI of zero (and should not affect results anyway)
            df[qt_col] = np.maximum(df[qt_col], 0.0)

            #  __NO_DET column may not be present
            df = df.drop(columns=col_mapper.values(), errors='ignore')

        qt_cols = self.mt.get_value_columns(self.constants.qt_abb)
        df = df[qt_cols]

        return df

    def _set_common_exclusions(
            self, table_name: str, purpose_exclusions: Iterable[str],
            exclude_deregulated: bool, exclude_below: float,
            exclude_below_scenario: str, exclude_below_percentile: int,
    ):
        """See docstring for set_optimise_flag, for which this method is a helper."""
        if table_name == 'SWABS_NBB':
            table = self.swabs
        elif table_name == 'GWABs_NBB':
            table = self.gwabs
        else:
            raise ValueError(f'Unexpected table name: {table_name}')

        for purpose in purpose_exclusions:
            table.data.loc[
                table.data[table.purpose_column].str.startswith(purpose),
                table.optimise_flag_column
            ] = 0

        if exclude_deregulated:
            table.data.loc[
                table.data[table.licence_expiry_column] == 'D',
                table.optimise_flag_column
            ] = 0

        if exclude_below > 0.0:
            impact_col = table.get_value_column(
                exclude_below_scenario, exclude_below_percentile,
            )
            table.data.loc[
                table.data[impact_col] <= exclude_below,
                table.optimise_flag_column
            ] = 0

    @property
    def tables(self) -> List[Table]:
        tables = []
        for _, attr_value in self.__dict__.items():
            if (
                isinstance(attr_value, Table)
                or isinstance(attr_value, DataTable)
            ):
                tables.append(attr_value)
        return tables

    @property
    def table_names(self) -> List[str]:
        return [table.name for table in self.tables]

    @property
    def factors(self) -> Dict[str, List[Union[str, int]]]:
        return {
            'scenarios': self.scenarios, 'percentiles': self.percentiles,
            'value_types': self.value_types,
        }

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def ai_table_names(self) -> List[str]:
        return ['GWABs_NBB', 'SWABS_NBB', 'Discharges_NBB', 'SupResGW_NBB']

    @property
    def ai_tables(self) -> List[DataTable]:
        return [table for table in self.tables if table.name in self.ai_table_names]

    @property
    def derived_table_names(self) -> List[str]:
        return ['REFS_NBB', 'Master', 'Fix_Flags']

    @property
    def input_tables(self) -> List[Table]:
        return [
            table for table in self.tables if table.name not in self.derived_table_names
        ]

    @property
    def ai_variable_abbs(self) -> List[str]:
        return [table.variable_abb for table in self.ai_tables]

    @property
    def routing_matrix(self) -> np.ndarray:
        return self._routing_matrix

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._adjacency_matrix

    @property
    def waterbody_id_tables(self) -> List[Table]:
        tables = []
        for table in self.tables:
            if table.index_name is not None:
                if table.index_name == self.constants.waterbody_id_column:
                    tables.append(table)
        return tables

    @property
    def waterbody_id_table_names(self) -> List[str]:
        return [table.name for table in self.waterbody_id_tables]

    @property
    def graph_file_name(self) -> str:
        return 'waterbody_network.graphml'

    @property
    def routing_matrix_file_name(self) -> str:
        return 'routing_matrix.npz'


def construct_waterbody_network(
        source_waterbodies: List[str], destination_waterbodies: List[str],
):
    """
    Construct a networkx directed graph from lists of source/destination waterbodies.

    Source and destination waterbodies typically taken from IntegratedWBs_NBB. Some
    waterbodies are "destination-only", i.e. the most downstream waterbodies that are
    not included/evaluated in key WRGIS/WBAT tables.

    Args:
        source_waterbodies: List of source waterbodies/nodes.
        destination_waterbodies: Corresponding list of destination waterbodies/nodes
            (i.e. length and order matches source_waterbodies).

    Returns:
        Directed graph with waterbody IDs as node labels.

    """
    # Find list of most-downstream waterbodies that should be excluded from graph due
    # to not featuring in key WRGIS/WBAT tables and calculations
    termination_wbs = list(
        set(destination_waterbodies).difference(set(source_waterbodies))
    )

    graph = nx.DiGraph()

    # All waterbodies must be added as nodes explicitly, as some may not have a
    # downstream waterbody that does not appear in termination_wbs
    graph.add_nodes_from(list(source_waterbodies))

    for src_wb, dst_wb in zip(source_waterbodies, destination_waterbodies):
        if dst_wb not in termination_wbs:
            graph.add_edge(src_wb, dst_wb)

    return graph


def construct_routing_matrix(graph: nx.DiGraph):
    """
    Modify the adjacency matrix to help with "sub" to "ups" conversions.

    Args:
        graph: Directed graph expressing waterbody relationships (see
            construct_waterbody_network function).

    Returns:
        Routing matrix suitable for input to propagate_quantity function.

    """
    nodes = list(graph.nodes)

    adj = nx.adjacency_matrix(graph).todense()
    mod_adj = adj.copy()

    for node in nodes:
        i = nodes.index(node)

        successors = nx.dfs_successors(graph, node)
        successors = set(list(itertools.chain(*successors.values())))

        idx = [j for j, val in enumerate(nodes) if val in successors]

        mod_adj[i, idx] = 1

    np.fill_diagonal(mod_adj, 1)

    return mod_adj


def subset_dataset_on_wbs(ds: Dataset, waterbodies: List[str]) -> Dataset:
    """
    Subset a Dataset to include only rows relevant to specified waterbodies.

    Args:
        ds: Dataset to subset.
        waterbodies: Waterbodies to use to subset domain.

    Returns:
        Subset dataset.

    """
    # Dataframe of waterbodies used to subset other tables
    df_wbs = pd.DataFrame({ds.wbs.index_name: waterbodies}).set_index(ds.wbs.index_name)

    ds1 = Dataset()

    for table in ds.tables:
        if table.name in ['ASBPercentages', 'Seasonal_Lookup']:
            df = table.data

        elif (table.name == 'Fix_Flags') and (table.data is None):
            df = None

        elif table.name in ds.waterbody_id_table_names:
            df = df_wbs.merge(
                table.data, how='left', left_index=True, right_index=True,
            )

        else:
            # GWABS handled separately to other AI, as it has multiple impacted
            # waterbody columns
            if table.name == 'GWABs_NBB':
                gwabs_dfs = []

                # Duplicates may arise in this loop - e.g. a gwab could impact two
                # waterbodies that are both within the domain - so duplicate rows are
                # filtered out below
                for i in range(len(ds.gwabs.impacted_waterbody_columns)):
                    df = table.data.loc[
                        table.data[ds.gwabs.impacted_waterbody_columns[i]]
                        .isin(waterbodies)
                    ]
                    gwabs_dfs.append(df)

                df = pd.concat(gwabs_dfs)

                # Need to reset index before checking for duplicates, otherwise
                # drop_duplicates will ignore index
                df = df.reset_index()
                df = df.drop_duplicates()
                df = df.set_index(ds.gwabs.index_name)

            else:
                # Non-GWABS AI
                df = df_wbs.merge(
                    table.data, how='inner', left_index=True,
                    right_on=table.waterbody_id_column,
                )
                df.index.name = table.index_name

        if df is None:
            pass
        else:
            ds1.get_table(table.name).set_data(df.copy())

    ds1.set_graph()
    ds1.set_routing_matrix()

    return ds1


def subset_dataset_on_columns(
        ds: Dataset, scenarios: List[str], percentiles: List[int],
) -> Dataset:
    """
    Subset dataset on specific value columns.

    Args:
        ds: Dataset to subset.
        scenarios: Scenarios to retain in value columns.
        percentiles: Percentiles to retain in value columns.

    Returns:
        Dataset restricted to relevant value columns.

    """
    ds1 = deepcopy(ds)
    ds1.scenarios = scenarios
    ds1.percentiles = percentiles

    for table in ds1.tables:
        if isinstance(table, DataTable) or (table.name == 'Master'):
            old_value_cols = table.value_columns

            # Set scenario + percentile factors, which in turn modifies value_columns
            # attribute
            if table.name == 'Master':
                table._scenarios = scenarios
                table._percentiles = percentiles
            else:
                table.set_factors(scenarios=scenarios, percentiles=percentiles)

            if table.data is None:
                pass
            else:
                aux_cols = [
                    col for col in table.data.columns if col not in old_value_cols
                ]

                if table.name in ['SWABS_NBB', 'GWABs_NBB']:
                    _aux_cols = []
                    for col in aux_cols:
                        if f'{table.variable_abb}LTA' in col:
                            scenario = col.replace(f'{table.variable_abb}LTA', '')[:-2]
                            if scenario in scenarios:
                                _aux_cols.append(col)
                        else:
                            _aux_cols.append(col)
                    aux_cols = _aux_cols

                cols = list(set(aux_cols + table.value_columns))
                cols = [col for col in cols if col in table.data.columns]
                df = table.data[cols].copy()
                table.set_data(df)

    return ds1


def concatenate_datasets(
        datasets: List[Dataset],
        tables_to_skip: Iterable[str] | None = (
            'IntegratedWBs_NBB', 'ASBPercentages', 'AbsSensBands_NBB'
        ),
) -> Dataset:
    """
    Concatenate dataset table columns (i.e. "wide" concatenation).

    Intended to help concatenate datasets that include the same domain and artificial
    influences but different value columns (scenario and percentile combinations). The
    method assumes that input datasets do not contain "overlapping" value columns.

    Args:
        datasets: Datasets to concatenate.
        tables_to_skip: Tables for which concatenation should not be attempted. These
            tables will just be taken from the first Dataset in the list when
            constructing the output Dataset.

    Returns:
        Concatenated Dataset.

    """
    if tables_to_skip is None:
        tables_to_skip = []

    ds1 = deepcopy(datasets[0])
    scenarios = [ds1.scenarios[0]]
    percentiles = [ds1.percentiles[0]]

    for ds in datasets[1:]:
        scenarios.append(ds.scenarios[0])
        percentiles.append(ds.percentiles[0])

        for table in ds.tables:
            if table.name not in tables_to_skip:
                merge_cols = []
                for col in table.value_columns:
                    if col not in ds1.get_table(table.name).data.columns:
                        merge_cols.append(col)

                df = pd.merge(
                    ds1.get_table(table.name).data, table.data[merge_cols],
                    how='left', left_index=True, right_index=True,
                )
                ds1.get_table(table.name).set_data(df)

    scenarios = list(set(scenarios))
    percentiles = list(set(percentiles))

    ds1.scenarios = scenarios
    ds1.percentiles = percentiles

    for table in ds1.tables:
        if isinstance(table, DataTable):
            table.set_factors(scenarios, percentiles, table.value_types)
        else:
            if hasattr(table, 'scenarios'):
                table._scenarios = scenarios
            if hasattr(table, 'percentiles'):
                table._percentiles = percentiles

    return ds1


def find_differences(
        ds1: Dataset, ds2: Dataset, table_names: List[str],
        require_rows_match: bool = True, require_columns_match: bool = True,
        significance_threshold: float | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate differences between data tables of two Dataset instances.

    Differences are calculated as ds2 - ds1. So ds1 should be treated as the reference
    Dataset and ds2 as the comparison Dataset. If e.g. an abstraction impact is lower
    in ds2 then the difference will come through as negative (i.e. a reduction in
    impact).

    Function will only work for tables that are derived from tables.DataTable class
    (currently).

    Args:
        ds1: Reference dataset.
        ds2: Comparison dataset.
        table_names: Names of tables to compare.
        require_rows_match: Whether data tables should match completely on rows
            (indexes) (ignoring order).
        require_columns_match: Whether data tables should match completely on columns
            (ignoring order).
        significance_threshold: If absolute differences are below this threshold then
            they will be set to zero (to avoid spurious precision in outputs). By
            default no changes are made.

    Returns:
        Dictionary with keys as table names and values as dataframes whose value
            columns contain the differences.

    """
    dfs = {}
    for table_name in table_names:
        df1 = ds1.get_table(table_name).data
        df2 = ds2.get_table(table_name).data

        if require_rows_match:
            if df1.shape[0] == df2.shape[0]:
                if not np.all(df1.sort_index() == df2.sort_index()):
                    raise ValueError(f'Indexes do not match for table {table_name}.')
            else:
                raise ValueError(f'Different numbers of rows for table {table_name}.')

        if require_columns_match:
            if df1.shape[1] == df2.shape[1]:
                if not np.all(sorted(df1.columns) == sorted(df2.columns)):
                    raise ValueError(f'Column names do not match for table {table_name}.')
            else:
                raise ValueError(f'Different numbers of columns for table {table_name}.')

        # Difference calculations
        ref_cols = [
            col for col in df1.columns if col in ds2.get_table(table_name).value_columns
        ]
        df3 = pd.merge(
            df2, df1[ref_cols], how='left', left_index=True, right_index=True,
            suffixes=(None, '__REF'),
        )
        for value_col in ds2.get_table(table_name).value_columns:
            df3[value_col] -= df3[f'{value_col}__REF']

            if significance_threshold is not None:
                df3[value_col] = np.where(
                    np.abs(df3[value_col]) < significance_threshold,
                    0.0,
                    df3[value_col]
                )

        cols_to_drop = [col for col in df3.columns if col.endswith('__REF')]
        df3 = df3.drop(columns=cols_to_drop)

        # Only calculating differences for value columns, so worth merging back in other
        # auxiliary/metadata columns (either left or inner join should be fine)
        other_cols = [col for col in df2.columns if col not in df3.columns]
        df3 = pd.merge(df2[other_cols], df3, how='inner', left_index=True, right_index=True)

        dfs[table_name] = df3.sort_index()

    return dfs
