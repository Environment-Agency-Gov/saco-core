"""
Optimisation of abstraction impact reductions to meet flow targets.

"""
from typing import List, Dict, Tuple
import itertools
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp

from .config import Constants
from .tables import GWABs_NBB, SWABS_NBB
from .dataset import Dataset, concatenate_datasets, find_differences
from .calculate import Calculator
from .model import DataPreparer, ArrayBuilder, Model, SWABS, GWABS, AuxiliaryInfo


class GWABS_Changes(GWABs_NBB):

    @property
    def name(self) -> str:
        return 'GWABS_Changes'

    @property
    def short_name(self) -> str:
        return 'gwabs_chg'


class SWABS_Changes(SWABS_NBB):

    @property
    def name(self) -> str:
        return 'SWABS_Changes'

    @property
    def short_name(self) -> str:
        return 'swabs_chg'


class OutputDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.swabs_chg = None
        self.gwabs_chg = None


class Optimiser:
    """
    For finding abstraction impact reductions that help to meet flow targets.

    In most use cases, the steps needed to use the Optimiser are (code example below):

        - Prepare/load an input ``Dataset``
        - Create an instance of this (the ``Optimiser``) class
        - Execute the optimisation using the ``run`` method.

    The ``run`` method obtains a solution via mixed integer (binary) linear
    programming and returns a ``Dataset`` augmented with required impact reductions
    (see below).

    Args:
        ds: Input Dataset.
        scenarios: Names/abbreviations of artificial influences scenarios for which
            optimisation should be performed. If None (default) then taken from
            input_dataset.
        percentiles: Flow percentiles (natural) for which optimisation should be
            performed. If None (default) then taken from input_dataset.
        domain: List of waterbody IDs indicating domain/catchment to be optimised. If
            None (default) then all waterbodies in input_dataset will be included in
            the domain.
        objectives: Sequence of objectives to use in optimisation. Current options are
            either 'max-abstraction' or ['max-abstraction', 'max-point-equality']. The
            latter is the default. See discussion above for a description of objectives.
        solver: Solver name understood by cvxpy.
        raise_external_hof_error: Whether to raise an error if the waterbody defining a
            HOF condition is located outside the domain. If False (default), it is
            assumed that the HOF condition is met (and it does not form an explicit
            constraint in the problem).
        primary_relaxation_factor: If running with both objectives, this factor is used
            to relax the total abstraction constraint used when solving for the second
            objective ('max-point-equality'). A value of 0.01 means that the constraint
            is relaxed by 1% of the maximum possible total abstraction. Default is not
            to apply any relaxation.
        reference_dataset: Optional dataset to use as a reference against which to
            derive changes (i.e. for inference of required impact reductions after
            optimisation conducted). Default (None) is not to use this argument (and
            derive changes relative to input_dataset).
        constants: Global constants defined by default in config.Constants.

    Notes:
        A starting point for the input_dataset might be a Dataset comprising the WRGIS
        tables. Make any desired modifications to this (or some other) "base" Dataset
        before creating an instance of ``Optimiser``. See :doc:`reference-dataset`
        documentation and examples.

        It is currently recommended to run with the default objectives. These are to solve
        for maximum total abstraction in the domain first and then maximum equality of
        any proportional impact reductions second. The maximum total abstraction from
        solving for the first objective becomes a constraint when solving for the second
        objective. This constraint can be relaxed using the primary_relaxation_factor
        argument (see above).

        Sometimes a flow target might be impossible meet. This can occur if some impacts
        are held constant (i.e. not available for the Optimiser to change). Currently,
        discharges (Discharges_NBB) and complex impacts (SupResGW_NBB) are held constant.
        A user may specify that certain rows in SWABS_NBB and GWABs_NBB should also be
        left alone. If a target cannot feasibly be met, the Optimiser drops the target
        and provides a warning to the user, so that they can reconsider the setup if need
        be.

    Examples:
        >>> from saco import Dataset, Optimiser
        >>>
        >>> ds = Dataset(data_folder='/path/to/data/files')
        >>> ds.load_data()
        >>> ds.set_flow_targets()
        >>> ds.set_optimise_flag()
        >>>
        >>> optimiser = Optimiser(ds)
        >>> output_dataset = optimiser.run()

    """
    def __init__(
            self,
            ds: Dataset,
            scenarios: List[str] = None,
            percentiles: List[int] = None,
            domain: List[str] = None,
            objectives: List[str] | str = None,
            solver: str = cp.SCIPY,
            raise_external_hof_error: bool = False,
            primary_relaxation_factor: float = None,
            reference_dataset: Dataset = None,
            constants: Constants = None,
    ):
        if constants is None:
            constants = Constants()
        self.constants = constants

        self.input_dataset = ds

        if scenarios is None:
            scenarios = self.input_dataset.scenarios
        if percentiles is None:
            percentiles = self.input_dataset.percentiles
        self.scenarios = scenarios
        self.percentiles = percentiles

        if domain is None:
            domain = self.input_dataset.wbs.data.index.tolist()

        self.domain = domain

        if objectives is None:
            objectives = ['max-abstraction', 'max-point-equality']
        if isinstance(objectives, str):
            objectives = [objectives]
        if objectives[0] != 'max-abstraction':
            raise ValueError('First objective must be "max-abstraction" currently.')
        self.objectives = objectives

        self.solver = solver
        self.raise_external_hof_error = raise_external_hof_error
        self.primary_relaxation_factor = primary_relaxation_factor
        self.reference_dataset = reference_dataset

        self.formatted_data = {}
        self.arrays = {}
        self.counts = {}

        # Key is tuple (scenario, percentile, objective) and value is Model instance
        self.models = {}

        self.output_dataset = None

    def prepare(
            self, scenario: str, percentile: int, dataset: Dataset = None,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare intermediate data tables and arrays needed for input to model class.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).
            dataset: Alternative Dataset to use in prepare (default is *input_dataset*
                attribute).

        Returns:
            Tuple of dictionaries:
                - Intermediate data tables (see model.DataPreparer.run)
                - Arrays to build model.Model (see model.ArrayBuilder.run)
                - Helper counts (e.g. numbers of different types of elements to build
                  model.Model (see model.ArrayBuilder.run)

        """
        if dataset is None:
            dataset = self.input_dataset

        # Intermediate data tables
        data_preparer = DataPreparer(
            dataset, scenario, percentile,self.domain,
        )
        formatted_data = data_preparer.run()

        # Arrays for cvxpy
        array_builder = ArrayBuilder(
            formatted_data['flows-table'], formatted_data['swabs-table'],
            formatted_data['gwabs-table'], formatted_data['subset-dataset'].graph,
            formatted_data['subdomain-dicts'], self.raise_external_hof_error,
        )
        arrays, counts = array_builder.run()

        return formatted_data, arrays, counts

    def run(self) -> OutputDataset:
        """
        Formulate and solve optimisation problem.

        Two additional tables are present in the output relative to a "normal" Dataset:
        SWABS_Changes and GWABS_Changes. These tables follow the format of SWABS_NBB and
        GWABs_NBB, but their value columns represent required impact reductions - see
        field descriptions in documentation for more details.

        Returns:
            Dataset but with abstraction impacts reflecting optimised values and
            additional tables giving the changes in impacts relative to their starting
            points.

        """
        self._check_for_required_cols()

        interim_datasets = []  # to store output datasets for each scenario+percentile
        for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
            formatted_data, arrays, counts = self.prepare(scenario, percentile)

            for objective in self.objectives:
                objective_number = self.objectives.index(objective)

                # Parent model used to help define constraints for secondary objective
                if objective_number == 0:
                    parent_model = None
                else:
                    previous_objective = self.objectives[objective_number - 1]
                    parent_model_key = (scenario, percentile, previous_objective)
                    parent_model = self.models[parent_model_key]

                # Special constraints are objectives prior to current objective and
                # auxiliary info gives the associated values
                special_constraints = self.objectives[:objective_number]
                auxiliary_info = self.get_auxiliary_info(
                    parent_model, self.primary_relaxation_factor,
                )

                # Run model - max-abstraction objective should work everytime (so no
                # error handling), but max-point-equality can be more sensitive so we
                # make a couple of tries
                model = Model(
                    objective, special_constraints, auxiliary_info, arrays, counts,
                    solver=self.solver,
                )

                if objective == 'max-abstraction':
                    model.set_problem()
                    model.run()
                else:
                    model = self._run_secondary(
                        model, auxiliary_info, objective, special_constraints, arrays,
                        counts, scenario, percentile,
                    )

                model_key = (scenario, percentile, objective)
                self.models[model_key] = model

            # Construct partial output dataset (for current scenario+percentile)
            model_key = (scenario, percentile, self.objectives[-1])
            model = self.models[model_key]

            ds = self._modify_dataset(
                scenario, percentile, model, formatted_data['swabs-table'],
                formatted_data['gwabs-table'], formatted_data['subset-dataset'],
            )
            interim_datasets.append(ds)

        # Construct final output dataset
        ds = concatenate_datasets(interim_datasets)
        self.output_dataset = self._initialise_output_dataset(ds)

        # Augment dataset with tables of SWABS and GWABS changes relative to reference
        # dataset
        self.derive_changes()

        ds = self.output_dataset

        return ds

    def _run_secondary(
            self, model: Model, auxiliary_info: AuxiliaryInfo, objective: str,
            special_constraints: List[str], arrays: Dict[str, np.ndarray],
            counts: Dict[str, int], scenario: str, percentile: int,
    ):
        """Solve for secondary objective with multiple tries in case of failure."""
        try:
            model.set_problem()
            model.run()
        except cp.error.SolverError:
            auxiliary_info.domain_total_abstraction -= 1e-1
            model = Model(
                objective, special_constraints, auxiliary_info, arrays,
                counts, solver=self.solver,
            )
            model.set_problem()
            model.run()

        if model.z.value is None:
            auxiliary_info.domain_total_abstraction -= 1e-1
            model = Model(
                objective, special_constraints, auxiliary_info, arrays,
                counts, solver=self.solver,
            )
            model.set_problem()
            model.run()

        if model.z.value is None:
            warnings.warn(
                'Unable to solve for maximum point equality objective - '
                'reverting to maximum abstraction solution.'
            )
            model_key = (scenario, percentile, 'max-abstraction')
            model = self.models[model_key]

        return model

    def derive_changes(self, table_names: List[str] = None):
        """
        Derive changes in optimised abstraction impacts relative to reference dataset.

        Args:
            table_names: Tables for which changes should be derived.

        """
        if self.reference_dataset is None:
            reference_dataset = self.input_dataset
        else:
            reference_dataset = self.reference_dataset

        if table_names is None:
            table_names = ['SWABS_NBB', 'GWABs_NBB']

        dfs = find_differences(
            reference_dataset, self.output_dataset, table_names=table_names,
            require_rows_match=False, require_columns_match=False,
        )

        # Remove value columns that are all nans (i.e. scenario/percentile combinations
        # not in optimiser outputs but possibly in input dataset)
        for table_name in table_names:
            mask = dfs[table_name].isnull().all().to_numpy()
            df = dfs[table_name].loc[:, ~mask].copy()

            if table_name == 'SWABS_NBB':
                swabs_changes = SWABS_Changes()
                swabs_changes.set_data(df)
                self.output_dataset.swabs_chg = swabs_changes
            elif table_name == 'GWABs_NBB':
                gwabs_changes = GWABS_Changes()
                gwabs_changes.set_data(df)
                self.output_dataset.gwabs_chg = gwabs_changes

    def _check_for_required_cols(self):
        """Check flow targets and optimise flag columns are present in input Dataset."""
        missing_qt_cols = []
        for scenario, percentile in itertools.product(self.scenarios, self.percentiles):
            qt_col = self.input_dataset.mt.get_qt_column(scenario, percentile)
            if qt_col not in self.input_dataset.mt.data.columns:
                missing_qt_cols.append(qt_col)

        if len(missing_qt_cols) > 0:
            raise ValueError(
                f'Missing flow target column(s) in input Dataset: {missing_qt_cols}. '
                f'Add manually or call Dataset.set_flow_targets() before initialising '
                f'Optimiser.'
            )

        opt_col = self.input_dataset.swabs.optimise_flag_column
        if opt_col not in self.input_dataset.swabs.data.columns:
            raise ValueError(
                'Missing Optimise_Flag column in SWABS table in input Dataset. Set '
                'manually or call Dataset.set_optimise_flag() before initialising '
                'Optimiser.'
            )

        opt_col = self.input_dataset.gwabs.optimise_flag_column
        if opt_col not in self.input_dataset.gwabs.data.columns:
            raise ValueError(
                'Missing Optimise_Flag column in GWABS table in input Dataset. Set '
                'manually or call Dataset.set_optimise_flag() before initialising '
                'Optimiser.'
            )

    @staticmethod
    def _modify_dataset(
            scenario: str, percentile: int, model: Model, swabs: SWABS, gwabs: GWABS,
            ds: Dataset,
    ) -> Dataset:
        """
        Return a copy of the Dataset but including optimised abstraction impacts.

        Steps involved are:

            - Copy SWABS and GWABS tables
            - Overwrite impact columns
            - Aggregate GWABS to LPP
            - Merge SWABS and GWABS into dataset tables
            - Update flows, surplus/deficits and compliance bands using calculator

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).
            model: Instance of Model containing optimised values.
            swabs: Instance of model.SWABS from data preparation.
            gwabs: Instance of model.SWABS from data preparation.
            ds: Subset Dataset from data preparation.

        Returns:
            Dataset including optimised abstraction impacts.

        """
        ds1 = deepcopy(ds)
        swabs1 = deepcopy(swabs)
        gwabs1 = deepcopy(gwabs)

        swabs1.data[swabs1.impact_column] = model.z.value[:model.n_swabs]
        gwabs1.data[gwabs1.impact_column] = model.z.value[model.n_swabs:model.n_abs]

        # Ensure that full lpp-level impact of a gwab is available for the gwabs table
        # (i.e. including impact components located outside of the domain). This can be
        # obtained based on impact components and proportions within the domain. The
        # calculator will then use the proportions to arrive at the correct impacts
        # within the domain in the master table. The output dataset's gwabs table will
        # be comparable with GWABs_NBB
        gwabs2_df = gwabs1.data.reset_index().copy()
        gwabs2_df['min_impact_number'] = gwabs2_df.groupby(gwabs1.index_name[0])[
            [gwabs1.index_name[1]]
        ].transform('min')
        gwabs2_df = gwabs2_df.loc[
            gwabs2_df[gwabs1.index_name[1]] == gwabs2_df['min_impact_number']
            ]
        gwabs2_df[gwabs1.impact_column] /= gwabs2_df[gwabs1.proportion_column]
        gwabs2_df = gwabs2_df[[gwabs1.index_name[0], gwabs1.impact_column]]
        gwabs2_df = gwabs2_df.set_index(gwabs1.index_name[0])

        # Note that we need to infill any abstractions not modelled explicitly in the
        # final tables, as they will come through as nans (whereas they should be either
        # zeros or other non-zero constants)

        # Calculate old vs new swabs differences (lpp)
        swab_col = ds1.swabs.get_value_column(scenario, percentile)
        df = pd.merge(
            ds1.swabs.data.rename(columns={swab_col: f'{swab_col}_OLD'}),
            swabs1.data[[swabs1.impact_column]].rename(columns={swabs1.impact_column: swab_col}),
            how='left', left_index=True, right_index=True,
        )
        df.loc[df[swab_col].isna(), swab_col] = df.loc[df[swab_col].isna(), f'{swab_col}_OLD']
        ds1.swabs.set_data(df.drop(columns=f'{swab_col}_OLD'))

        # Calculate old vs new gwabs differences (lpp)
        gwab_col = ds1.gwabs.get_value_column(scenario, percentile)
        df = pd.merge(
            ds1.gwabs.data.rename(columns={gwab_col: f'{gwab_col}_OLD'}),
            gwabs2_df[[gwabs1.impact_column]].rename(columns={gwabs1.impact_column: gwab_col}),
            how='left', left_index=True, right_index=True,
        )
        df.loc[df[gwab_col].isna(), gwab_col] = df.loc[df[gwab_col].isna(), f'{gwab_col}_OLD']
        ds1.gwabs.set_data(df.drop(columns=f'{gwab_col}_OLD'))

        calculator = Calculator(
            ds1, scenarios=[scenario], percentiles=[percentile],
            capping_method='cap-net-impacts',
        )
        ds2 = calculator.run()

        return ds2

    @staticmethod
    def get_auxiliary_info(
            model: Model, primary_relaxation_factor: float = None
    ) -> AuxiliaryInfo:
        """
        Extract information from model needed for running secondary objective.

        Args:
            model: Model instance.
            primary_relaxation_factor: See self.primary_relaxation_factor.

        Returns:
            Data class of key information for running secondary objective.

        """
        if model is None:
            aux = AuxiliaryInfo()
        else:
            aux = AuxiliaryInfo()
            aux.domain_total_abstraction = model.metrics.domain_total_abstraction
            aux.subdomain_proportions_fulfilled = (
                model.metrics.subdomain_proportions_fulfilled
            )
            aux.y = model.y.value
            aux.z = model.z.value

        if (model is not None) and (primary_relaxation_factor is not None):
            aux.domain_total_abstraction -= (
                aux.domain_total_abstraction * primary_relaxation_factor
            )
            aux.subdomain_proportions_fulfilled -= (
                    aux.subdomain_proportions_fulfilled * primary_relaxation_factor
            )

        return aux

    @staticmethod
    def _initialise_output_dataset(ds: Dataset) -> OutputDataset:
        ds1 = OutputDataset()
        for attr_key, attr_value in ds.__dict__.items():
            ds1.__dict__[attr_key] = attr_value
        return ds1


def calculate_mad(
        output_dataset: OutputDataset, scenario: str, percentile: int,
) -> float:
    """
    Calculate MAD equality metric on proportions fulfilled based on Optimiser output.

    Note that the calculation is currently at the licence-point-purpose level, rather
    than the licence-level calculation.

    """
    def get_arrays(df1, df2, value_col):
        # Return arrays of initial/target abstractions and proportions fulfilled
        df1 = df1.copy()
        df2 = df2.copy()
        df1['__VALUE'] = df1[value_col]
        df2['__CHANGE'] = df2[value_col]

        df3 = pd.merge(
            df1[['__VALUE']], df2[['__CHANGE']], how='left', left_index=True,
            right_index=True,
        )

        # Nans arise in proportion fulfilled where initial/target abstraction is zero
        # - exclude these cases
        df3['__PROP_FULFILLED'] = df3['__VALUE'] / (df3['__VALUE'] - df3['__CHANGE'])
        df3['__PROP_FULFILLED'] = np.where(
            ~np.isfinite(df3['__PROP_FULFILLED']), 1.0, df3['__PROP_FULFILLED']
        )
        df3 = df3.loc[~((df3['__VALUE'] == 0.0) & (df3['__CHANGE'] == 0.0))]

        return (
            (df3['__VALUE'] - df3['__CHANGE']).to_numpy(),  # initial/target
            df3['__PROP_FULFILLED'].to_numpy()  # proportions fulfilled
        )

    swabs_base, swabs_props = get_arrays(
        output_dataset.swabs.data, output_dataset.swabs_chg.data,
        output_dataset.swabs.get_value_column(scenario, percentile),
    )
    gwabs_base, gwabs_props = get_arrays(
        output_dataset.gwabs.data, output_dataset.gwabs_chg.data,
        output_dataset.gwabs.get_value_column(scenario, percentile),
    )

    # Overall proportion of target abstraction impact fulfilled at domain scale
    overall_prop = (
        (np.sum(swabs_props * swabs_base) + np.sum(gwabs_props * gwabs_base))
        / (np.sum(swabs_base) + np.sum(gwabs_base))
    )

    mad = np.mean(np.abs(np.concatenate([swabs_props, gwabs_props]) - overall_prop))

    return mad
