"""
Classes to bind core data tables with helper methods and properties.

"""
import itertools
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import pandera.pandas as pa

from .config import Constants


class Table(ABC):
    """
    Table base class.

    """
    def __init__(self, constants: Constants = None):

        #: pd.DataFrame: Table data as a pandas.DataFrame.
        self.data = None

        self._schema = None

        if constants is None:
            self._constants = Constants()
        else:
            self._constants = constants

    def load_data(
            self, input_folder: Path, set_index: bool = False, validate: bool = False,
    ):
        """
        Read table into pandas dataframe from parquet file (sets *data* attribute).

        File name is taken from *file_name* attribute.

        Args:
            input_folder: Folder containing table parquet files.
            set_index: Whether to explicitly set the dataframe index.
            validate: Whether to validate against schema.

        """
        df = pd.read_parquet(os.path.join(input_folder, self.file_name))
        self.set_data(df, set_index, validate)

    def set_data(
            self, df: pd.DataFrame, set_index: bool = False, validate: bool = False,
            filter_columns: bool = True,
    ):
        """
        Set *data* attribute.

        Args:
            df: Dataframe to set as data attribute.
            set_index: Whether to explicitly set the dataframe index.
            validate: Whether to validate against schema.
            filter_columns: If validating against schema, whether to filter to only
                columns in the schema.

        """
        if set_index:
            df = df.set_index(self.index_name)
            df = df.sort_index()

        self.data = df

        if validate is True:
            if filter_columns and (self.schema is not None):
                cols = set(self.schema.columns.keys()).intersection(
                    set(self.data.columns)
                )
                self.data = self.data[sorted(list(cols))].copy()
            self.data = self.schema.validate(self.data)  # new df in case types coerced

    def merge_data(
            self, df: pd.DataFrame, infill_values: Dict[str, float] = None,
            default_infill_value: float = 0, validate: bool = False,
    ):
        """
        Merge a dataframe into *data* attribute.

        If a column already exists in the *data* attribute, the input dataframe (df)
        will overwrite it - but only for rows given in the input df. I.e. the "old"
        values of data will be retained for rows that are not present in the input df.

        If a column does not already exist in the *data* attribute, the input df will
        provide its values. If any rows are missing from the input df, they will be
        infilled using infill_values (if supplied) or default_infill_value.

        The default for the validate argument is False because user input tables
        permitted to include only some columns.

        Args:
            df: User-input dataframe to merge into *data* attribute.
            infill_values: Dictionary whose keys are column names and whose values are
                the infilling values.
            default_infill_value: Used as a global infilling value for any columns in df
                that are not currently in *data* and if infill_values is not supplied.
            validate: Whether to validate input df.

        """
        if infill_values is None:
            infill_values = {}

        if validate:
            self.schema.validate(df)

        cols_to_replace = df.columns.tolist()
        df = df.copy()
        temp_new_cols = [f'{col}__NEW' for col in df.columns]
        df.columns = temp_new_cols

        df1 = pd.merge(
            self.data, df, how='left', left_index=True, right_index=True,
        )

        for new_col in temp_new_cols:
            old_col = new_col.replace('__NEW', '')
            if old_col in self.data.columns:
                df1.loc[~df1[new_col].isna(), old_col] = (
                    df1.loc[~df1[new_col].isna(), new_col]
                )
            else:
                df1[old_col] = df1[new_col]
                if old_col in infill_values.keys():
                    infill_value = infill_values[old_col]
                else:
                    infill_value = default_infill_value
                df1.loc[df1[old_col].isna(), old_col] = infill_value

        df1 = df1.drop(columns=temp_new_cols)

        self.set_data(df1)

        return df.shape[0], cols_to_replace

    @property
    def constants(self) -> Constants:
        """Instance of config.Constants (or similar) containing global constants."""
        return self._constants

    @property
    def file_name(self) -> str:
        """File name to use in read/write operations."""
        return f'{self.name}.parquet'

    @property
    def schema(self) -> pa.DataFrameSchema:
        """Schema for *data* dataframe attribute."""
        return self._schema

    @abstractmethod
    def set_schema(self, *args):
        """Should set *_schema* attribute."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Table name (following WRGIS names where possible)."""
        pass

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short name to use as e.g. variable name elsewhere."""
        pass

    @property
    @abstractmethod
    def index_name(self) -> Union[str, List[str]]:
        """Name of index (column(s)) for data table."""
        pass


class DataTable(Table, ABC):
    """
    Base class for WRGIS data tables that contains multiple value columns.

    """
    def __init__(self):
        super().__init__()
        self._scenarios = None
        self._percentiles = None
        self._value_types = None
        self._factor_variables = None

        self.set_factors()
        self.set_schema()

    def set_factors(
            self, scenarios: List[str] = None, percentiles: List[int] = None,
            value_types: List[str] = None,
    ):
        """
        Set key factors whose combination defines value columns in *data*.

        Args:
            scenarios: Names/abbreviations of artificial influences scenarios.
            percentiles: Flow percentiles (natural).
            value_types: Further differentiator of value columns (typically indicates
                sub vs ups aggregations at the waterbody scale).

        """
        if scenarios is None:
            scenarios = self.constants.valid_scenarios
        if percentiles is None:
            percentiles = self.constants.valid_percentiles
        if value_types is None:
            value_types = self.constants.valid_value_types

        self._scenarios = scenarios
        self._percentiles = percentiles
        self._value_types = value_types

        factor_variables = []
        if 'scenario' in self.factor_names:
            factor_variables.append(self.scenarios)
        if 'percentile' in self.factor_names:
            factor_variables.append(self.percentiles)
        if 'value_type' in self.factor_names:
            factor_variables.append(self.value_types)
        self._factor_variables = factor_variables

    def _schema_helper(
            self, auxiliary_columns: Dict, check_positive: Union[bool, dict],
            unique_index: bool = True, strict: Union[bool, str] = False,
            auxiliary_mode: bool = False,
    ) -> pa.DataFrameSchema:
        """
        Form dataframe schema applicable to *data* attribute.

        Method only works for single-index dataframes (i.e. not multi-index ones).

        auxiliary_columns should be a dictionary of the form (dummy example):
            ``{'waterbody_id': {'type': str, 'nullable': False, 'required': True}}``

        Args:
            auxiliary_columns: Details of columns in *data* that are additional to its
                *value_columns* (defined by table factors).
            check_positive: Whether all values should be positive (globally or can
                be provided column-wise as a dictionary of column name: boolean).
            unique_index: Whether index should be unique.
            strict: Whether columns not in schema should be dropped on validation
                (see pandera docs for behaviour/valid args - default of False does
                not drop columns).
            auxiliary_mode: Whether schema is being developed for a user input table
                that might be merged into *data* (and therefore may not contain all
                columns).

        Returns:
            Dataframe schema for *data* attribute.

        """
        index_column = self.index_name

        # If using for auxiliary tables (user input) then do not require all columns
        # (technically only index)
        if auxiliary_mode:
            values_required = False
            auxiliary_required = False
        else:
            values_required = True
            auxiliary_required = True

        dc = {}
        for col in self.value_columns:
            if isinstance(check_positive, dict):
                _check_positive = check_positive[col]
            else:
                _check_positive = check_positive
            if _check_positive:
                dc[col] = pa.Column(
                    float, pa.Check(lambda x: x >= 0.0), required=values_required,
                )
            else:
                dc[col] = pa.Column(float, required=values_required)

        if auxiliary_columns is not None:
            for col, details in auxiliary_columns.items():
                if 'required' in details.keys():
                    required = details['required']
                else:
                    required = auxiliary_required
                dc[col] = pa.Column(
                    details['type'], nullable=details['nullable'], required=required,
                )

        schema = pa.DataFrameSchema(
            dc, index=pa.Index(str, unique=unique_index, name=index_column),
            coerce=True, strict=strict,
        )

        return schema

    def _auxiliary_columns_helper(
            self, nullable_waterbody_column: bool = False,
    ) -> Dict:
        auxiliary_columns = {
            self.waterbody_id_column: {
                'type': str, 'nullable': nullable_waterbody_column,
            },
        }
        return auxiliary_columns

    @property
    def waterbody_id_column(self) -> str:
        """Waterbody ID column name."""
        return self.constants.waterbody_id_column

    @property
    def scenarios(self) -> List[str] | None:
        """Names/abbreviations of artificial influences scenarios."""
        return self._scenarios

    @property
    def percentiles(self) -> List[int] | None:
        """Flow percentiles (natural)."""
        return self._percentiles

    @property
    def value_types(self) -> List[str] | None:
        """Value type indicators (typically indicating sub and/or ups aggregations.)"""
        return self._value_types

    @property
    def factor_variables(self) -> List[List]:
        """Values of each factor variable (list of lists)."""
        return self._factor_variables

    @property
    def value_columns(self) -> List[str]:
        """Names of value columns in *data* attribute."""
        value_columns = []
        for variables in itertools.product(*self.factor_variables):
            value_columns.append(self.get_value_column(*variables))
        return value_columns

    @abstractmethod
    def get_value_column(self, *args) -> str:
        """
        Return column name for specific factor variables.

        Used to get the column name for, say, groundwater abstraction impacts under the
        fully licensed (FL) scenario at Q95 from the GWABs_NBB table.

        """
        pass

    @property
    @abstractmethod
    def factor_names(self) -> List[str]:
        """
        Factor (names) relevant to concrete table class.

        Typically some combination of 'scenario', 'percentile' and 'value_type'.

        """
        pass

    @property
    @abstractmethod
    def variable_abb(self) -> str:
        """Abbreviation used for variable in column names in *data*."""
        pass


class GWABs_NBB(DataTable):
    """
    Table for groundwater abstraction impacts at the licence-point-purpose level.

    """
    def set_schema(
            self, unique_index: bool = True, check_positive: bool = False,
            nullable_waterbody_column: bool = False, auxiliary_mode: bool = False,
    ):
        auxiliary_columns = self._auxiliary_columns_helper(nullable_waterbody_column)

        for wb_number, wb_proportion in zip(
                self.impacted_waterbody_columns, self.impact_proportion_columns
        ):
            auxiliary_columns[wb_number] = {'type': str, 'nullable': True}
            auxiliary_columns[wb_proportion] = {'type': float, 'nullable': False}

        auxiliary_columns[self.optimise_flag_column] = {
            'type': bool, 'nullable': False, 'required': False,
        }
        auxiliary_columns[self.purpose_column] = {'type': str, 'nullable': False}
        auxiliary_columns[self.licence_expiry_column] = {'type': str, 'nullable': True}

        # Metadata columns that are not used in the code (and do not have a property to
        # indicate the column name)
        auxiliary_columns['LICNUMBER'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['SITENAME'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['LICHOLDER'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['FLPTPANQM3'] = {
            'type': float, 'nullable': True, 'required': False,
        }
        auxiliary_columns['RAPTPANQM3'] = {
            'type': float, 'nullable': True, 'required': False,
        }
        auxiliary_columns['GWPROPCONS'] = {
            'type': float, 'nullable': True, 'required': False,
        }
        auxiliary_columns['IMPFAC'] = {
            'type': float, 'nullable': True, 'required': False,
        }

        schema = self._schema_helper(
            auxiliary_columns, check_positive, unique_index=unique_index,
            auxiliary_mode=auxiliary_mode,
        )
        self._schema = schema

    def get_value_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return f'{self.variable_abb}Q{percentile}{scenario}WR'

    @property
    def name(self) -> str:
        return 'GWABs_NBB'

    @property
    def short_name(self) -> str:
        return 'gwabs'

    @property
    def index_name(self) -> str:
        return 'UNIQUEID'

    @property
    def factor_names(self) -> List[str]:
        return ['scenario', 'percentile']

    @property
    def variable_abb(self) -> str:
        return self.constants.gwabs_abb

    @property
    def impacted_waterbody_columns(self) -> List[str]:
        """Names of columns indicating waterbodies impacted by each abstraction."""
        return ['WB_1ST', 'WB_2ND', 'WB_3RD', 'WB_4TH', 'WB_5TH']

    @property
    def impact_proportion_columns(self) -> List[str]:
        """
        Names of columns indicating proportion of impact felt by each impacted waterbody.

        Total impact of abstraction is given in the *value_columns*. Together with
        *impacted_waterbody_columns*, the columns in this attribute define how the
        total impact is split between waterbodies.

        """
        return [f'{col}_PRO' for col in self.impacted_waterbody_columns]

    @property
    def optimise_flag_column(self) -> str:
        """Name of column indicating whether a row should be included in optimisation."""
        return self.constants.optimise_flag_column

    @property
    def purpose_column(self) -> str:
        """Name of column indicating abstraction purpose code."""
        return 'PURPCODE'

    @property
    def licence_expiry_column(self) -> str:
        """Name of column with date/flag indicating licence expiry date."""
        return 'LICN_EXPD'

    @property
    def infill_mapping(self) -> Dict:
        """
        Values to use for infilling missing entries in particular columns.

        Dictionary key is the infill value and dictionary value is a list of columns
        to which the infill value should be applied.

        """
        return {0: self.impact_proportion_columns, '': self.impacted_waterbody_columns}


class SWABS_NBB(DataTable):
    """
    Table for surface water abstraction impacts at the licence-point-purpose level.

    """
    def set_schema(
            self, unique_index: bool = True, check_positive: bool = False,
            nullable_waterbody_column: bool = False, auxiliary_mode: bool = False,
    ):
        auxiliary_columns = self._auxiliary_columns_helper(nullable_waterbody_column)
        auxiliary_columns[self.hof_value_column] = {'type': float, 'nullable': False}
        auxiliary_columns[self.hof_waterbody_column] = {'type': str, 'nullable': False}
        auxiliary_columns[self.optimise_flag_column] = {
            'type': bool, 'nullable': False, 'required': False,
        }
        auxiliary_columns[self.purpose_column] = {'type': str, 'nullable': False}
        auxiliary_columns[self.reservoir_flag_column] = {'type': int, 'nullable': False}
        auxiliary_columns[self.ldmu_flag_column] = {'type': int, 'nullable': False}
        for lake_col in self.lake_flag_columns:
            auxiliary_columns[lake_col] = {'type': int, 'nullable': False}
        auxiliary_columns[self.licence_expiry_column] = {'type': str, 'nullable': True}

        # Metadata columns that are not used in the code (and do not have a property to
        # indicate the column name)
        auxiliary_columns['LICNUMBER'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['SITENAME'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['LICHOLDER'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['FLPTPANQM3'] = {
            'type': float, 'nullable': True, 'required': False,
        }
        auxiliary_columns['RAPTPANQM3'] = {
            'type': float, 'nullable': True, 'required': False,
        }
        auxiliary_columns['SWPROPCONS'] = {
            'type': float, 'nullable': True, 'required': False,
        }

        schema = self._schema_helper(
            auxiliary_columns, check_positive, unique_index=unique_index,
            auxiliary_mode=auxiliary_mode,
        )
        self._schema = schema

    def get_value_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return f'{self.variable_abb}Q{percentile}{scenario}WR'

    @property
    def name(self) -> str:
        return 'SWABS_NBB'

    @property
    def short_name(self) -> str:
        return 'swabs'

    @property
    def index_name(self) -> str:
        return 'UNIQUEID'

    @property
    def factor_names(self) -> List[str]:
        return ['scenario', 'percentile']

    @property
    def variable_abb(self) -> str:
        return self.constants.swabs_abb

    @property
    def hof_value_column(self) -> str:
        """Name of column indicating HOF flow conditions."""
        return 'HOFMLD'

    @property
    def hof_waterbody_column(self) -> str:
        """Name of column indicating waterbodies (IDs) that define HOF conditions."""
        return 'HOFWBID'

    @property
    def optimise_flag_column(self) -> str:
        """Name of column indicating whether a row should be included in optimisation."""
        return self.constants.optimise_flag_column

    @property
    def purpose_column(self) -> str:
        """Name of column indicating abstraction purpose code."""
        return 'PURPCODE'

    @property
    def reservoir_flag_column(self) -> str:
        """Name of column indicating whether an abstraction is reservoir-related."""
        return 'RESRVRFLAG'

    @property
    def ldmu_flag_column(self) -> str:
        """Name of column indicating whether an abstraction is LDMU-related."""
        return 'SW_LDMU_NO'

    @property
    def lake_flag_columns(self) -> List[str]:
        """Name of columns indicating whether an abstraction is lake-related."""
        return [f'SW_LAKE{i}' for i in range(1, 6)]

    @property
    def licence_expiry_column(self) -> str:
        """Name of column with date/flag indicating licence expiry date."""
        return 'LICN_EXPD'

    @property
    def infill_mapping(self) -> Dict:
        """
        Values to use for infilling missing entries in particular columns.

        Dictionary key is the infill value and dictionary value is a list of columns
        to which the infill value should be applied.

        """
        return {
            0: [self.hof_value_column, self.ldmu_flag_column],
            '': [self.hof_waterbody_column],
        }


class Discharges_NBB(DataTable):
    """
    Table for groundwater abstraction impacts at the point level.

    """
    def set_schema(
            self, unique_index: bool = True, check_positive: bool = True,
            nullable_waterbody_column: bool = False, auxiliary_mode: bool = False,
    ):
        auxiliary_columns = self._auxiliary_columns_helper(nullable_waterbody_column)

        # Metadata columns that are not used in the code (and do not have a property to
        # indicate the column name)
        auxiliary_columns['DISNUMBER'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['SITENAME'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['CONSHOLDER'] = {
            'type': str, 'nullable': True, 'required': False,
        }

        schema = self._schema_helper(
            auxiliary_columns, check_positive, unique_index=unique_index,
            auxiliary_mode=auxiliary_mode,
        )
        self._schema = schema

    def get_value_column(self, scenario: str) -> str:
        """
        Return name of value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.

        Returns:
             Name of value column.

        """
        return f'{self.variable_abb}{scenario}'

    @property
    def name(self) -> str:
        return 'Discharges_NBB'

    @property
    def short_name(self) -> str:
        return 'dis'

    @property
    def index_name(self) -> str:
        return 'UNID'

    @property
    def factor_names(self) -> List[str]:
        return ['scenario']

    @property
    def variable_abb(self) -> str:
        return self.constants.dis_abb


class SupResGW_NBB(DataTable):
    """
    Table for complex impacts at the point level.

    """
    def set_schema(
            self, unique_index: bool = True, check_positive: bool = False,
            nullable_waterbody_column: bool = False, auxiliary_mode: bool = False,
    ):
        auxiliary_columns = self._auxiliary_columns_helper(nullable_waterbody_column)
        auxiliary_columns[self.purpose_column] = {'type': str, 'nullable': False}

        # Metadata columns that are not used in the code (and do not have a property to
        # indicate the column name)
        auxiliary_columns['NAME'] = {'type': str, 'nullable': True, 'required': False}
        auxiliary_columns['OPERATOR'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['TYPE_SUPRESGW'] = {
            'type': str, 'nullable': True, 'required': False,
        }
        auxiliary_columns['PURPOSE'] = {
            'type': str, 'nullable': True, 'required': False,
        }

        schema = self._schema_helper(
            auxiliary_columns, check_positive, unique_index=unique_index,
            auxiliary_mode=auxiliary_mode,
        )
        self._schema = schema

    def get_value_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return f'{self.variable_abb}{scenario}Q{percentile}'

    def get_upper_limit_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of column giving upper limit of allow compensation flow effects.

        This is only relevant if complex impacts are being considered in the Optimiser -
        specifically complex impacts where increases in reservoir compensation flow are
        being explored. Otherwise, this column is not required in the tool

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of column giving upper limit on potential reservoir compensation flow
             increase effect.

        """
        return f'{self.variable_abb}{scenario}Q{percentile}_MAX_INCREASE'

    @property
    def name(self) -> str:
        return 'SupResGW_NBB'

    @property
    def short_name(self) -> str:
        return 'sup'

    @property
    def index_name(self) -> str:
        return 'UNID'

    @property
    def factor_names(self) -> List[str]:
        return ['scenario', 'percentile']

    @property
    def variable_abb(self) -> str:
        return self.constants.sup_abb

    @property
    def purpose_column(self) -> str:
        """Name of column indicating impact purpose."""
        return 'PURPOSE'

    @property
    def purposes_to_exclude(self) -> List[str]:
        """Impact purposes that should be excluded from aggregation/summation."""
        return ['Trib Summary Impacts', 'LDMU']

    @property
    def optimise_flag_column(self) -> str:
        """Column indicating whether/how a row should be included in optimisation."""
        return self.constants.optimise_flag_column


class QNaturalFlows_NBB(DataTable):
    """
    Table for natural flows at the waterbody level.

    """
    def set_schema(
            self, unique_index: bool = True, check_positive: bool = None,
            auxiliary_mode: bool = False,
    ):
        check_positive = {}
        for col in self.value_columns:
            if col.endswith(self.constants.sub_abb):
                check_positive[col] = False
            else:
                check_positive[col] = True
        self._schema = self._schema_helper(
            None, check_positive, unique_index=unique_index,
            auxiliary_mode=auxiliary_mode,
        )

    def get_value_column(self, percentile: int, value_type: str) -> str:
        """
        Return name of value column.

        Args:
            percentile: Flow percentile (natural).
            value_type: Indicator of aggregation type - either 'sub' or 'ups'.

        Returns:
             Name of value column.

        """
        return f'{self.variable_abb}{percentile}{value_type}'

    @property
    def name(self) -> str:
        return 'QNaturalFlows_NBB'

    @property
    def short_name(self) -> str:
        return 'qnat'

    @property
    def index_name(self) -> str:
        return self.waterbody_id_column

    @property
    def factor_names(self) -> List[str]:
        return ['percentile', 'value_type']

    @property
    def variable_abb(self) -> str:
        return self.constants.qnat_abb


class IntegratedWBs_NBB(Table):
    """
    Table for basic waterbody metadata.

    """
    def set_schema(self, unique_index: bool = True, strict='filter'):
        self._schema = pa.DataFrameSchema(
            {
                self.downstream_waterbody_column: pa.Column(str),
                self.waterbody_type_column: pa.Column(str),
                self.basin_column: pa.Column(str),
            },
            index=pa.Index(
                str, unique=unique_index, name=self.index_name,
            ),
            coerce=True, strict=strict,
        )

    @property
    def name(self) -> str:
        return 'IntegratedWBs_NBB'

    @property
    def short_name(self) -> str:
        return 'wbs'

    @property
    def index_name(self) -> str:
        return self.constants.waterbody_id_column

    @property
    def downstream_waterbody_column(self) -> str:
        """Name of column indicating waterbody (ID) immediately downstream."""
        return 'DSTREAM_WB'

    @property
    def waterbody_type_column(self) -> str:
        """Name of column indicating waterbody type."""
        return 'Type_IWB'

    @property
    def basin_column(self) -> str:
        """Name of column indicating river basin district of waterbody."""
        return 'RBD_NAME'

    @property
    def unassessed_types(self) -> List[str]:
        """Types of waterbodies that are not assessed for compliance."""
        return ['Seaward Transitional', 'Saline Lagoon']


class AbsSensBands_NBB(Table):
    """
    Table for abstraction sensitivity bands (ASBs) per waterbody.

    """
    def set_schema(self, unique_index: bool = True):
        self._schema = pa.DataFrameSchema(
            {self.asb_column: pa.Column(int)},
            index=pa.Index(str, unique=unique_index, name=self.index_name),
            coerce=True,
        )

    @property
    def name(self) -> str:
        return 'AbsSensBands_NBB'

    @property
    def short_name(self) -> str:
        return 'asbs'

    @property
    def index_name(self) -> str:
        return self.constants.waterbody_id_column

    @property
    def asb_column(self) -> str:
        """Name of column containing abstraction sensitivity bands."""
        return 'ASBFinal'


class ASBPercentages(Table):
    """
    Table defining permitted deviations from natural flow by ASB.

    """
    def set_schema(self):
        self._schema = pa.DataFrameSchema(
            {self.percent_column: pa.Column(
                float, pa.Check(lambda x: (x >= 0.0) & (x <= 1.0))
            )},
            index=pa.MultiIndex([
                pa.Index(str, name=self.index_name[0]),
                pa.Index(int, name=self.index_name[1]),
            ], unique=self.index_name),
            coerce=True,
        )

    @staticmethod
    def percentile_label(percentile: int) -> str:
        """Flow percentile label helper (for index of *data*)."""
        return f'Q{percentile}'

    @property
    def name(self) -> str:
        return 'ASBPercentages'

    @property
    def short_name(self) -> str:
        return 'asb_percs'

    @property
    def index_name(self) -> List[str]:
        return ['QFLOW', 'ASB']

    @property
    def percent_column(self) -> str:
        """Permitted deviation (as a **factor** in WRGIS, not actually a percentage)."""
        return 'PERCENT_'

    @property
    def percentiles(self) -> List[int]:
        """Percentiles for which ASBs are available in *data*."""
        return self.constants.valid_percentiles

    @property
    def percentile_labels(self) -> List[str]:
        """Labels of percentiles in index of *data*."""
        percentile_labels = []
        for percentile in self.percentiles:
            percentile_labels.append(self.percentile_label(percentile))
        return percentile_labels

    @property
    def asb_values(self) -> List[int]:
        """Valid ASB values."""
        return [1, 2, 3, 11, 12, 13]


class EFI(DataTable):
    """
    Table for environmental flow indicator (EFI) per waterbody.

    """
    def set_schema(
            self, unique_index: bool = True, check_positive: bool = True,
            auxiliary_mode: bool = False,
    ):
        self._schema = self._schema_helper(
            None, check_positive, unique_index=unique_index,
            auxiliary_mode=auxiliary_mode,
        )

    def get_value_column(self, percentile) -> str:
        """
        Return name of value column.

        Args:
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return f'{self.variable_abb}Q{percentile}'

    @property
    def name(self) -> str:
        return 'EFI'

    @property
    def short_name(self) -> str:
        return 'efi'

    @property
    def index_name(self) -> str:
        return self.waterbody_id_column

    @property
    def factor_names(self) -> List[str]:
        return ['percentile']

    @property
    def variable_abb(self) -> str:
        return self.constants.efi_abb


class Master(Table):
    """
    Master table to bring together all key columns at the waterbody level.

    Key columns here are those needed to calculate the water balance and assess the
    compliance of scenario flows.

    See get_value_column method docstring as a guide to the arguments for the other
    methods that get value column names for specific variables.

    """
    def __init__(self, scenarios: List[str] = None, percentiles: List[int] = None):
        super().__init__()
        self._scenarios = scenarios
        self._percentiles = percentiles

    def set_schema(self) -> pa.DataFrameSchema:
        # TODO: Implement schema
        pass

    def get_value_column(
            self, variable_abb: str, scenario: str = None, percentile: int = None,
            value_type: str = None,
    ) -> str:
        """
        Return name of value column.

        See also get_value_columns and other per variable methods (e.g.
        get_gwabs_column).

        Args:
            variable_abb: Variable abbreviation.
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).
            value_type: Indicator of aggregation type ('sub' or 'ups').

        Returns:
            Value column name.

        """
        q = 'Q'
        if 'scenario' not in self.factor_names[variable_abb]:
            scenario = ''
        if 'percentile' not in self.factor_names[variable_abb]:
            percentile = ''
            q = ''
        if 'value_type' not in self.factor_names[variable_abb]:
            value_type = ''

        if variable_abb == self.constants.qnat_abb:
            col = f'{variable_abb}{percentile}{value_type}'
        else:
            col = f'{variable_abb}{scenario}{q}{percentile}{value_type}'

        return col

    def get_value_columns(self, variable_abb: str) -> List[str]:
        """
        Return all value column names for a given variable.

        Args:
            variable_abb: Variable abbreviation used in value column names - see
                config.Constants.

        Returns:
             Names of value column.

        """
        value_columns = []

        for scenario, percentile, value_type in itertools.product(
                self.scenarios, self.percentiles, self.value_types
        ):
            col = self.get_value_column(
                variable_abb, scenario, percentile, value_type
            )

            # To avoid duplicating qn, efi and discharge columns
            if col not in value_columns:
                value_columns.append(col)

        return value_columns

    def get_qnat_column(self, percentile: int, value_type: str) -> str:
        """
        Return name of natural flow value column.

        Args:
            percentile: Flow percentile (natural).
            value_type: Indicator of aggregation type - either 'sub' or 'ups'.

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.qnat_abb, percentile=percentile, value_type=value_type,
        )

    def get_gwabs_column(self, scenario: str, percentile: int, value_type: str) -> str:
        """
        Return name of groundwater abstraction impact value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).
            value_type: Indicator of aggregation type - either 'sub' or 'ups'.

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.gwabs_abb, scenario, percentile, value_type
        )

    def get_swabs_column(self, scenario: str, percentile: int, value_type: str) -> str:
        """
        Return name of surface water abstraction impact value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).
            value_type: Indicator of aggregation type - either 'sub' or 'ups'.

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.swabs_abb, scenario, percentile, value_type
        )

    def get_dis_column(self, scenario: str, value_type: str) -> str:
        """
        Return name of discharge impact value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            value_type: Indicator of aggregation type - either 'sub' or 'ups'.

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.dis_abb, scenario, value_type=value_type
        )

    def get_sup_column(self, scenario: str, percentile: int, value_type: str) -> str:
        """
        Return name of complex impact value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).
            value_type: Indicator of aggregation type - either 'sub' or 'ups'.

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.sup_abb, scenario, percentile, value_type
        )

    def get_efi_column(self, percentile: int) -> str:
        """
        Return name of environmental flow indicator (EFI) value column.

        Args:
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.efi_abb, percentile=percentile
        )

    def get_scen_column(self, scenario: str, percentile: int, value_type: str) -> str:
        """
        Return name of scenario flow value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).
            value_type: Indicator of aggregation type - either 'sub' or 'ups'.

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.scen_abb, scenario, percentile, value_type
        )

    def get_sd_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of surplus/deficit (relative to EFI) value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.sd_abb, scenario, percentile
        )

    def get_comp_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of compliance band value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.comp_abb, scenario, percentile
        )

    def get_qt_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of flow target value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.qt_abb, scenario, percentile
        )  # append ups?

    def get_sdt_column(self, scenario: str, percentile: int) -> str:
        """
        Return name of surplus/deficit relative to flow target value column.

        Args:
            scenario: Name/abbreviation of artificial influences scenario.
            percentile: Flow percentile (natural).

        Returns:
             Name of value column.

        """
        return self.get_value_column(
            self.constants.sdt_abb, scenario, percentile
        )

    @property
    def name(self) -> str:
        return 'Master'

    @property
    def short_name(self) -> str:
        return 'mt'

    @property
    def index_name(self) -> str:
        return self.constants.waterbody_id_column

    @property
    def variable_abbs(self) -> List[str]:
        return [
            self.constants.qnat_abb, self.constants.gwabs_abb, self.constants.swabs_abb,
            self.constants.dis_abb, self.constants.sup_abb, self.constants.efi_abb,
            self.constants.scen_abb, self.constants.sd_abb, self.constants.comp_abb,
            self.constants.qt_abb, self.constants.sdt_abb,
        ]

    @property
    def scenarios(self) -> List[str]:
        return self._scenarios

    @property
    def percentiles(self) -> List[int]:
        return self._percentiles

    @property
    def value_types(self) -> List[str]:
        return self.constants.valid_value_types

    @property
    def value_columns(self) -> List[str]:
        value_columns = []

        for variable_abb, scenario, percentile, value_type in itertools.product(
                self.variable_abbs, self.scenarios, self.percentiles, self.value_types
        ):
            col = self.get_value_column(
                variable_abb, scenario, percentile, value_type
            )

            # To avoid duplicating qn, efi and discharge columns
            if col not in value_columns:
                value_columns.append(col)

        return value_columns

    @property
    def factor_names(self) -> Dict[str, List[str]]:
        return {
            self.constants.qnat_abb: ['percentile', 'value_type'],
            self.constants.gwabs_abb: ['scenario', 'percentile', 'value_type'],
            self.constants.swabs_abb: ['scenario', 'percentile', 'value_type'],
            self.constants.dis_abb: ['scenario', 'value_type'],
            self.constants.sup_abb: ['scenario', 'percentile', 'value_type'],
            self.constants.efi_abb: ['percentile'],
            self.constants.scen_abb: ['scenario', 'percentile', 'value_type'],
            self.constants.sd_abb: ['scenario', 'percentile'],
            self.constants.comp_abb: ['scenario', 'percentile'],
            self.constants.qt_abb: ['scenario', 'percentile'],
            self.constants.sdt_abb: ['scenario', 'percentile'],
        }
