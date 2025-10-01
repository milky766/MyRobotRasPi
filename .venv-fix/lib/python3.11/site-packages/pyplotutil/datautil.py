"""Data Handling and Manipulation Module.

This module provides classes for managing and manipulating tabular data, with functionalities to
load data from various sources, group data structure by specified tags, and access columns or rows
with intuitive syntax. The primary classes, `Data`, and `TaggedData`, facilitate working with
tabular data in pandas DataFrame while allowing access to specific features like data grouping,
dynamic attribute setting, and easy retrieval of parameter values.

Classes
-------
BaseData : Abstract base class providing the core attributes and methods for data handling.
    Defines basic properties for data path and DataFrame storage.

Data : Extends BaseData to represent a single tabular data.
    Provides methods to access columns and retrieve specific parameters.

TaggedData : Extends BaseData to handle grouped data based on a specified tag column.
    Allows grouping data by a tag and accessing each group as a separate `Data` object.

Examples
--------
Basic usage:
    >>> data = Data("data.csv")
    >>> print(data.param("column_name"))

Tagged data usage:
    >>> tagged_data = TaggedData("data.csv", tag="tag")
    >>> data = tagged_data.get("specific_tag")
    >>> print(group.param("column_name"))

This module is designed to streamline operations with tabular data in data analysis, data plotting,
and other applications requiring structured data handling.

"""

from __future__ import annotations

from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TextIO, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable, ItemsView, Iterator, KeysView, Sequence

    from pandas.io.parsers.readers import UsecolsArgType

FilePath: TypeAlias = str | Path
DataSourceType: TypeAlias = FilePath | StringIO | pd.DataFrame | pd.Series
NumericType: TypeAlias = int | float | complex | np.number
NumericTypeVar = TypeVar("NumericTypeVar", bound=NumericType)
Unknown: TypeAlias = Any


class _NoDefault(Enum):
    """Enum to represent the absence of a default value in method parameters."""

    no_default = auto()


no_default: Final = _NoDefault.no_default
NoDefault: TypeAlias = Literal[_NoDefault.no_default]


class BaseData:
    """Base class for data handling and manipulation.

    This class provides functionalities for setting and retrieving the path and the main DataFrame
    associated with the data.

    """

    _datapath: Path
    _dataframe: pd.DataFrame

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        sep: str,
        header: int | Sequence[int] | Literal["infer"] | None,
        names: Sequence[Hashable] | None,
        usecols: UsecolsArgType,
        nrows: int | None,
        comment: str | None,
    ) -> None:
        """Initialize the BaseData object with the provided data source.

        Parameters
        ----------
        data_source : str | Path | StringIO | pd.DataFrame | pd.Series
            The data source.
        sep : str
            Delimiter for CSV data.
        header : int or Sequence[int] or Literal["infer"], optional
            Row(s) to use as the column names.
        names : Sequence[Hashable], optional
            Column names to use.
        usecols : Sequence[Hashable] or range, optional
            Columns to read from the data source.
        nrows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        if isinstance(data_source, pd.DataFrame):
            self._set_dataframe(data_source)
        elif isinstance(data_source, pd.Series):
            self._set_dataframe(data_source.to_frame())
        elif isinstance(data_source, StringIO | FilePath):
            self._set_dataframe(
                self.read_csv(
                    data_source,
                    sep=sep,
                    header=header,
                    names=names,
                    usecols=usecols,
                    nrows=nrows,
                    comment=comment,
                ),
            )
            if isinstance(data_source, FilePath):
                self._set_datapath(data_source)
        else:
            msg = f"Unsupported data source type: {type(data_source)}"
            raise TypeError(msg)

    @staticmethod
    def read_commented_column_names(file_or_buffer: FilePath | StringIO, *, sep: str, comment: str) -> list[str] | None:
        """Return a list of column names extracted from commented lines in the file.

        Parameters
        ----------
        file_or_buffer : str | Path | StringIO
            The file or buffer containing the data.
        sep : str
            Delimiter for the data.
        comment : str
            Character indicating commented lines.

        Returns
        -------
        list of str or None
            List of column names if found; otherwise, None.

        """

        def last_commented_header(buffer: TextIO, comment: str) -> str:
            header = ""
            for line in buffer:
                if line.startswith(comment):
                    header = line
                else:
                    break
            return header

        if isinstance(file_or_buffer, FilePath):
            with Path(file_or_buffer).open() as f:
                header = last_commented_header(f, comment)
        else:
            header = last_commented_header(file_or_buffer, comment)
        if len(header) > 0:
            return header[1:].strip().split(sep)
        return None

    @staticmethod
    def read_csv(
        file_or_buffer: FilePath | StringIO,
        *,
        sep: str = ",",
        header: int | Sequence[int] | Literal["infer"] | None,
        names: Sequence[Hashable] | None,
        usecols: UsecolsArgType,
        nrows: int | None,
        comment: str | None,
    ) -> pd.DataFrame:
        """Return a pandas DataFrame loaded from a file or string buffer.

        Parameters
        ----------
        file_or_buffer : str | Path | StringIO
            The file or buffer to read from.
        sep : str, optional
            Delimiter for the data.
        header : int or Sequence[int] or Literal["infer"], optional
            Row(s) to use as the column names.
        names : Sequence[Hashable], optional
            Column names to use.
        usecols : Sequence[Hashable] or range, optional
            Columns to read from the data source.
        nrows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame.

        """
        if comment is not None and names is None:
            names = BaseData.read_commented_column_names(file_or_buffer, sep=sep, comment=comment)
        if isinstance(file_or_buffer, StringIO):
            file_or_buffer.seek(0)
        return pd.read_csv(
            file_or_buffer,
            sep=sep,
            header=header,
            names=names,
            usecols=usecols,
            nrows=nrows,
            comment=comment,
            iterator=False,
            chunksize=None,
        )

    def _set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set the DataFrame associated with the data object.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to associate with the data.

        """
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        """Retrieve the raw DataFrame associated with the data.

        Returns
        -------
        pd.DataFrame
            The DataFrame associated with the data.

        """
        return self._dataframe

    @property
    def df(self) -> pd.DataFrame:
        """Alias for `dataframe` attribute.

        Returns
        -------
        pd.DataFrame
            The DataFrame associated with the data.

        """
        return self.dataframe

    def is_loaded_from_file(self) -> bool:
        """Check if the Data object was loaded from a file.

        Returns
        -------
        bool
            True if loaded from a file; otherwise, False.

        """
        try:
            _ = self._datapath
        except AttributeError:
            return False
        return True

    def _set_datapath(self, datapath: str | Path) -> None:
        """Set the path to the data file.

        Parameters
        ----------
        datapath : str or Path
            Path to the data file.

        """
        self._datapath = Path(datapath)

    @property
    def datapath(self) -> Path:
        """Retrieve the path to the data file.

        Returns
        -------
        Path
            Path to the data file.

        """
        try:
            return self._datapath
        except AttributeError as e:
            msg = "Data object may not be loaded from a file."
            raise AttributeError(msg) from e

    @property
    def datadir(self) -> Path:
        """Retrieve the directory of the data file.

        Returns
        -------
        Path
            Directory of the data file.

        """
        return self.datapath.parent

    def __str__(self) -> str:
        """Return a string of the associated DataFrame.

        Returns
        -------
        str
            String representation of the associated DataFrame object.

        """
        return str(self.dataframe)

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns
        -------
        str
            String representation of the object.

        """
        if self.is_loaded_from_file():
            return f"{self.__class__.__name__}({self.datapath})"
        return f"{self.__class__.__name__}({self.dataframe})"


class Data(BaseData):
    """A class representing tabular data loaded from a source.

    This class provides methods for easy data access and specific data parameter extraction. When
    the source data contains column names, this class provides attributes for access the data column
    with its name.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the tabular data.
    datapath : Path
        The path to the data file.
    datadir : Path
        The directory of the data file

    Examples
    --------
    Initialization of the data object and access for column data with its name.

    >>> import pandas as pd
    >>> data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [5, 5, 5]}))
    >>> data.a
    0    1
    1    2
    2    3
    Name: a, dtype: int64
    >>> data.param("c")
    np.int64(5)

    """

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        sep: str = ",",
        header: int | Sequence[int] | Literal["infer"] | None = "infer",
        names: Sequence[Hashable] | None = None,
        usecols: UsecolsArgType = None,
        nrows: int | None = None,
        comment: str | None = None,
    ) -> None:
        """Initialize the BaseData object with the provided data source.

        Parameters
        ----------
        data_source : str | Path | StringIO | pd.DataFrame | pd.Series
            The data source.
        sep : str
            Delimiter for CSV data.
        header : int or Sequence[int] or Literal["infer"], optional
            Row(s) to use as the column names.
        names : Sequence[Hashable], optional
            Column names to use.
        usecols : Sequence[Hashable] or range, optional
            Columns to read from the data source.
        nrows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        super().__init__(
            data_source,
            sep=sep,
            header=header,
            names=names,
            usecols=usecols,
            nrows=nrows,
            comment=comment,
        )

    def __getitem__(self, key: Unknown) -> pd.Series | pd.DataFrame:
        """Access a specific column(s).

        Parameters
        ----------
        key : str or int or Sequence of str or int
            Column name or column index.

        Returns
        -------
        pd.Series or pd.DataFrame
            Series or frame of the specified column(s).

        """
        return self.dataframe.__getitem__(key)

    def __len__(self) -> int:
        """Return the number of rows in the `Data` object.

        Returns
        -------
        int
            Number of rows in the `Data` object.

        """
        return len(self.dataframe)

    def __getattr__(self, name: str) -> Unknown:
        """Access DataFrame attributes not explicitly defined in Data.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Unknown
            The attribute from the DataFrame.

        """
        if name in ("datapath", "datadir"):
            return self.__getattribute__(name)
        return getattr(self.dataframe, name)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Return an iterator over the Data objects.

        Returns
        -------
        Iterator[np.ndarray]
            An iterator over the Data objects.

        """
        return iter(self.dataframe.to_numpy())

    def split_by_row(self, row_index: int, *, reset_index: bool = True) -> tuple[Data, Data]:
        """Split the Data object into two parts at a specified row index.

        Parameters
        ----------
        row_index : int
            The index at which to split the data object. Rows from the start up to
            `row_index` will go to the first split, and rows from `row_index` to
            the end will go to the second split.
        reset_index : bool, optional
            Whether to reset the index of the second split data, by default True.

        Returns
        -------
        tuple[Data, Data]
            A tuple containing two Data objects. The first contains rows from the
            start to `row_index`, and the second contains rows from `row_index`
            to the end, with the index reset if `reset_index` is True.

        """
        df1 = self.dataframe.iloc[:row_index]
        df2 = self.dataframe.iloc[row_index:]
        if reset_index:
            df2 = df2.reset_index(drop=True)
        return Data(df1), Data(df2)

    @overload
    def param(self, key: int | str) -> NumericType: ...

    @overload
    def param(self, key: Sequence) -> pd.Series: ...

    def param(self, key):
        """Retrieve specific parameter(s) for column(s).

        Parameters
        ----------
        key : int or str or Sequence of int or str
            The column(s) for which to retrieve the parameter.

        Returns
        -------
        Numeric type or pd.Series
            Retrieved parameter value(s).

        """
        row = self.dataframe.loc[0, key]
        if isinstance(row, pd.Series | pd.DataFrame):
            row = pd.to_numeric(row)
        return row


class TaggedData(BaseData):
    """A class for handling data grouped by a specified tag.

    This class provides methods to load and access data grouped by a tag, enabling
    easy access to each group's data as individual `Data` objects.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the tabular data with tags.
    datadict : dict of str to Data
        A dictionary mapping each tag value to a corresponding `Data` object.

    """

    _datadict: dict[str, Data]
    _tag: Unknown

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        sep: str = ",",
        header: int | Sequence[int] | Literal["infer"] | None = "infer",
        names: Sequence[Hashable] | None = None,
        usecols: UsecolsArgType = None,
        nrows: int | None = None,
        comment: str | None = None,
        tag: Unknown = "tag",
    ) -> None:
        """Initialize the BaseData object with the provided data source.

        Parameters
        ----------
        data_source : str | Path | StringIO | pd.DataFrame | pd.Series
            The data source.
        sep : str
            Delimiter for CSV data.
        header : int or Sequence[int] or Literal["infer"], optional
            Row(s) to use as the column names.
        names : Sequence[Hashable], optional
            Column names to use.
        usecols : Sequence[Hashable] or range, optional
            Columns to read from the data source.
        nrows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.
        tag : str, optional
            Column name used to tag and group data.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        super().__init__(
            data_source,
            sep=sep,
            header=header,
            names=names,
            usecols=usecols,
            nrows=nrows,
            comment=comment,
        )
        self._tag = tag
        self._make_groups(self._tag)

    def __iter__(self) -> Iterator[Data]:
        """Return an iterator over the grouped Data objects.

        Returns
        -------
        Iterator[Data]
            An iterator over the grouped Data objects.

        """
        return iter(self.datadict.values())

    def _make_groups(self, by: Unknown) -> None:
        """Group the data by the specified tag and stores it in `datadict`."""
        self._datadict = {}
        try:
            groups = self.dataframe.groupby(by)
        except KeyError:
            self._datadict = {"unknown": Data(self.dataframe)}
        else:
            self._datadict = {str(k): Data(groups.get_group(k).reset_index(drop=True)) for k in groups.groups}

    @property
    def datadict(self) -> dict[str, Data]:
        """Retrieve the dictionary of grouped Data objects.

        Returns
        -------
        dict of str, Data
            Dictionary of grouped Data objects.

        """
        return self._datadict

    def tags(self) -> KeysView[str]:
        """Return the tags associated with the data groups.

        Returns
        -------
        KeysView of str
            Dictionary keys of tags.

        """
        return self.datadict.keys()

    def items(self) -> ItemsView[str, Data]:
        """Retrieve the items (tag and Data object) of the grouped data.

        Returns
        -------
        ItemsView of tuple of (str, Data)
            Dictionary items of tag-Data object pairs.

        """
        return self.datadict.items()

    @overload
    def get(self, tag: str) -> Data: ...

    @overload
    def get(self, tag: str, default: Data | NoDefault) -> Data: ...

    @overload
    def get(self, tag: str, default: None) -> Data | None: ...

    def get(self, tag, default=no_default):
        """Retrieve the Data object associated with the specified tag.

        Parameters
        ----------
        tag : str
            Tag of the data group to retrieve.

        default : Data or None, optional
            The default data object when the specified tag is not found.

        Returns
        -------
        Data
            Data object corresponding to the tag.

        Raises
        ------
        KeyError
            If the specified tag value does not exist and no default value is given.

        """
        if default is no_default:
            return self.datadict[tag]
        if default is None:
            # note: this return statement seems nonsense, but without it type checker produces an
            # error, somehow.
            return self.datadict.get(tag, None)
        return self.datadict.get(tag, default)

    @overload
    def param(self, tag: str, key: int | str) -> NumericType: ...

    @overload
    def param(self, tag: str, key: Sequence) -> pd.Series: ...

    def param(self, tag, key):
        """Retrieve specific parameter(s) for column(s) from a tagged Data object.

        Parameters
        ----------
        tag : str
            Tag of the data group to retrieve.

        key : int or str or Sequence of int or str
            The column(s) for which to compute the parameter.

        Returns
        -------
        Numeric type or pd.Series
            Computed parameter value(s).

        """
        return self.get(tag).param(key)

    def __str__(self) -> str:
        """Return a string of the grouped mapping of tag to Data.

        Returns
        -------
        str
            String representation of the grouped mapping of tag to Data.

        """
        return str(self.datadict)

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns
        -------
        str
            String representation of the object.

        """
        if self.is_loaded_from_file():
            return f"{self.__class__.__name__}({self.datapath}, tag={self._tag})"
        return f"{self.__class__.__name__}({self.dataframe}, tag={self._tag})"


# Local Variables:
# jinx-local-words: "Enum Hashable StringIO csv datadict datadir dataframe datapath dtype ndarray np nrows param sep str usecols" # noqa: E501
# End:
