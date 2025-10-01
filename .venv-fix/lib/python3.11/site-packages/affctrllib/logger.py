from __future__ import annotations

import copy
import sys
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import SupportsWrite


class Logger:
    _sep: str
    _eol: str
    _labels: list[str]
    _rawdata: list[list[Any]]
    _fpath: Path
    _lock: Lock

    def __init__(self, fname: str | Path | None = None, sep: str = ",", eol: str = "\n") -> None:
        self._sep = sep
        self._eol = eol
        self._labels = []
        self._rawdata = []
        self._lock = Lock()
        self.fpath = fname  # type: ignore[assignment]

    def __len__(self) -> int:
        with self._lock:
            return len(self._rawdata)

    @property
    def sep(self) -> str:
        return self._sep

    @property
    def eol(self) -> str:
        return self._eol

    @property
    def fpath(self) -> str | None:
        try:
            with self._lock:
                return str(self._fpath)
        except AttributeError:
            return None

    @fpath.setter
    def fpath(self, fname: str | Path | None) -> None:
        if fname is not None:
            with self._lock:
                self._fpath = Path(fname)
        else:
            try:
                with self._lock:
                    del self._fpath
            except AttributeError:
                pass

    def get_filename(self) -> str | None:
        return self.fpath

    def set_filename(self, fname: str | Path) -> None:
        self.fpath = fname  # type: ignore[assignment]

    def set_labels(self, *args: str | Iterable[str]) -> None:
        listed = [[arg] if isinstance(arg, str) else arg for arg in args]
        with self._lock:
            self._labels = [x for s in listed for x in s]

    def extend_labels(self, *args: str | Iterable[str]) -> None:
        listed = [[arg] if isinstance(arg, str) else arg for arg in args]
        with self._lock:
            self._labels.extend([x for s in listed for x in s])

    def get_header(self) -> str:
        with self._lock:
            return self.sep.join(self._labels)

    def store_data(self, data: Iterable[Any]) -> None:
        with self._lock:
            self._rawdata.append(list(data))

    def extend_data(self, data: Iterable[Any]) -> None:
        try:
            with self._lock:
                self._rawdata[-1].extend(list(data))
        except IndexError:
            self.store_data(data)

    def store(self, *args: float | Iterable[Any]) -> None:
        listed = [[arg] if isinstance(arg, float | int) else arg for arg in args]
        data = [x for s in listed for x in s]
        self.store_data(data)

    def get_data(self) -> list[list[Any]]:
        with self._lock:
            return self._rawdata

    def copy_data(self) -> list[list[Any]]:
        with self._lock:
            return copy.deepcopy(self._rawdata)

    def _slice_data_get_index_list(self, label: str | int | Iterable[str | int]) -> list[int]:
        def ensure_index(label_or_index: str | int) -> int:
            if isinstance(label_or_index, str):
                return self._labels.index(label_or_index)
            return label_or_index

        if isinstance(label, str | int):
            return [ensure_index(label)]
        return [ensure_index(i) for i in label]

    def slice_data(
        self,
        label: str | int | Iterable[str | int],
        index: tuple[int, int] | None = None,
    ) -> list[list[Any]]:
        with self._lock:
            cols = self._slice_data_get_index_list(label)
            row_range = (0, len(self._rawdata))
            if index is not None:
                row_range = (
                    index[0] if len(index) > 0 else 0,
                    index[1] if len(index) > 1 else len(self._rawdata),
                )
            extracted_data = self._rawdata[row_range[0] : row_range[1]]
        return [[row[c] for row in extracted_data] for c in cols]

    def erase_data(self) -> None:
        del self._rawdata
        self._rawdata = []

    def _generate_alternative_fname(self, basefn: Path) -> Path:
        stem = basefn.stem
        suffix = basefn.suffix
        cnt = 1
        while True:
            cand = basefn.parent / Path(f"{stem}.{cnt}{suffix}")
            if not cand.exists():
                return cand
            cnt += 1

    def _dump_print(self, output: SupportsWrite[str] | None = None) -> None:
        if len(self._labels):
            print(self.get_header(), end=self.eol, file=output)
        with self._lock:
            for line in self._rawdata:
                print(self.sep.join(f"{x}" for x in line), end=self.eol, file=output)

    def _dump_open(self, fpath: Path, mode: str) -> None:
        with Path(fpath).open(mode=mode) as fobj:
            if len(self._labels):
                fobj.write(self.get_header() + self.eol)
            with self._lock:
                fobj.writelines(self.sep.join(f"{x}" for x in line) + self.eol for line in self._rawdata)

    def dump(
        self,
        fname: str | Path | None = None,
        mode: str = "w",
        *,
        overwrite: bool = True,
        quiet: bool = False,
    ) -> None:
        fpath: Path
        if fname is not None:
            fpath = Path(fname)
        elif self.fpath is not None:
            fpath = self._fpath
        else:
            self._dump_print()
            return

        if not overwrite:
            fpath = self._generate_alternative_fname(fpath)
        if not quiet:
            sys.stdout.write(f"Saving data in <{fpath}>... ")
            sys.stdout.flush()
        self._dump_open(fpath, mode)
        if not quiet:
            sys.stdout.write("done.\n")
