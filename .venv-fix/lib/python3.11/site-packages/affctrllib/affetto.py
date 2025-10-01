from __future__ import annotations

import sys
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, final

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


class Chain:
    _dof: int
    _link_names: list[str]
    _link_normalized_names: list[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._translation_table = str.maketrans({" ": "", "-": "", "_": ""})
        if config is not None:
            self.load(config)

    @property
    def dof(self) -> int:
        return self._dof

    def set_dof(self, dof: int) -> None:
        self._dof = dof

    def translate_name(self, name: str) -> str:
        return name.translate(self._translation_table)

    def load(self, config: dict[str, Any]) -> None:
        dof = 0
        self._link_names = []
        self._link_normalized_names = []
        links: list[dict[str, Any]] = config["link"]
        for i, link in enumerate(links):
            name = link.get("name", f"link{i:02}")
            if link["jointtype"] in ["revolute", "prismatic"]:
                # Only movable links are stored.
                self._link_names.append(name)
                self._link_normalized_names.append(self.translate_name(name))
                dof += 1
        self._dof = dof
        assert self._dof == len(self._link_names)

    def find(self, name: str) -> int:
        # Only movable links are accessible.
        return self._link_normalized_names.index(self.translate_name(name))

    def name(self, index: int) -> str:
        # Only movable links are accessible.
        return self._link_names[index]


class Affetto:
    _config_path: Path
    _config: dict[str, Any]
    _name: str
    _chain: Chain

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            self.load_config_path(config_path)

    @property
    def config_path(self) -> Path:
        return self._config_path

    @final
    def load_config_path(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        with Path(self._config_path).open("rb") as f:
            c = tomllib.load(f)
        self.load_config(c)

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @abstractmethod
    def load_config(self, config_dict: dict[str, Any]) -> None:
        affetto_config = config_dict["affetto"]
        self._config = affetto_config
        self.load_name(affetto_config)
        self.load_chain(affetto_config)

    @property
    def name(self) -> str:
        return self._name

    def load_name(self, config: dict[str, Any]) -> None:
        try:
            self._name = config["name"]
        except KeyError:
            self._name = "affetto"

    @property
    def chain(self) -> Chain:
        return self._chain

    def load_chain(self, config: dict[str, Any]) -> None:
        try:
            self._chain = Chain(config["chain"])
        except KeyError:
            warnings.warn("'chain' field is not defined", UserWarning, stacklevel=2)

    @property
    def dof(self) -> int:
        try:
            return self._chain.dof
        except AttributeError:
            return 13

    def joint_index(self, name: str) -> int:
        return self._chain.find(name)

    def joint_name(self, index: int) -> str:
        return self._chain.name(index)


# Local Variables:
# jinx-local-words: "affetto jointtype rb revolute"
# End:
