"""
File:        printing.py
Created by:  Louise Naud
On:          6/16/23
At:          3:13 PM
For project: docugami-challenge
Description:
Usage:
"""
from typing import Sequence

from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.syntax import Syntax
from rich.tree import Tree


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "preprocessing_pipeline",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Based on: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    print(tree)
