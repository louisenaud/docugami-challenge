"""
File:        hydra_sklearn_pipeline.py
Created by:  Louise Naud
On:          6/16/23
At:          11:48 AM
For project: docugami-challenge
Description:
Usage:
"""
import hydra
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline


def make_pipeline(steps_config: DictConfig) -> Pipeline:
    """Creates a pipeline with all the preprocessing steps specified in `steps_config`, ordered in a sequential manner

    Args:
        steps_config (DictConfig): the config containing the instructions for
                                    creating the feature selectors or transformers

    Returns:
        [sklearn.pipeline.Pipeline]: a pipeline with all the preprocessing steps, in a sequential manner
    """
    steps = []

    for step_config in steps_config:
        its = list(step_config.items())
        step_items = its[0]
        # retrieve the name and parameter dictionary of the current steps
        step_name, step_params = step_items

        # instantiate the pipeline step, and append to the list of steps
        pipeline_step = (step_name, hydra.utils.instantiate(step_params))
        steps.append(pipeline_step)

    return Pipeline(steps)
