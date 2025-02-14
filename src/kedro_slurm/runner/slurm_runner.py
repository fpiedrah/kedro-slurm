from __future__ import annotations

import collections
import itertools
import os
import typing

from kedro.io import CatalogProtocol, MemoryDataset
from kedro.runner.runner import AbstractRunner

from kedro_slurm import slurm
from kedro_slurm.pipeline.node import SLURMNode

if typing.TYPE_CHECKING:
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node
    from pluggy import PluginManager


class SLURMRunner(AbstractRunner):
    def __init__(self, is_async: bool = False):
        super().__init__(is_async=is_async, extra_dataset_patterns=None)

    def _build_command(self, node: str) -> str:
        KEDRO_COMMAND = "kedro run"

        # FIND A BETTER WAY TO PASS THE ENV
        env = os.environ.get("KEDRO_ENV", None)
        params = os.environ.get("KEDRO_PARAMS", None)

        command_parts = [f"{KEDRO_COMMAND}", f"--nodes '{node}'"]

        if env:
            command_parts.append(f"--env {env}")

        if params:
            print(params)
            print(type(params))
            command_parts.append(f"--params '{params}'")

        return " ".join(command_parts)

    #def _build_command(self, node: str) -> str:
    #    KEDRO_COMMAND = "kedro run"

    #    # FIND A BETTER WAY TO PASS THE ENV AND PARAMS
    #    env = os.environ.get("KEDRO_ENV", None)
    #    params = os.environ.get("KEDRO_PARAMS", None)

    #    cmd = [f"{KEDRO_COMMAND}", f"--nodes '{node}'"]

    #    if not env:
    #        return cmd.append(f"--env {env}")

    #    if not params:
    #        return cmd.append(f"--params '{params}'")

    #    print(" ".join(cmd))
    #    return " ".join(cmd)

    @classmethod
    def _validate_catalog(cls, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        datasets = catalog._datasets

        memory_datasets = []
        for name, dataset in datasets.items():
            if name in pipeline.all_outputs() and isinstance(dataset, MemoryDataset):
                memory_datasets.append(name)

        if memory_datasets:
            raise AttributeError(
                f"The following datasets are memory datasets: "
                f"{sorted(memory_datasets)}\n"
                f"SLURsMRunner does not support output to MemoryDataSets"
            )

    def _run(
        self,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: str | None = None,
    ) -> None:
        self._validate_catalog(catalog, pipeline)

        nodes = pipeline.nodes
        load_counts = collections.Counter(
            itertools.chain.from_iterable(node.inputs for node in nodes)
        )

        identifier_mapping: dict[Node, str] = {}
        node_dependencies: dict = pipeline.node_dependencies
        todo_nodes: set[Node] = set(node_dependencies.keys())
        done_nodes: set[Node] = set()
        futures: set[slurm.Job] = set()

        while True:
            ready = {
                node for node in todo_nodes if node_dependencies[node] <= done_nodes
            }

            todo_nodes -= ready
            ran_some_node = False

            for node in ready:
                future_dependencies: list = []
                resources: slurm.Resources = SLURMNode._DEFAULT_RESOURCES
                configuration: slurm.Configuration = SLURMNode._DEFAULT_CONFIGURATION

                if not isinstance(node, SLURMNode):
                    self._logger.warning(
                        f"Node {node} is not of type SLURMNode (actual type: {type(node).__name__}).\n"
                        f"It will be executed with default resources and configuration."
                    )

                if isinstance(node, SLURMNode):
                    resources = node.resources
                    configuration = node.configuration

                job = slurm.Job(
                    resources,
                    configuration,
                    node.func.__name__,
                    self._build_command(node.name),
                    [
                        identifier_mapping[dependency]
                        for dependency in node_dependencies[node]
                    ],
                )

                try:
                    identifier = job.submit()

                    ran_some_node = True
                    identifier_mapping[node] = identifier
                except Exception:
                    self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                    raise

                done_nodes.add(node)
                self._logger.info(
                    f"Submitted node '{node.func.__name__}' with ID '{identifier}'"
                )

            if not ran_some_node:
                if todo_nodes:
                    self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                break
