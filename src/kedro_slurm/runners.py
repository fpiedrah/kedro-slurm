from __future__ import annotations

import collections
import itertools
import typing

from kedro_slurm import slurm

from kedro.io import CatalogProtocol, MemoryDataset
from kedro.runner.runner import AbstractRunner

if typing.TYPE_CHECKING:
    from pluggy import PluginManager

    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node


# EACH NODE SHOULD HAVE THEIR OWN
DEFAULT_RESOURCES = slurm.Resources(cpus=4, memory=10)
DEFAULT_CONFIGURATION = slurm.Configuration(time_limit="1:00:00")


class SLURMRunner(AbstractRunner):
    def __init__(self, is_async: bool = False):
        super().__init__(is_async=is_async, extra_dataset_patterns=None)

    @classmethod
    def _build_command(cls, node: str) -> str:
        KEDRO_COMMAND = "kedro"

        return f"{KEDRO_COMMAND} run --nodes {node} --async"

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

        node_dependencies: dict = pipeline.node_dependencies
        todo_nodes: set[Node] = set(node_dependencies.keys())
        done_nodes: set[Node] = set()
        futures: set[slurm.Future] = set()

        while True:
            ready = {
                node for node in todo_nodes if node_dependencies[node] <= done_nodes
            }

            todo_nodes -= ready
            for node in ready:
                job =  slurm.Job(
                    DEFAULT_RESOURCES, 
                    DEFAULT_CONFIGURATION,
                    node.name,
                    self._build_command(node.name)
                )

                futures.add(job.submit())

            if not futures:
                if todo_nodes:
                    debug_data = {
                        "todo_nodes": todo_nodes,
                        "done_nodes": done_nodes,
                        "ready_nodes": ready,
                        "done_futures": done,
                    }

                    debug_data_str = "\n".join(
                        f"{key} = {value}" for key, value in debug_data.items()
                    )

                    raise RuntimeError(
                        f"Unable to schedule new tasks although some nodes "
                        f"have not been run:\n{debug_data_str}"
                    )

                break

            # MISSING ERROR HANDLING
            slurm.wait(futures)
            for node in ready:
                done_nodes.add(node)
                
                self._release_datasets(node, catalog, load_counts, pipeline)
