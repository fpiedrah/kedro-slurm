from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING, Any

from kedro.runner.runner import AbstractRunner
from kedro.runner.task import Task

if TYPE_CHECKING:
    from pluggy import PluginManager

    from kedro.io import CatalogProtocol
    from kedro.pipeline import Pipeline

from kedro_slurm import slurm


DEFAULT_RESOURCES = slurm.Resources(cpus=4, memory=10)
DEFAULT_CONFIGURATION = slurm.Configuration(time_limit="1:00:00")


def _build(node: str) -> str:
    KEDRO_COMMAND = "kedro"

    return f"{KEDRO_COMMAND} --nodes {node}"


class SLURMSequentialRunner(AbstractRunner):

    def __init__(
        self,
        is_async: bool = False,
        extra_dataset_patterns: dict[str, dict[str, Any]] | None = None,
    ):
        default_dataset_pattern = {"{default}": {"type": "MemoryDataset"}}
        self._extra_dataset_patterns = extra_dataset_patterns or default_dataset_pattern
        super().__init__(
            is_async=is_async, extra_dataset_patterns=self._extra_dataset_patterns
        )

    def _run(
        self,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: str | None = None,
    ) -> None:
        nodes = pipeline.nodes
        done_nodes = set()

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))

        for exec_index, node in enumerate(nodes):
            try:
                future = slurm.Job(
                    DEFAULT_RESOURCES, 
                    DEFAULT_CONFIGURATION,
                    node.name,
                    _build(node.name)
                ).submit()

                slurm.wait([future])
            except Exception:
                self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                raise

            self._release_datasets(node, catalog, load_counts, pipeline)
            self._logger.info(
                "Completed %d out of %d tasks", len(done_nodes), len(nodes)
            )
