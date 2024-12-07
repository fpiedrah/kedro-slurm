import typing

from kedro.pipeline.node import Node

from kedro_slurm import slurm


class SLURMNode(Node):

    _DEFAULT_RESOURCES = slurm.Resources(cpus=4, memory=10)
    _DEFAULT_CONFIGURATION = slurm.Configuration(time_limit="1:00:00")

    def __init__(
        self,
        func: typing.Callable,
        inputs: str | list[str] | dict[str, str] | None,
        outputs: str | list[str] | dict[str, str] | None,
        *,
        name: str | None = None,
        tags: str | typing.Iterable[str] | None = None,
        confirms: str | list[str] | None = None,
        namespace: str | None = None,
        resources: slurm.Resources | None = None,
        configuration: slurm.Configuration | None = None,
    ):
        if resources and not isinstance(resources, slurm.Resources):
            raise ValueError(
                f"Invalid type for 'resources': "
                f"expected None or slurm.Resources, "
                f"got {type(resources).__name__}."
            )

        if configuration and not isinstance(configuration, slurm.Configuration):
            raise ValueError(
                f"Invalid type for 'configuration': "
                f"expected None or slurm.Configuration, "
                f"got {type(configuration).__name__}."
            )

        self._resources = resources if resources else self._DEFAULT_RESOURCES
        self._configuration = (
            configuration if configuration else self._DEFAULT_CONFIGURATION
        )

        super().__init__(
            func=func,
            inputs=inputs,
            outputs=outputs,
            name=name,
            tags=tags,
            confirms=confirms,
            namespace=namespace,
        )

    @property
    def resources(self):
        return self._resources

    @property
    def configuration(self):
        return self._configuration

    def _copy(self, **overwrite_params: typing.Any) -> Node:
        params = {
            "func": self._func,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "name": self._name,
            "namespace": self._namespace,
            "tags": self._tags,
            "confirms": self._confirms,
            "resources": self._resources,
            "configuration": self._configuration,
        }

        params.update(overwrite_params)
        return SLURMNode(**params)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SLURMNode({self._func_name}, {self._inputs!r}, {self._outputs!r}, "
            f"{self._name!r})"
        )


def node(
    func: typing.Callable,
    inputs: str | list[str] | dict[str, str] | None,
    outputs: str | list[str] | dict[str, str] | None,
    *,
    name: str | None = None,
    tags: str | typing.Iterable[str] | None = None,
    confirms: str | list[str] | None = None,
    namespace: str | None = None,
    resources: slurm.Resources | None = None,
    configuration: slurm.Configuration | None = None,
) -> SLURMNode:
    return SLURMNode(
        func,
        inputs,
        outputs,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
        resources=resources,
        configuration=configuration,
    )
