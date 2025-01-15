import dataclasses
import enum
import logging
import os
import pathlib
import subprocess
import time
import typing

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Resources:
    cpus: int
    memory: int
    gpus: int | None = None


@dataclasses.dataclass(frozen=True)
class Configuration:
    time_limit: str
    partition_name: str | None = None


class Job:
    _SBATCH_COMMAND: str = "sbatch"
    _DEPENDENCY_TEMPLATE: str = "afterok:{id}"

    def __init__(
        self,
        resources: Resources,
        configuration: Configuration,
        name: str,
        command: str,
        dependencies: list[str] | None = None,
        path: str | os.PathLike = pathlib.Path("./logs/%j/"),
    ):
        self._resources = resources
        self._configuration = configuration
        self._command = command
        self._name = name
        self._dependencies = dependencies
        self._path = path

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str | os.PathLike:
        return self._path

    @property
    def command(self) -> str:
        return self._command

    @property
    def dependencies(self) -> list[str] | None:
        return self._dependencies

    def _build_sbatch(self) -> list[str]:
        options = {
            "--nodes": 1,
            "--job-name": self._name,
            "--output": os.path.join(self._path, f"{"%j"}_{self._name}.out"),
            "--error": os.path.join(self._path, f"{"%j"}_{self._name}.err"),
            "--cpus-per-task": self._resources.cpus,
            "--gpus": self._resources.gpus,
            "--mem": f"{self._resources.memory}G",
            "--time": self._configuration.time_limit,
            "--partition": self._configuration.partition_name,
            "--dependency": (
                ",".join(
                    [
                        self._DEPENDENCY_TEMPLATE.format(id=identifier)
                        for identifier in self._dependencies
                    ]
                )
                if self._dependencies
                else None
            ),
        }

        command = [self._SBATCH_COMMAND]
        for option, value in options.items():
            if value is not None:
                command.append(f"{option}={value}")

        return command + ["--parsable", "--wrap", f'"{self._command}"']

    def submit(self) -> str:
        def _parse(output: str) -> str:
            NEW_LINE = "\n"

            if NEW_LINE in output:
                return output.replace("\n", "")

            return output

        command = " ".join(self._build_sbatch())

        result = subprocess.run(
            command,
            shell=True,
            text=True,
            check=True,
            env=os.environ.copy(),
            capture_output=True,
        )

        return _parse(result.stdout)
