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


class Status(str, enum.Enum):
    CANCELLED: str = "CANCELLED"
    COMPLETED: str = "COMPLETED"
    COMPLETING: str = "COMPLETING"
    FAILED: str = "FAILED"
    PENDING: str = "PENDING"
    PREEMPTED: str = "PREEMPTED"
    RUNNING: str = "RUNNING"
    SUSPENDED: str = "SUSPENDED"
    STOPPED: str = "STOPPED"


class Future:
    _SACCT_COMMAND: str = "sacct"
    _SCANCEL_COMMAND: str = "scancel"

    def __init__(self, identifier: str):
        self._identifier: str = identifier
        self._previous_state: Status | None = None
        self._state: Status | None = None

    @property
    def cancelled(self):
        return self._state in [Status.SUSPENDED, Status.STOPPED, Status.CANCELLED]

    @property
    def failed(self):
        return self._state in [
            Status.FAILED,
            Status.PREEMPTED,
        ]

    @property
    def running(self):
        return self._state in [Status.RUNNING, Status.COMPLETING]

    @property
    def done(self):
        return self._state == Status.COMPLETED or self.cancelled

    @property
    def queued(self):
        return self._state in [Status.PENDING, Status.SUSPENDED]

    @property
    def changed(self):
        if not self._previous_state == self._state:
            return True

        return False

    def _build_sacct(self) -> list[str]:
        return [
            self._SACCT_COMMAND,
            f"--jobs={self._identifier}",
            "--format=State",
            "--noheader",
            "--parsable2",
        ]

    def _get_status(self) -> Status:

        def _parse(output: str) -> str:
            NEW_LINE = "\n"
            WHITE_SPACE = " "

            if NEW_LINE in output:
                output = output.split(NEW_LINE)[0]

            if WHITE_SPACE in output:
                output = output.split(WHITE_SPACE)[0]

            return output

        command = self._build_sacct()
        result = subprocess.run(
            command,
            text=True,
            check=True,
            capture_output=True,
        )

        if not result.stdout:
            return Status.PENDING

        return Status(_parse(result.stdout))

    def update(self) -> None:
        self._previous_state = self._state
        self._state = self._get_status()

    def _build_scancel(self) -> list[str]:
        return [self._SCANCEL_COMMAND, self._identifier]

    def cancel(self) -> None:
        command = self._build_scancel()
        subprocess.run(
            command,
            text=True,
            check=True,
            capture_output=True,
        )


class Job:
    _SBATCH_COMMAND: str = "sbatch"

    def __init__(
        self,
        resources: Resources,
        configuration: Configuration,
        name: str,
        command: str,
        path: str | os.PathLike = pathlib.Path("./logs/%j/"),
    ):
        self._resources = resources
        self._configuration = configuration
        self._command = command
        self._name = name
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
        }

        command = [self._SBATCH_COMMAND]
        for option, value in options.items():
            if value is not None:
                command.append(f"{option}={value}")

        return command + ["--parsable", "--wrap", f"'{self._command}'"]

    def submit(self) -> Future:

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
            capture_output=True,
        )

        return Future(_parse(result.stdout))


class SlurmExecutionError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def wait(futures: typing.Iterable[Future], interval: int = 5) -> None:
    failed = False
    missing = set(futures)

    while missing:
        for index, future in enumerate(futures):
            if future not in missing:
                continue

            future.update()

            if future.changed:
                logger.info(
                    f"[b][{future._identifier}]:[/b] "
                    f"[i orange1]{future._previous_state}[/i orange1] "
                    f"[b]->[/b] [i green]{future._state}[/i green]",
                    extra={"markup": True},
                )

            if future.cancelled:
                missing.discard(future)

            if future.failed:
                failed = True
                missing.discard(future)

                for future in missing:
                    future.cancel()

            elif future.done:
                missing.discard(future)

        if failed:
            raise SlurmExecutionError(
                "SLURM encountered an error while attempting to execute this task."
            )

        time.sleep(interval)
