"""Console output logging -- captures stdout/stderr to a log file.

Adapted from assignment2/src/utils/console_logger.py.
Driven by the :class:`configs.base.Console` dataclass.
"""

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class TeeStream:
    """Stream wrapper that writes to both a file and the original stream."""

    def __init__(
        self,
        file_handle: TextIO,
        original_stream: TextIO,
        tee_to_console: bool = True,
        line_timestamps: bool = False,
        timestamp_format: str = "%H:%M:%S",
    ):
        self.file_handle = file_handle
        self.original_stream = original_stream
        self.tee_to_console = tee_to_console
        self.line_timestamps = line_timestamps
        self.timestamp_format = timestamp_format
        self._line_buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        output_text = self._add_timestamps(text) if self.line_timestamps else text

        self.file_handle.write(output_text)
        self.file_handle.flush()

        if self.tee_to_console and self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()

        return len(text)

    def _add_timestamps(self, text: str) -> str:
        lines = text.split("\n")
        result = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1 and line == "":
                result.append("")
            elif line or i < len(lines) - 1:
                ts = datetime.now().strftime(self.timestamp_format)
                if self._line_buffer:
                    result.append(line)
                    self._line_buffer = ""
                else:
                    result.append(f"[{ts}] {line}")
        if text and not text.endswith("\n"):
            self._line_buffer = lines[-1] if lines else ""
        return "\n".join(result)

    def flush(self) -> None:
        self.file_handle.flush()
        if self.tee_to_console and self.original_stream:
            self.original_stream.flush()

    def fileno(self) -> int:
        if self.original_stream:
            return self.original_stream.fileno()
        return self.file_handle.fileno()

    def isatty(self) -> bool:
        if self.tee_to_console and self.original_stream:
            return self.original_stream.isatty()
        return False

    @property
    def encoding(self) -> str:
        if self.original_stream:
            return self.original_stream.encoding
        return "utf-8"


class ConsoleLogger:
    """Context manager that tees stdout/stderr to a log file.

    Usage::

        with ConsoleLogger(run_dir, config.console):
            print("hello")  # goes to both console and run_dir/console.log
    """

    def __init__(self, run_dir: Path, console_config=None):
        """
        Args:
            run_dir: Directory for the log file.
            console_config: A :class:`configs.base.Console` dataclass
                (or *None* for defaults).
        """
        self.run_dir = Path(run_dir)

        # Accept either a Console dataclass or a plain dict
        if console_config is None:
            self._cfg: dict = {}
        elif hasattr(console_config, "__dataclass_fields__"):
            self._cfg = asdict(console_config)
        else:
            self._cfg = dict(console_config)

        self.enabled: bool = self._cfg.get("enabled", True)

        self._original_stdout: Optional[TextIO] = None
        self._original_stderr: Optional[TextIO] = None
        self._file: Optional[TextIO] = None
        self._stdout_tee: Optional[TeeStream] = None
        self._stderr_tee: Optional[TeeStream] = None
        self._is_active = False

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "ConsoleLogger":
        if self.enabled:
            self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- start / stop --------------------------------------------------------

    def _start(self) -> None:
        if self._is_active:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)

        filename = self._cfg.get("filename", "console.log")
        tee = self._cfg.get("tee_to_console", True)
        ts = self._cfg.get("line_timestamps", False)
        ts_fmt = self._cfg.get("timestamp_format", "%H:%M:%S")

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        self._file = open(self.run_dir / filename, "w", encoding="utf-8")

        self._stdout_tee = TeeStream(
            self._file, self._original_stdout,
            tee_to_console=tee, line_timestamps=ts, timestamp_format=ts_fmt,
        )
        self._stderr_tee = TeeStream(
            self._file, self._original_stderr,
            tee_to_console=tee, line_timestamps=ts, timestamp_format=ts_fmt,
        )

        sys.stdout = self._stdout_tee  # type: ignore[assignment]
        sys.stderr = self._stderr_tee  # type: ignore[assignment]
        self._is_active = True

    def close(self) -> None:
        """Stop capturing and restore original streams."""
        if not self._is_active:
            return
        if self._stdout_tee:
            self._stdout_tee.flush()
        if self._stderr_tee:
            self._stderr_tee.flush()
        sys.stdout = self._original_stdout  # type: ignore[assignment]
        sys.stderr = self._original_stderr  # type: ignore[assignment]
        if self._file:
            self._file.close()
            self._file = None
        self._is_active = False
