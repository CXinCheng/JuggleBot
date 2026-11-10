"""Optional UTF-8 console-and-file output for the v6 verification CLIs."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import sys


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def tee_output(path=None):
    """Replace an explicitly named log; preserve console streams and exit codes."""
    if path is None:
        yield
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8', newline='\n') as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            yield
