"""The `modal` CLI, using the operating system's certificate store.

Only needed on networks that intercept TLS (antivirus / corporate proxy), where plain `modal` fails with
"Could not connect to the Modal server". Same arguments as `modal`:

    uvx --native-tls --from modal --with truststore python scripts/modal_cli.py run modal_app.py::check
"""

import sys

import rich.console
import truststore

truststore.inject_into_ssl()
# Windows: write UTF-8 and plain ANSI, so output also works when redirected to a file.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
rich.console.detect_legacy_windows = lambda: False

from modal.cli.entry_point import entrypoint_cli  # noqa: E402

sys.argv[0] = "modal"
entrypoint_cli()
