"""opceil command line: `python -m opceil <command> ...`."""
from __future__ import annotations

import sys

COMMANDS = {
    "predict": "opceil.model.pcli",
    "validate": "opceil.model.validate",
    "trace": "opceil.frontends.tcli",
    "check": "opceil.model.check",
    "transfer": "opceil.model.transfer",
    "anchors": "opceil.model.anchors",
    "calibrate": "opceil.calibrate.run",
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: python -m opceil {%s} [args...]" % ",".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    import importlib
    mod = importlib.import_module(COMMANDS[argv[0]])
    return mod.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
