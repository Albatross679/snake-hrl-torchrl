"""Entry point for running aprx_model_dismech as a module.

Usage:
    python -m aprx_model_dismech collect   -- collect training data
    python -m aprx_model_dismech train     -- train surrogate model
    python -m aprx_model_dismech validate  -- validate surrogate accuracy
    python -m aprx_model_dismech rl        -- train RL with surrogate env
    python -m aprx_model_dismech monitor   -- monitor running collection
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift argv so submodule argparse works

    if command == "collect":
        from aprx_model_dismech.collect_data import main as collect_main
        collect_main()
    elif command == "train":
        from aprx_model_dismech.train_surrogate import main as train_main
        train_main()
    elif command == "validate":
        from aprx_model_dismech.validate import main as validate_main
        validate_main()
    elif command == "rl":
        from aprx_model_dismech.train_rl import main as rl_main
        rl_main()
    elif command == "monitor":
        from aprx_model_dismech.monitor import main as monitor_main
        monitor_main()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
