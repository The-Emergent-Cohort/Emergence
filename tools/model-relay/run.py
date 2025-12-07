#!/usr/bin/env python3
"""
Model Relay Runner
Entry point for the multi-model browser relay system.

Usage:
    python run.py                    # Start with all enabled models
    python run.py --models silas gemini   # Start with specific models
    python run.py --list             # List available models
    python run.py --test silas       # Test connection to a model
"""

import argparse
import sys
from config import MODELS


def list_models():
    """Display available models and their status."""
    print("\nAvailable Models:")
    print("-" * 50)
    for name, cfg in MODELS.items():
        enabled = cfg.get("enabled", True)
        model_type = cfg.get("type", "browser")
        url = cfg.get("url", "N/A")
        status = "enabled" if enabled else "disabled"
        print(f"  {name:12} [{model_type:7}] {status:8} - {url}")
    print()


def test_model(name: str):
    """Test connection to a single model."""
    if name not in MODELS:
        print(f"Unknown model: {name}")
        return False

    print(f"\nTesting connection to {name}...")

    from models import MODEL_CLASSES

    if name not in MODEL_CLASSES:
        print(f"No interface class for {name}")
        return False

    try:
        interface = MODEL_CLASSES[name](name=name)
        print(f"  Launching browser...")
        interface.connect()
        print(f"  Connected successfully!")

        input("  Press Enter to disconnect...")

        interface.disconnect()
        print(f"  Disconnected.")
        return True

    except Exception as e:
        print(f"  Failed: {e}")
        return False


def run_relay(models=None):
    """Run the main relay loop."""
    from relay import create_relay

    print("\n" + "=" * 50)
    print("  Model Relay System")
    print("=" * 50)

    relay = create_relay(models)

    if not relay.interfaces:
        print("\nNo models connected. Exiting.")
        return

    relay.run_loop()


def interactive_mode():
    """Run in interactive mode for manual message sending."""
    from relay import create_relay

    print("\n" + "=" * 50)
    print("  Model Relay - Interactive Mode")
    print("=" * 50)

    relay = create_relay()

    if not relay.interfaces:
        print("\nNo models connected. Exiting.")
        return

    print("\nCommands:")
    print("  send <model> <message>  - Send message to a model")
    print("  check                   - Check for new messages")
    print("  list                    - List connected models")
    print("  quit                    - Exit")
    print()

    try:
        while True:
            cmd = input("> ").strip()

            if not cmd:
                continue

            parts = cmd.split(maxsplit=2)
            action = parts[0].lower()

            if action == "quit" or action == "exit":
                break

            elif action == "list":
                print("Connected models:", list(relay.interfaces.keys()))

            elif action == "check":
                messages = relay.check_for_messages()
                if messages:
                    for msg in messages:
                        print(f"\n[{msg['from']}]:")
                        print(msg['content'][:500])
                        if len(msg['content']) > 500:
                            print("... (truncated)")
                else:
                    print("No new messages")

            elif action == "send":
                if len(parts) < 3:
                    print("Usage: send <model> <message>")
                    continue
                target = parts[1]
                message = parts[2]
                relay.send_to(target, message, from_name="operator")

            else:
                print(f"Unknown command: {action}")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        relay.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model browser relay system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Specific models to connect (default: all enabled)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--test", "-t",
        metavar="MODEL",
        help="Test connection to a specific model"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.test:
        success = test_model(args.test)
        sys.exit(0 if success else 1)

    if args.interactive:
        interactive_mode()
        return

    # Default: run relay loop
    run_relay(args.models)


if __name__ == "__main__":
    main()
