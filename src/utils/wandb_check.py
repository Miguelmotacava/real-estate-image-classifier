"""Quick W&B authentication smoke test."""
from __future__ import annotations

import os
import sys

import wandb
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    if not api_key or not entity or not project:
        print("[ERROR] Missing W&B env vars; check your .env file.")
        return 1

    try:
        wandb.login(key=api_key, relogin=True)
        run = wandb.init(
            entity=entity,
            project=project,
            name="connection-test",
            mode="online",
            tags=["connectivity"],
        )
        print(f"Run URL: {run.get_url()}")
        wandb.log({"smoke_ok": 1.0})
        wandb.finish()
        return 0
    except Exception as exc:  # noqa: BLE001 - expose full message
        print(f"[ERROR] W&B connection failed: {exc!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
