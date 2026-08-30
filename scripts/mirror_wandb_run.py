from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import time

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination-entity", required=True)
    parser.add_argument("--destination-project", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination_api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not destination_api_key:
        raise ValueError("destination W&B API key file is empty")
    source_api = wandb.Api(timeout=60)
    source = source_api.run(args.source)
    destination = wandb.init(
        entity=args.destination_entity,
        project=args.destination_project,
        id=source.id,
        name=source.name,
        config={**dict(source.config), "mirrored_from": "/".join(source.path)},
        resume="allow",
        reinit="finish_previous",
        settings=wandb.Settings(api_key=destination_api_key),
    )
    last_step = int(destination.summary.get("mirrored_through_step", -1))
    print(
        json.dumps(
            {
                "event": "mirror_ready",
                "source": "/".join(source.path),
                "destination": f"{args.destination_entity}/{args.destination_project}/{source.id}",
                "last_step": last_step,
                "user": getpass.getuser(),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    while True:
        source_api.flush()
        source = source_api.run(args.source)
        copied = 0
        newest = last_step
        for row in source.scan_history(page_size=1000):
            step = row.get("_step")
            if step is None or int(step) <= last_step:
                continue
            payload = {key: value for key, value in row.items() if not key.startswith("_") and value is not None}
            destination.log(payload, step=int(step))
            newest = max(newest, int(step))
            copied += 1
        last_step = newest
        destination.summary.update(
            {
                "mirrored_source_state": source.state,
                "mirrored_through_step": last_step,
                "sync_updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            }
        )
        print(
            json.dumps(
                {
                    "event": "mirror_progress",
                    "source_state": source.state,
                    "copied": copied,
                    "last_step": last_step,
                    "destination_url": (
                        f"https://wandb.ai/{args.destination_entity}/{args.destination_project}/runs/{source.id}"
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        if source.state not in {"running", "pending"}:
            destination.summary.update(dict(source.summary))
            destination.finish(exit_code=0 if source.state == "finished" else 1)
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
