#!/usr/bin/env python3
"""Generate a mini-enclava attestation policy document."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

SCHEMA_ID = "mini-enclava-attestation-policy/v1"
DEFAULT_ATTESTATION_TYPE = "coco-sev-snp"
DEFAULT_RUNTIME_CLASS = "kata-qemu-snp"
DEFAULT_KBS_RESOURCE_PATH = "default/flowforge-storage/workload-secret-seed"
DEFAULT_REPORT_POLICY = 196608


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_policy(args: argparse.Namespace) -> dict:
    return {
        "schema": SCHEMA_ID,
        "generated_at": utc_now(),
        "generator": {
            "name": "mini-enclava-policy-ci",
            "source_ref": args.source_ref,
            "image_tag": args.image_tag,
        },
        "expected": {
            "attestation_type": args.attestation_type,
            "runtime_class": args.runtime_class,
            "workload_image": args.workload_image,
            "kbs_resource_path": args.kbs_resource_path,
        },
        "report_constraints": {
            "vmpl": args.vmpl,
            "policy": args.report_policy,
            "measurement_hex": args.measurement_hex.lower() if args.measurement_hex else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mini-enclava attestation policy JSON")
    parser.add_argument("--output", required=True, help="Output policy path")
    parser.add_argument("--workload-image", required=True, help="Pinned image reference (must include @sha256)")
    parser.add_argument("--source-ref", default="", help="Git ref that produced this policy")
    parser.add_argument("--image-tag", default="", help="Human-friendly image tag")
    parser.add_argument("--attestation-type", default=DEFAULT_ATTESTATION_TYPE)
    parser.add_argument("--runtime-class", default=DEFAULT_RUNTIME_CLASS)
    parser.add_argument("--kbs-resource-path", default=DEFAULT_KBS_RESOURCE_PATH)
    parser.add_argument("--vmpl", type=int, default=0)
    parser.add_argument("--report-policy", type=int, default=DEFAULT_REPORT_POLICY)
    parser.add_argument("--measurement-hex", default=None)
    parser.add_argument("--allow-non-digest-image", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "@sha256:" not in args.workload_image and not args.allow_non_digest_image:
        raise SystemExit("workload image must be digest-pinned (@sha256:...)")

    policy = build_policy(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
        f.write("\n")

    print(f"wrote policy: {args.output}")
    print(f"workload_image={args.workload_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
