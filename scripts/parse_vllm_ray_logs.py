#!/usr/bin/env python3
"""Parse vLLM metrics from vLLM/Ray logs.

The parser extracts periodic vLLM log lines such as:
  - "Avg prompt throughput: ..."
  - "MFU: ... TF/s/GPU ..."
  - uvicorn completion lines for /v1/completions and /v1/chat/completions

It can aggregate these samples either:
  - by fixed wall-clock interval, or
  - by approximate completed-request count.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path


THROUGHPUT_RE = re.compile(
    r"""
    (?P<month>\d{2})-(?P<day>\d{2})\s+
    (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})
    .*?Avg\ prompt\ throughput:\s+(?P<prompt>[0-9.]+)\s+tokens/s,
    \s+Avg\ generation\ throughput:\s+(?P<generation>[0-9.]+)\s+tokens/s,
    \s+Running:\s+(?P<running>\d+)\s+reqs,
    \s+Waiting:\s+(?P<waiting>\d+)\s+reqs,
    .*?Prefix\ cache\ hit\ rate:\s+(?P<prefix>[0-9.]+)%
    """,
    re.VERBOSE,
)

MFU_RE = re.compile(
    r"""
    (?P<month>\d{2})-(?P<day>\d{2})\s+
    (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})
    .*?MFU:\s+(?P<mfu_tflops>[0-9.]+)\s+TF/s/GPU
    \s+(?P<bandwidth>[0-9.]+)\s+GB/s/GPU
    """,
    re.VERBOSE,
)

REQUEST_RE = re.compile(
    r"""
    ^.*?"POST\s+/(?:v1/completions|v1/chat/completions|skyrl/v1/generate)\s+HTTP/1\.1"\s+200
    """,
    re.VERBOSE,
)


@dataclass
class Sample:
    ts: dt.datetime
    prompt_tokps: float | None = None
    generation_tokps: float | None = None
    running_reqs: int | None = None
    waiting_reqs: int | None = None
    prefix_hit_rate_pct: float | None = None
    mfu_tflops_per_gpu: float | None = None
    bandwidth_gbps_per_gpu: float | None = None
    request_completions_since_prev: int = 0
    source_file: str = ""


@dataclass
class Aggregate:
    start: dt.datetime
    end: dt.datetime
    samples: list[Sample] = field(default_factory=list)

    def mean(self, attr: str) -> float | None:
        vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
        return statistics.mean(vals) if vals else None

    def weighted_mean(self, value_attr: str, weight_attr: str) -> float | None:
        pairs = [
            (getattr(s, value_attr), getattr(s, weight_attr))
            for s in self.samples
            if getattr(s, value_attr) is not None and getattr(s, weight_attr) not in (None, 0)
        ]
        if not pairs:
            return None
        num = sum(v * w for v, w in pairs)
        den = sum(w for _, w in pairs)
        return num / den if den else None

    @property
    def requests(self) -> int:
        return sum(s.request_completions_since_prev for s in self.samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory containing logs to parse. If omitted, auto-discovers /tmp/skyrl-logs first, then Ray worker logs.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        action="append",
        default=[],
        help="Specific log file to parse. Can be passed multiple times.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=dt.datetime.now().year,
        help="Year to assume for log timestamps.",
    )
    parser.add_argument(
        "--bucket-minutes",
        type=float,
        default=5.0,
        help="Wall-clock bucket size in minutes when using --mode time.",
    )
    parser.add_argument(
        "--requests-per-bucket",
        type=int,
        default=200,
        help="Approximate completed requests per bucket when using --mode requests.",
    )
    parser.add_argument(
        "--mode",
        choices=["time", "requests"],
        default="time",
        help="Aggregation mode.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print which worker log files contributed samples.",
    )
    return parser.parse_args()


def parse_ts(match: re.Match[str], year: int) -> dt.datetime:
    return dt.datetime(
        year,
        int(match.group("month")),
        int(match.group("day")),
        int(match.group("hour")),
        int(match.group("minute")),
        int(match.group("second")),
    )


def discover_worker_logs(logs_dir: Path) -> list[Path]:
    return sorted(p for p in logs_dir.glob("worker-*.out") if p.is_file())


def discover_infra_logs(logs_dir: Path) -> list[Path]:
    return sorted(p for p in logs_dir.glob("infra-*.log") if p.is_file())


def discover_logs(logs_dir: Path | None, explicit_files: list[Path]) -> list[Path]:
    if explicit_files:
        return [p for p in explicit_files if p.is_file()]

    candidates: list[Path] = []
    if logs_dir is not None:
        candidates.extend(discover_infra_logs(logs_dir))
        candidates.extend(discover_worker_logs(logs_dir))
        return candidates

    default_infra_dir = Path("/tmp/skyrl-logs")
    default_ray_dir = Path("/tmp/ray/session_latest/logs")
    if default_infra_dir.exists():
        candidates.extend(discover_infra_logs(default_infra_dir))
    if not candidates and default_ray_dir.exists():
        candidates.extend(discover_worker_logs(default_ray_dir))
    return candidates


def parse_worker_log(path: Path, year: int) -> list[Sample]:
    samples: list[Sample] = []
    pending_requests = 0

    for line in path.read_text(errors="ignore").splitlines():
        if REQUEST_RE.search(line):
            pending_requests += 1
            continue

        m = THROUGHPUT_RE.search(line)
        if m:
            samples.append(
                Sample(
                    ts=parse_ts(m, year),
                    prompt_tokps=float(m.group("prompt")),
                    generation_tokps=float(m.group("generation")),
                    running_reqs=int(m.group("running")),
                    waiting_reqs=int(m.group("waiting")),
                    prefix_hit_rate_pct=float(m.group("prefix")),
                    request_completions_since_prev=pending_requests,
                    source_file=path.name,
                )
            )
            pending_requests = 0
            continue

        m = MFU_RE.search(line)
        if m:
            ts = parse_ts(m, year)
            if samples and abs((samples[-1].ts - ts).total_seconds()) <= 5:
                samples[-1].mfu_tflops_per_gpu = float(m.group("mfu_tflops"))
                samples[-1].bandwidth_gbps_per_gpu = float(m.group("bandwidth"))
            else:
                samples.append(
                    Sample(
                        ts=ts,
                        mfu_tflops_per_gpu=float(m.group("mfu_tflops")),
                        bandwidth_gbps_per_gpu=float(m.group("bandwidth")),
                        request_completions_since_prev=pending_requests,
                        source_file=path.name,
                    )
                )
                pending_requests = 0

    return samples


def overall_aggregate(samples: list[Sample]) -> Aggregate:
    return Aggregate(start=samples[0].ts, end=samples[-1].ts, samples=samples)


def bucket_by_time(samples: list[Sample], bucket_minutes: float) -> list[Aggregate]:
    if not samples:
        return []
    bucket_seconds = bucket_minutes * 60.0
    epoch = samples[0].ts
    buckets: dict[int, list[Sample]] = {}
    for s in samples:
        idx = math.floor((s.ts - epoch).total_seconds() / bucket_seconds)
        buckets.setdefault(idx, []).append(s)
    out = []
    for idx in sorted(buckets):
        start = epoch + dt.timedelta(seconds=idx * bucket_seconds)
        end = start + dt.timedelta(seconds=bucket_seconds)
        out.append(Aggregate(start=start, end=end, samples=buckets[idx]))
    return out


def bucket_by_requests(samples: list[Sample], requests_per_bucket: int) -> list[Aggregate]:
    if not samples:
        return []
    out: list[Aggregate] = []
    cur: list[Sample] = []
    reqs = 0
    start = samples[0].ts
    for s in samples:
        cur.append(s)
        reqs += s.request_completions_since_prev
        if reqs >= requests_per_bucket:
            out.append(Aggregate(start=start, end=s.ts, samples=cur))
            cur = []
            reqs = 0
            start = s.ts
    if cur:
        out.append(Aggregate(start=start, end=cur[-1].ts, samples=cur))
    return out


def fmt(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def main() -> None:
    args = parse_args()
    logs = discover_logs(args.logs_dir, args.log_file)
    all_samples: list[Sample] = []
    used_files = []
    for log in logs:
        samples = parse_worker_log(log, args.year)
        if samples:
            all_samples.extend(samples)
            used_files.append(log.name)

    all_samples.sort(key=lambda s: s.ts)
    if not all_samples:
        raise SystemExit(f"No vLLM metric samples found under {args.logs_dir}")

    if args.mode == "time":
        buckets = bucket_by_time(all_samples, args.bucket_minutes)
    else:
        buckets = bucket_by_requests(all_samples, args.requests_per_bucket)

    if args.show_files:
        print("files=")
        for name in used_files:
            print(f"  {name}")
        print()

    total = overall_aggregate(all_samples)
    print(
        "overall,"
        f"start={total.start.isoformat(sep=' ')},"
        f"end={total.end.isoformat(sep=' ')},"
        f"samples={len(total.samples)},"
        f"requests={total.requests},"
        f"prompt_tokps_mean={fmt(total.mean('prompt_tokps'), 1)},"
        f"generation_tokps_mean={fmt(total.mean('generation_tokps'), 1)},"
        f"prefix_hit_rate_pct_mean={fmt(total.mean('prefix_hit_rate_pct'), 1)},"
        f"running_reqs_mean={fmt(total.mean('running_reqs'), 1)},"
        f"waiting_reqs_mean={fmt(total.mean('waiting_reqs'), 1)},"
        f"mfu_tflops_per_gpu_weighted={fmt(total.weighted_mean('mfu_tflops_per_gpu', 'running_reqs'), 1)},"
        f"bandwidth_gbps_per_gpu_weighted={fmt(total.weighted_mean('bandwidth_gbps_per_gpu', 'running_reqs'), 1)}"
    )
    print()

    header = (
        "bucket_start,bucket_end,samples,requests,"
        "prompt_tokps_mean,generation_tokps_mean,prefix_hit_rate_pct_mean,"
        "running_reqs_mean,waiting_reqs_mean,mfu_tflops_per_gpu_weighted,"
        "bandwidth_gbps_per_gpu_weighted"
    )
    print(header)
    for b in buckets:
        row = [
            b.start.isoformat(sep=" "),
            b.end.isoformat(sep=" "),
            str(len(b.samples)),
            str(b.requests),
            fmt(b.mean("prompt_tokps"), 1),
            fmt(b.mean("generation_tokps"), 1),
            fmt(b.mean("prefix_hit_rate_pct"), 1),
            fmt(b.mean("running_reqs"), 1),
            fmt(b.mean("waiting_reqs"), 1),
            fmt(b.weighted_mean("mfu_tflops_per_gpu", "running_reqs"), 1),
            fmt(b.weighted_mean("bandwidth_gbps_per_gpu", "running_reqs"), 1),
        ]
        print(",".join(row))


if __name__ == "__main__":
    main()
