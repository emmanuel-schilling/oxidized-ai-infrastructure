"""Runs the benchmark suite and writes raw data and a summary to results/benchmarks/:

    uv run python reinforcement_learning/python/benchmark.py

Follows the benchmarking methodology: release builds, fixed seeds and episode
counts, warmup runs, repeated samples in rounds that rotate the variant order,
and a spread check that flags any training run or microbenchmark whose
coefficient of variation exceeds 5%. End-to-end timing needs hyperfine on PATH.

Each variant also runs with zero episodes, which times startup alone (process
launch, Python imports, building the environment), so training time is the
difference between the two. Startup's spread is reported but not checked: it
only enters the results through that difference, whose standard error
includes it. Native Rust trains for many more episodes than the others, so
its startup stays a small part of its run; the variants compare per step.
"""

import argparse
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import microbenchmarks
from microbenchmarks import Stats, summarize

REPO = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).with_name("train_frozen_lake.py")
MAX_COEFFICIENT_OF_VARIATION = 0.05
VARIANTS = ["rust_native", "rust_gymnasium", "python", "rust_from_python"]


def run_env() -> dict[str, str]:
    """The environment for child processes: the Rust programs embed Python, which
    finds the virtual environment's packages only through PYTHONPATH."""
    env = dict(os.environ)
    paths = [sysconfig.get_paths()["purelib"], env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    env["PYO3_PYTHON"] = sys.executable
    return env


def build_rust_example(env: dict[str, str]) -> Path:
    """Builds the training example in release mode and returns its executable,
    as Cargo reports it, so the platform's file name rules don't matter."""
    output = subprocess.run(
        ["cargo", "build", "--release", "--example", "train_frozen_lake"]
        + ["--message-format=json-render-diagnostics"],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in output.splitlines():
        message = json.loads(line)
        if message.get("target", {}).get("name") == "train_frozen_lake" and message.get(
            "executable"
        ):
            return Path(message["executable"])
    raise RuntimeError("cargo build reported no train_frozen_lake executable")


def repo_relative(path: Path | str) -> str:
    """`path` relative to the repository when it lies inside it, so recorded
    commands and settings don't carry the machine's absolute paths."""
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


def commands(rust_binary: Path, episodes: dict[str, int], seed: int) -> dict[str, list[str]]:
    """Each variant's training command for its episode count, and its
    zero-episode startup command, to run from the repository root."""
    binary, python, script = (
        repo_relative(path) for path in (rust_binary, sys.executable, TRAIN_SCRIPT)
    )
    programs = {
        "rust_native": [binary, "native"],
        "rust_gymnasium": [binary, "gymnasium"],
        "python": [python, script, "python"],
        "rust_from_python": [python, script, "rust"],
    }
    result = {}
    for variant in VARIANTS:
        result[f"{variant}/train"] = programs[variant] + [str(episodes[variant]), str(seed)]
        result[f"{variant}/startup"] = programs[variant] + ["0", str(seed)]
    return result


def count_steps(train_commands: dict[str, list[str]], env: dict[str, str]) -> dict[str, int]:
    """Runs each training command once and reads its step count. The seeds are
    fixed, so every timed run repeats exactly these steps."""
    steps = {}
    for name, command in train_commands.items():
        output = subprocess.run(
            command, cwd=REPO, env=env, check=True, capture_output=True, text=True
        )
        steps[name] = json.loads(output.stdout)["steps"]
    return steps


def rotated(names: list[str], round_index: int) -> list[str]:
    """`names` rotated by `round_index`, so each round starts with a different one."""
    shift = round_index % len(names)
    return names[shift:] + names[:shift]


def run_hyperfine(
    named_commands: dict[str, list[str]],
    order: list[str],
    warmup: int,
    runs: int,
    export: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    arguments = ["hyperfine", "--shell=none", "--warmup", str(warmup), "--runs", str(runs)]
    arguments += ["--export-json", str(export)]
    for name in order:
        arguments += ["--command-name", name, shlex.join(named_commands[name])]
    subprocess.run(arguments, cwd=REPO, env=env, check=True)
    return json.loads(export.read_text(encoding="utf-8"))


def merge_samples(rounds: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Pools each command's wall-clock times, in seconds, across hyperfine's JSON
    exports. Reads export schema 2, which hyperfine 2 writes."""
    samples: dict[str, list[float]] = {}
    for exported in rounds:
        if exported.get("schema_version") != 2:
            raise ValueError("expected hyperfine JSON schema 2; install hyperfine 2.0.0")
        for result in exported["results"]:
            times = []
            for measurement in result["measurements"]:
                wall_clock = measurement["time_wall_clock"]
                if wall_clock["unit"] != "second":
                    raise ValueError(f"unexpected time unit {wall_clock['unit']!r}")
                times.append(wall_clock["value"])
            samples.setdefault(result["name"], []).extend(times)
    return samples


def read_criterion(criterion_dir: Path) -> dict[str, Stats]:
    """Per-iteration seconds for each benchmark, from criterion's raw samples."""
    results = {}
    for benchmark in sorted(criterion_dir.rglob("new/benchmark.json")):
        name = json.loads(benchmark.read_text(encoding="utf-8"))["full_id"]
        sample = json.loads(benchmark.with_name("sample.json").read_text(encoding="utf-8"))
        nanoseconds = [t / n for t, n in zip(sample["times"], sample["iters"], strict=True)]
        results[name] = summarize([t * 1e-9 for t in nanoseconds])
    return results


def run_criterion(env: dict[str, str], output: Path) -> dict[str, Stats]:
    subprocess.run(
        ["cargo", "bench", "--bench", "agent", "--bench", "environment", "--", "--noplot"],
        cwd=REPO,
        env=env,
        check=True,
    )
    criterion_dir = REPO / "target" / "criterion"
    shutil.copytree(criterion_dir, output / "criterion", dirs_exist_ok=True)
    return read_criterion(criterion_dir)


def training_summary(
    samples: dict[str, list[float]], steps: dict[str, int]
) -> dict[str, dict[str, float]]:
    """Training time with startup subtracted, per run and per step, for each
    variant, with standard errors that combine both measurements' spread."""
    summary = {}
    for variant in VARIANTS:
        train = summarize(samples[f"{variant}/train"])
        startup = summarize(samples[f"{variant}/startup"])
        seconds = train.mean - startup.mean
        standard_error = math.hypot(
            train.std_dev / math.sqrt(train.samples), startup.std_dev / math.sqrt(startup.samples)
        )
        variant_steps = steps[f"{variant}/train"]
        summary[variant] = {
            "steps": variant_steps,
            "training_seconds": seconds,
            "training_seconds_standard_error": standard_error,
            "seconds_per_step": seconds / variant_steps,
            "seconds_per_step_standard_error": standard_error / variant_steps,
        }
    return summary


def too_noisy(stats: dict[str, Stats]) -> list[str]:
    """The checked measurements whose spread exceeds the limit: everything but
    startup, whose spread is carried into the training-time standard error."""
    return [
        name
        for name, s in stats.items()
        if not name.endswith("/startup")
        and s.coefficient_of_variation > MAX_COEFFICIENT_OF_VARIATION
    ]


def command_output(arguments: list[str]) -> str | None:
    try:
        return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_and_memory() -> tuple[str | None, int | None]:
    """The CPU model and the RAM in bytes, where the platform reports them.

    PLATFORM: macOS, Linux — read from sysctl and /proc; Windows would need
    the registry or WMI, so it reports both as unknown.
    """
    if sys.platform == "darwin":
        memory = command_output(["sysctl", "-n", "hw.memsize"])
        cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        return cpu, int(memory) if memory else None
    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        cpu = next(
            (line.split(":", 1)[1].strip() for line in cpuinfo if "model name" in line), None
        )
        kilobytes = next((line.split()[1] for line in meminfo if line.startswith("MemTotal")), None)
        return cpu, int(kilobytes) * 1024 if kilobytes else None
    return None, None


def machine_description() -> dict[str, Any]:
    cpu, memory = cpu_and_memory()
    status = command_output(["git", "-C", str(REPO), "status", "--porcelain"])
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "memory_bytes": memory,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "rustc": command_output(["rustc", "--version"]),
        "hyperfine": command_output(["hyperfine", "--version"]),
        "commit": command_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "uncommitted_changes": bool(status),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Q-learning benchmark suite.")
    parser.add_argument(
        "--episodes", type=int, default=20_000, help="for the Python-using variants"
    )
    parser.add_argument(
        "--native-episodes",
        type=int,
        default=1_000_000,
        help="for native Rust, which runs 20,000 episodes in about 20 ms",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3, help="hyperfine rounds")
    parser.add_argument("--runs", type=int, default=4, help="timed runs per command per round")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per command per round")
    parser.add_argument("--micro-repeats", type=int, default=30, help="timeit batches per case")
    parser.add_argument("--output", type=Path, help="default: results/benchmarks/<UTC time>")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("hyperfine") is None:
        print("hyperfine not found; install it with:", file=sys.stderr)
        print("  cargo install --locked hyperfine --version 2.0.0", file=sys.stderr)
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or REPO / "results" / "benchmarks" / stamp
    (output / "hyperfine").mkdir(parents=True, exist_ok=True)
    env = run_env()

    episodes = {variant: args.episodes for variant in VARIANTS} | {
        "rust_native": args.native_episodes
    }
    named = commands(build_rust_example(env), episodes, args.seed)
    steps = count_steps({name: c for name, c in named.items() if name.endswith("/train")}, env)
    rounds = [
        run_hyperfine(
            named,
            rotated(list(named), index),
            args.warmup,
            args.runs,
            output / "hyperfine" / f"round_{index}.json",
            env,
        )
        for index in range(args.rounds)
    ]
    end_to_end = {name: summarize(times) for name, times in merge_samples(rounds).items()}
    micro = {**run_criterion(env, output), **microbenchmarks.run(args.micro_repeats)}

    noisy = too_noisy(end_to_end) + too_noisy(micro)
    summary = {
        "settings": vars(args) | {"output": repo_relative(output)},
        "machine": machine_description(),
        "end_to_end": {name: s.to_dict() for name, s in end_to_end.items()},
        "training": training_summary(merge_samples(rounds), steps),
        "microbenchmarks": {name: s.to_dict() for name, s in micro.items()},
        "too_noisy": noisy,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nResults in {output}")
    for variant, row in summary["training"].items():
        print(
            f"  {variant:18} {row['training_seconds']:8.3f} s  {row['seconds_per_step']:.3e} s/step"
        )
    if noisy:
        print(
            f"Coefficient of variation above {MAX_COEFFICIENT_OF_VARIATION:.0%}: "
            + ", ".join(noisy)
            + ". Find the source of noise and run again.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
