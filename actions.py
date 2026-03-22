"""
noosphere/actions.py
====================
Action Spaces, Executors, and the Plan→Act Bridge

Shell vocabulary — v1.5.0
--------------------------
Maximum coverage across every Linux digital domain, organised in tiers.
The world model learns consequences domain by domain as experience accumulates.

Tier 0  SAFE_READ       — read-only exploration, no side effects, always available
Tier 1  SAFE_WRITE      — creates/modifies files, reversible, low risk
Tier 2  PROCESS         — process creation, signals, scheduling
Tier 3  NETWORK         — network I/O, requires connectivity
Tier 4  SYSTEM          — system-level changes, package installs, config
Tier 5  DESTRUCTIVE     — irreversible operations, requires explicit unlock

The ShellExecutor enforces tiers via allow_tiers. Start with Tier 0 and
expand as the world model demonstrates reliable consequence prediction.

Feature encoding — v1.5.0
--------------------------
Command output is encoded as a 32-dim feature vector (was 10-dim).
New dimensions capture: numeric values parsed from output, file counts,
network addresses found, process counts, error keyword detection,
output structure (table vs. free text vs. JSON), and change detection
via rolling diff against previous output for the same command.

DigitalStateObservation
-----------------------
Structured snapshot of the Linux environment at each step.
Encodes: filesystem state, running processes, open ports, git state,
active Python envs, GPU utilisation, memory pressure.
Passed as obs["structured"] so the world model trains to predict
how actions change digital state — not just scalar reward.
"""

import json
import math
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ── Tier constants ─────────────────────────────────────────────────────────────


class Tier:
    SAFE_READ = 0
    SAFE_WRITE = 1
    PROCESS = 2
    NETWORK = 3
    SYSTEM = 4
    DESTRUCTIVE = 5


# ── Action vocabulary ─────────────────────────────────────────────────────────


@dataclass
class Action:
    index: int
    name: str
    description: str
    task_type: str = "multiclass"
    tier: int = Tier.SAFE_READ
    payload: Any = None


@dataclass
class ActionSpace:
    name: str
    actions: List[Action] = field(default_factory=list)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx: int) -> Action:
        return self.actions[idx]

    def add(
        self,
        name: str,
        description: str,
        task_type: str = "multiclass",
        tier: int = Tier.SAFE_READ,
        payload: Any = None,
    ) -> "ActionSpace":
        self.actions.append(
            Action(
                index=len(self.actions),
                name=name,
                description=description,
                task_type=task_type,
                tier=tier,
                payload=payload,
            )
        )
        return self

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def by_tier(self, max_tier: int) -> "ActionSpace":
        """Return a new ActionSpace containing only actions up to max_tier."""
        sp = ActionSpace(f"{self.name}_t{max_tier}")
        for a in self.actions:
            if a.tier <= max_tier:
                sp.actions.append(
                    Action(
                        index=len(sp.actions),
                        name=a.name,
                        description=a.description,
                        task_type=a.task_type,
                        tier=a.tier,
                        payload=a.payload,
                    )
                )
        return sp

    def describe(self) -> str:
        tier_names = {
            0: "READ",
            1: "WRITE",
            2: "PROC",
            3: "NET",
            4: "SYS",
            5: "DESTROY",
        }
        lines = [f"ActionSpace: {self.name}  ({self.n_actions} actions)"]
        for t in range(6):
            acts = [a for a in self.actions if a.tier == t]
            if acts:
                lines.append(f"  ── Tier {t} [{tier_names[t]}] ──")
                for a in acts:
                    lines.append(f"    [{a.index:3d}] {a.name:<32} {a.description}")
        return "\n".join(lines)


# ── Built-in action spaces ─────────────────────────────────────────────────────


def make_apparatus_space() -> ActionSpace:
    return (
        ActionSpace("apparatus")
        .add(
            "shoulder_yaw_cw",
            "Rotate shoulder CW",
            tier=0,
            payload={"joint": 0, "delta_deg": +5.0},
        )
        .add(
            "shoulder_yaw_ccw",
            "Rotate shoulder CCW",
            tier=0,
            payload={"joint": 0, "delta_deg": -5.0},
        )
        .add(
            "shoulder_pitch_up",
            "Raise shoulder",
            tier=0,
            payload={"joint": 1, "delta_deg": +5.0},
        )
        .add(
            "shoulder_pitch_dn",
            "Lower shoulder",
            tier=0,
            payload={"joint": 1, "delta_deg": -5.0},
        )
        .add(
            "elbow_extend",
            "Extend elbow",
            tier=0,
            payload={"joint": 3, "delta_deg": +8.0},
        )
        .add(
            "elbow_flex", "Flex elbow", tier=0, payload={"joint": 3, "delta_deg": -8.0}
        )
        .add(
            "wrist_flex", "Flex wrist", tier=0, payload={"joint": 4, "delta_deg": +8.0}
        )
        .add(
            "wrist_extend",
            "Extend wrist",
            tier=0,
            payload={"joint": 4, "delta_deg": -8.0},
        )
    )


def make_binary_space(
    positive_action: str,
    negative_action: str,
    pos_payload: Any = None,
    neg_payload: Any = None,
) -> ActionSpace:
    return (
        ActionSpace("binary")
        .add(
            positive_action, f"Execute: {positive_action}", tier=0, payload=pos_payload
        )
        .add(negative_action, f"Do not: {positive_action}", tier=0, payload=neg_payload)
    )


def make_shell_space(working_dir: str = ".") -> ActionSpace:
    """
    Maximum-coverage Linux shell vocabulary.

    Structure: every action has a tier, a name, a description, and a payload
    with the shell command. Commands that fail gracefully (2>/dev/null, || echo)
    are preferred so the world model sees consistent structure even on error.

    The caller can restrict to safe tiers via space.by_tier(Tier.SAFE_READ).
    """
    T = Tier
    sp = ActionSpace("shell")

    # ── Tier 0: SAFE_READ ─────────────────────────────────────────────────────
    # No side effects. The world model trains on these first.

    # Identity / navigation
    sp.add("wait", "No-op / wait one step", T.SAFE_READ, payload={"cmd": None})
    sp.add("pwd", "Print working directory", T.SAFE_READ, payload={"cmd": "pwd"})
    sp.add("whoami", "Current user", T.SAFE_READ, payload={"cmd": "whoami"})
    sp.add("id", "User/group IDs", T.SAFE_READ, payload={"cmd": "id"})
    sp.add("hostname", "System hostname", T.SAFE_READ, payload={"cmd": "hostname"})
    sp.add("date", "Current date/time", T.SAFE_READ, payload={"cmd": "date -Iseconds"})
    sp.add("uptime", "System uptime and load", T.SAFE_READ, payload={"cmd": "uptime"})

    # Filesystem exploration
    sp.add(
        "ls",
        "List directory (long)",
        T.SAFE_READ,
        payload={"cmd": "ls -lah --color=never"},
    )
    sp.add(
        "ls_all",
        "List including hidden files",
        T.SAFE_READ,
        payload={"cmd": "ls -lah --color=never -A"},
    )
    sp.add(
        "ls_tree",
        "Directory tree (depth 3)",
        T.SAFE_READ,
        payload={"cmd": "find . -maxdepth 3 | sort"},
    )
    sp.add(
        "ls_recent",
        "10 most recently modified files",
        T.SAFE_READ,
        payload={
            "cmd": "find . -maxdepth 3 -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -10"
        },
    )
    sp.add(
        "disk_usage",
        "Disk usage by directory",
        T.SAFE_READ,
        payload={"cmd": "df -h . && du -sh * 2>/dev/null | sort -h | tail -20"},
    )
    sp.add(
        "file_type",
        "Detect type of files in dir",
        T.SAFE_READ,
        payload={"cmd": "file * 2>/dev/null | head -30"},
    )
    sp.add(
        "cat_readme",
        "Read README",
        T.SAFE_READ,
        payload={
            "cmd": "cat README.md 2>/dev/null || cat readme.txt 2>/dev/null || echo 'no README'"
        },
    )
    sp.add(
        "cat_makefile",
        "Read Makefile",
        T.SAFE_READ,
        payload={"cmd": "cat Makefile 2>/dev/null || echo 'no Makefile'"},
    )
    sp.add(
        "cat_dockerfile",
        "Read Dockerfile",
        T.SAFE_READ,
        payload={"cmd": "cat Dockerfile 2>/dev/null || echo 'no Dockerfile'"},
    )
    sp.add(
        "head_file",
        "First 50 lines of largest file",
        T.SAFE_READ,
        payload={"cmd": "ls -S | head -1 | xargs head -50 2>/dev/null"},
    )

    # File search
    sp.add(
        "find_python",
        "Find Python files",
        T.SAFE_READ,
        payload={"cmd": "find . -name '*.py' -maxdepth 5 2>/dev/null | head -30"},
    )
    sp.add(
        "find_config",
        "Find config files",
        T.SAFE_READ,
        payload={
            "cmd": "find . \\( -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.ini' -o -name '*.conf' \\) -maxdepth 4 2>/dev/null | head -30"
        },
    )
    sp.add(
        "find_logs",
        "Find log files",
        T.SAFE_READ,
        payload={
            "cmd": "find . /var/log -name '*.log' -maxdepth 4 2>/dev/null | head -20"
        },
    )
    sp.add(
        "find_large",
        "Files >10MB",
        T.SAFE_READ,
        payload={"cmd": "find . -size +10M -type f 2>/dev/null | head -20"},
    )
    sp.add(
        "find_recent",
        "Files changed in last 10 min",
        T.SAFE_READ,
        payload={"cmd": "find . -mmin -10 -type f 2>/dev/null | head -20"},
    )

    # Text processing (read-only)
    sp.add(
        "wc_dir",
        "Count lines/words/bytes in dir",
        T.SAFE_READ,
        payload={
            "cmd": "find . -name '*.py' -o -name '*.txt' 2>/dev/null | xargs wc -l 2>/dev/null | sort -n | tail -20"
        },
    )
    sp.add(
        "grep_errors",
        "Grep ERROR/WARN in logs",
        T.SAFE_READ,
        payload={
            "cmd": "grep -r 'ERROR\\|WARN\\|CRITICAL\\|Exception' . --include='*.log' -l 2>/dev/null | head -10"
        },
    )
    sp.add(
        "grep_todos",
        "Grep TODO/FIXME in code",
        T.SAFE_READ,
        payload={
            "cmd": "grep -r 'TODO\\|FIXME\\|HACK\\|XXX' . --include='*.py' --include='*.js' --include='*.ts' -n 2>/dev/null | head -20"
        },
    )
    sp.add(
        "diff_git",
        "Diff against last commit",
        T.SAFE_READ,
        payload={"cmd": "git diff 2>/dev/null | head -60 || echo 'not a git repo'"},
    )

    # Process state
    sp.add(
        "ps_all",
        "All running processes",
        T.SAFE_READ,
        payload={"cmd": "ps aux --sort=-%cpu | head -25"},
    )
    sp.add(
        "ps_mem",
        "Processes by memory",
        T.SAFE_READ,
        payload={"cmd": "ps aux --sort=-%mem | head -15"},
    )
    sp.add(
        "pstree",
        "Process tree",
        T.SAFE_READ,
        payload={
            "cmd": "pstree -p 2>/dev/null | head -30 || ps --forest -e 2>/dev/null | head -30"
        },
    )
    sp.add(
        "lsof_net",
        "Open network connections",
        T.SAFE_READ,
        payload={
            "cmd": "ss -tunap 2>/dev/null | head -25 || netstat -tunap 2>/dev/null | head -25"
        },
    )
    sp.add(
        "lsof_files",
        "Open file descriptors (top procs)",
        T.SAFE_READ,
        payload={
            "cmd": "lsof -n 2>/dev/null | awk '{print $1}' | sort | uniq -c | sort -rn | head -15"
        },
    )

    # System info
    sp.add(
        "mem_info",
        "Memory usage",
        T.SAFE_READ,
        payload={
            "cmd": "free -h && cat /proc/meminfo 2>/dev/null | grep -E 'MemTotal|MemFree|MemAvail|Buffers|Cached|SwapTotal|SwapFree'"
        },
    )
    sp.add(
        "cpu_info",
        "CPU info and count",
        T.SAFE_READ,
        payload={
            "cmd": "lscpu 2>/dev/null | grep -E 'CPU\\(s\\)|Model|MHz|Cache' | head -10 || nproc"
        },
    )
    sp.add(
        "load_avg",
        "Load average (1m/5m/15m)",
        T.SAFE_READ,
        payload={"cmd": "cat /proc/loadavg && uptime"},
    )
    sp.add(
        "uname",
        "Kernel and OS info",
        T.SAFE_READ,
        payload={"cmd": "uname -a && cat /etc/os-release 2>/dev/null | head -6"},
    )
    sp.add(
        "mounts",
        "Mounted filesystems",
        T.SAFE_READ,
        payload={
            "cmd": "mount | grep -v 'proc\\|sysfs\\|devtmpfs\\|cgroup\\|tmpfs' | head -20"
        },
    )
    sp.add(
        "gpu_info",
        "GPU status (if available)",
        T.SAFE_READ,
        payload={
            "cmd": "nvidia-smi 2>/dev/null || rocm-smi 2>/dev/null || echo 'no GPU detected'"
        },
    )
    sp.add(
        "temp_sensors",
        "Hardware temperature sensors",
        T.SAFE_READ,
        payload={
            "cmd": "sensors 2>/dev/null || cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | awk '{print $1/1000 \"°C\"}' | head -8 || echo 'no sensors'"
        },
    )
    sp.add(
        "cpu_usage",
        "Instantaneous CPU usage per core",
        T.SAFE_READ,
        payload={
            "cmd": "mpstat -P ALL 1 1 2>/dev/null | tail -n +4 | head -20 || top -bn1 | head -20"
        },
    )
    sp.add(
        "io_stats",
        "I/O statistics",
        T.SAFE_READ,
        payload={
            "cmd": "iostat -xz 1 1 2>/dev/null | head -30 || vmstat 1 2 2>/dev/null | tail -5"
        },
    )
    sp.add(
        "vm_stats",
        "Virtual memory statistics",
        T.SAFE_READ,
        payload={"cmd": "vmstat 1 3 2>/dev/null"},
    )
    sp.add(
        "interrupts",
        "Hardware interrupts",
        T.SAFE_READ,
        payload={"cmd": "cat /proc/interrupts | head -20"},
    )

    # Hardware
    sp.add(
        "lspci",
        "PCI devices",
        T.SAFE_READ,
        payload={"cmd": "lspci 2>/dev/null | head -20 || echo 'lspci not available'"},
    )
    sp.add(
        "lsusb",
        "USB devices",
        T.SAFE_READ,
        payload={"cmd": "lsusb 2>/dev/null | head -20 || echo 'lsusb not available'"},
    )
    sp.add(
        "block_devices",
        "Block devices and partitions",
        T.SAFE_READ,
        payload={"cmd": "lsblk 2>/dev/null | head -20"},
    )

    # Network read
    sp.add(
        "ip_addr",
        "Network interfaces and addresses",
        T.SAFE_READ,
        payload={
            "cmd": "ip addr show 2>/dev/null | head -30 || ifconfig 2>/dev/null | head -30"
        },
    )
    sp.add(
        "ip_route",
        "Routing table",
        T.SAFE_READ,
        payload={"cmd": "ip route show 2>/dev/null || route -n 2>/dev/null"},
    )
    sp.add(
        "dns_info",
        "DNS resolver config",
        T.SAFE_READ,
        payload={
            "cmd": "cat /etc/resolv.conf 2>/dev/null && cat /etc/hosts 2>/dev/null | grep -v '^#' | head -10"
        },
    )
    sp.add(
        "listening_ports",
        "Listening ports",
        T.SAFE_READ,
        payload={"cmd": "ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null"},
    )
    sp.add(
        "connections",
        "Active TCP connections",
        T.SAFE_READ,
        payload={"cmd": "ss -tnp 2>/dev/null | head -20"},
    )

    # Timezone / locale
    sp.add(
        "timezone",
        "Current timezone",
        T.SAFE_READ,
        payload={"cmd": "timedatectl 2>/dev/null || date +'%Z %z'"},
    )
    sp.add(
        "locale", "Locale settings", T.SAFE_READ, payload={"cmd": "locale 2>/dev/null"}
    )

    # Users / security
    sp.add(
        "who", "Logged-in users", T.SAFE_READ, payload={"cmd": "who && last | head -10"}
    )
    sp.add(
        "groups", "Current user's groups", T.SAFE_READ, payload={"cmd": "groups && id"}
    )
    sp.add(
        "sudo_check",
        "Available sudo commands",
        T.SAFE_READ,
        payload={"cmd": "sudo -l 2>/dev/null | head -20 || echo 'no sudo'"},
    )

    # Crypto / hashing
    sp.add(
        "hash_files",
        "SHA256 of files in cwd",
        T.SAFE_READ,
        payload={
            "cmd": "find . -maxdepth 1 -type f | xargs sha256sum 2>/dev/null | head -20"
        },
    )

    # Environment
    sp.add(
        "env_vars",
        "Environment variables",
        T.SAFE_READ,
        payload={"cmd": "env | sort | head -40"},
    )
    sp.add(
        "path_inspect",
        "Inspect PATH entries",
        T.SAFE_READ,
        payload={"cmd": "echo $PATH | tr ':' '\\n'"},
    )
    sp.add(
        "shell_info",
        "Current shell and version",
        T.SAFE_READ,
        payload={"cmd": "echo $SHELL && $SHELL --version 2>/dev/null | head -2"},
    )
    sp.add("ulimits", "Resource limits", T.SAFE_READ, payload={"cmd": "ulimit -a"})

    # Python / toolchain
    sp.add(
        "python_version",
        "Python version",
        T.SAFE_READ,
        payload={"cmd": "python3 --version 2>&1 && which python3"},
    )
    sp.add(
        "python_packages",
        "Installed Python packages",
        T.SAFE_READ,
        payload={
            "cmd": "pip list 2>/dev/null | head -30 || pip3 list 2>/dev/null | head -30"
        },
    )
    sp.add(
        "venv_check",
        "Active virtualenv / conda env",
        T.SAFE_READ,
        payload={
            "cmd": "echo VIRTUAL_ENV=$VIRTUAL_ENV && echo CONDA_ENV=$CONDA_DEFAULT_ENV && python3 -c 'import sys; print(sys.prefix)'"
        },
    )
    sp.add(
        "node_version",
        "Node.js version",
        T.SAFE_READ,
        payload={
            "cmd": "node --version 2>/dev/null && npm --version 2>/dev/null || echo 'node not found'"
        },
    )
    sp.add(
        "rust_version",
        "Rust/cargo version",
        T.SAFE_READ,
        payload={
            "cmd": "rustc --version 2>/dev/null && cargo --version 2>/dev/null || echo 'rust not found'"
        },
    )
    sp.add(
        "go_version",
        "Go version",
        T.SAFE_READ,
        payload={"cmd": "go version 2>/dev/null || echo 'go not found'"},
    )
    sp.add(
        "gcc_version",
        "GCC version",
        T.SAFE_READ,
        payload={"cmd": "gcc --version 2>/dev/null | head -2 || echo 'gcc not found'"},
    )
    sp.add(
        "make_targets",
        "Available make targets",
        T.SAFE_READ,
        payload={
            "cmd": "make -qp 2>/dev/null | grep -E '^[a-zA-Z0-9_-]+:' | head -20 || echo 'no Makefile'"
        },
    )
    sp.add(
        "nix_info",
        "Nix store and channel info",
        T.SAFE_READ,
        payload={
            "cmd": "nix-info -m 2>/dev/null || nix --version 2>/dev/null || echo 'nix not found'"
        },
    )

    # Git (read)
    sp.add(
        "git_status",
        "Git working tree status",
        T.SAFE_READ,
        payload={"cmd": "git status 2>/dev/null || echo 'not a git repo'"},
    )
    sp.add(
        "git_log",
        "Recent git commits",
        T.SAFE_READ,
        payload={
            "cmd": "git log --oneline --graph -15 2>/dev/null || echo 'not a git repo'"
        },
    )
    sp.add(
        "git_branches",
        "Local and remote branches",
        T.SAFE_READ,
        payload={"cmd": "git branch -av 2>/dev/null || echo 'not a git repo'"},
    )
    sp.add(
        "git_remotes",
        "Git remote URLs",
        T.SAFE_READ,
        payload={"cmd": "git remote -v 2>/dev/null || echo 'not a git repo'"},
    )
    sp.add(
        "git_stash",
        "Git stash list",
        T.SAFE_READ,
        payload={"cmd": "git stash list 2>/dev/null || echo 'not a git repo'"},
    )
    sp.add(
        "git_blame",
        "Last modifier of each file",
        T.SAFE_READ,
        payload={
            "cmd": "git log --format='%an %ar %s' -- . 2>/dev/null | head -10 || echo 'not a git repo'"
        },
    )

    # Docker / containers (read)
    sp.add(
        "docker_ps",
        "Running Docker containers",
        T.SAFE_READ,
        payload={
            "cmd": "docker ps 2>/dev/null | head -15 || echo 'docker not running'"
        },
    )
    sp.add(
        "docker_images",
        "Docker images",
        T.SAFE_READ,
        payload={
            "cmd": "docker images 2>/dev/null | head -15 || echo 'docker not running'"
        },
    )
    sp.add(
        "docker_volumes",
        "Docker volumes",
        T.SAFE_READ,
        payload={"cmd": "docker volume ls 2>/dev/null || echo 'docker not running'"},
    )
    sp.add(
        "docker_networks",
        "Docker networks",
        T.SAFE_READ,
        payload={"cmd": "docker network ls 2>/dev/null || echo 'docker not running'"},
    )
    sp.add(
        "systemd_status",
        "Systemd service status",
        T.SAFE_READ,
        payload={
            "cmd": "systemctl status 2>/dev/null | head -25 || echo 'systemd not available'"
        },
    )
    sp.add(
        "systemd_failed",
        "Failed systemd services",
        T.SAFE_READ,
        payload={
            "cmd": "systemctl --failed 2>/dev/null | head -15 || echo 'systemd not available'"
        },
    )
    sp.add(
        "journal_errors",
        "Recent journal errors",
        T.SAFE_READ,
        payload={
            "cmd": "journalctl -p err -n 30 --no-pager 2>/dev/null | head -30 || echo 'journalctl not available'"
        },
    )

    # Databases (read-only queries)
    sp.add(
        "sqlite_tables",
        "SQLite tables in cwd",
        T.SAFE_READ,
        payload={
            "cmd": "find . -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' 2>/dev/null | head -5 | xargs -I{} sh -c 'echo \"=== {} ===\"; sqlite3 {} .tables 2>/dev/null'"
        },
    )
    sp.add(
        "psql_tables",
        "PostgreSQL tables (current DB)",
        T.SAFE_READ,
        payload={
            "cmd": "psql -c '\\dt' 2>/dev/null | head -20 || echo 'psql not available or not connected'"
        },
    )

    # Logs (read)
    sp.add(
        "tail_syslog",
        "Last 20 syslog lines",
        T.SAFE_READ,
        payload={
            "cmd": "tail -20 /var/log/syslog 2>/dev/null || journalctl -n 20 --no-pager 2>/dev/null | head -20 || echo 'no syslog'"
        },
    )
    sp.add(
        "tail_auth",
        "Last 20 auth.log lines",
        T.SAFE_READ,
        payload={
            "cmd": "tail -20 /var/log/auth.log 2>/dev/null || journalctl -u ssh -n 20 --no-pager 2>/dev/null | head -20 || echo 'no auth log'"
        },
    )
    sp.add(
        "cron_jobs",
        "Cron jobs for current user",
        T.SAFE_READ,
        payload={
            "cmd": "crontab -l 2>/dev/null && cat /etc/cron.d/* 2>/dev/null | head -20 || echo 'no cron jobs'"
        },
    )
    sp.add(
        "at_jobs",
        "Scheduled at jobs",
        T.SAFE_READ,
        payload={"cmd": "atq 2>/dev/null || echo 'at not available'"},
    )

    # AI / ML tooling (read)
    sp.add(
        "nvidia_smi",
        "GPU utilisation (NVIDIA)",
        T.SAFE_READ,
        payload={
            "cmd": "nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'no NVIDIA GPU'"
        },
    )
    sp.add(
        "torch_devices",
        "Available PyTorch devices",
        T.SAFE_READ,
        payload={
            "cmd": 'python3 -c \'import torch; print("CUDA:",torch.cuda.is_available(),"Devices:",torch.cuda.device_count(),"MPS:",torch.backends.mps.is_available())\' 2>/dev/null || echo \'torch not installed\''
        },
    )
    sp.add(
        "ml_procs",
        "Running ML training processes",
        T.SAFE_READ,
        payload={
            "cmd": "ps aux | grep -E 'python|torch|train|jupyter' | grep -v grep | head -10"
        },
    )

    # IPC / sockets
    sp.add(
        "unix_sockets",
        "Unix domain sockets",
        T.SAFE_READ,
        payload={"cmd": "ss -xlp 2>/dev/null | head -20"},
    )
    sp.add(
        "named_pipes",
        "Named pipes (FIFOs)",
        T.SAFE_READ,
        payload={"cmd": "find /tmp /var/run . -type p 2>/dev/null | head -15"},
    )
    sp.add(
        "shared_mem",
        "System V shared memory",
        T.SAFE_READ,
        payload={"cmd": "ipcs -m 2>/dev/null | head -15"},
    )
    sp.add(
        "dbus_services",
        "Running D-Bus services",
        T.SAFE_READ,
        payload={
            "cmd": "busctl list 2>/dev/null | head -15 || dbus-send --system --type=method_call --print-reply --dest=org.freedesktop.DBus /org/freedesktop/DBus org.freedesktop.DBus.ListNames 2>/dev/null | head -20 || echo 'dbus not available'"
        },
    )

    # Compiler / build output (read)
    sp.add(
        "build_output",
        "Recent build/compile output",
        T.SAFE_READ,
        payload={
            "cmd": "find . -name 'build.log' -o -name 'compile.log' 2>/dev/null | head -3 | xargs tail -30 2>/dev/null || echo 'no build logs'"
        },
    )
    sp.add(
        "cargo_check",
        "Cargo check (Rust, read-only)",
        T.SAFE_READ,
        payload={
            "cmd": "cargo check --message-format short 2>/dev/null | head -20 || echo 'no Cargo.toml'"
        },
    )
    sp.add(
        "python_lint",
        "Pylint/flake8 on changed files",
        T.SAFE_READ,
        payload={
            "cmd": "git diff --name-only 2>/dev/null | grep '\\.py$' | head -5 | xargs flake8 --max-line-length=120 2>/dev/null | head -20 || echo 'no changed py files or flake8 not installed'"
        },
    )

    # Security / crypto (read)
    sp.add(
        "ssl_certs",
        "SSL certificates (expiry)",
        T.SAFE_READ,
        payload={
            "cmd": "find /etc/ssl /etc/letsencrypt . -name '*.pem' -o -name '*.crt' 2>/dev/null | head -5 | xargs -I{} sh -c 'echo {}; openssl x509 -noout -dates -in {} 2>/dev/null' | head -20"
        },
    )
    sp.add(
        "gpg_keys",
        "GPG keys",
        T.SAFE_READ,
        payload={
            "cmd": "gpg --list-keys 2>/dev/null | head -20 || echo 'gpg not available'"
        },
    )
    sp.add(
        "ssh_keys",
        "SSH public keys",
        T.SAFE_READ,
        payload={
            "cmd": "ls ~/.ssh/ 2>/dev/null && cat ~/.ssh/authorized_keys 2>/dev/null | head -5 || echo 'no ssh dir'"
        },
    )
    sp.add(
        "firewall_rules",
        "Firewall rules (iptables/nft)",
        T.SAFE_READ,
        payload={
            "cmd": "iptables -L -n 2>/dev/null | head -20 || nft list ruleset 2>/dev/null | head -20 || echo 'no firewall info'"
        },
    )

    # ── Tier 1: SAFE_WRITE ────────────────────────────────────────────────────
    # Creates or modifies files. Reversible with git or rm.

    sp.add(
        "mkdir_tmp",
        "Create temp working directory",
        T.SAFE_WRITE,
        payload={"cmd": "mkdir -p /tmp/noosphere_workspace_$(date +%s)"},
    )
    sp.add(
        "touch_marker",
        "Touch a marker file",
        T.SAFE_WRITE,
        payload={"cmd": "touch .noosphere_last_run"},
    )
    sp.add(
        "write_log",
        "Append timestamp to noosphere.log",
        T.SAFE_WRITE,
        payload={"cmd": 'echo "$(date -Iseconds) noosphere step" >> noosphere.log'},
    )
    sp.add(
        "git_add_all",
        "Stage all changes",
        T.SAFE_WRITE,
        payload={"cmd": "git add -A 2>/dev/null || echo 'not a git repo'"},
    )
    sp.add(
        "git_commit",
        "Commit staged changes",
        T.SAFE_WRITE,
        payload={
            "cmd": "git commit -m 'noosphere: auto-commit' 2>/dev/null || echo 'nothing to commit'"
        },
    )
    sp.add(
        "git_stash_push",
        "Stash working changes",
        T.SAFE_WRITE,
        payload={"cmd": "git stash 2>/dev/null || echo 'nothing to stash'"},
    )
    sp.add(
        "pip_install_r",
        "pip install -r requirements.txt",
        T.SAFE_WRITE,
        payload={
            "cmd": "pip install -r requirements.txt 2>&1 | tail -5 || echo 'no requirements.txt'"
        },
    )
    sp.add(
        "make_build",
        "Run make (default target)",
        T.SAFE_WRITE,
        payload={"cmd": "make 2>&1 | tail -20 || echo 'no Makefile'"},
    )
    sp.add(
        "cargo_build",
        "cargo build",
        T.SAFE_WRITE,
        payload={"cmd": "cargo build 2>&1 | tail -20 || echo 'no Cargo.toml'"},
    )
    sp.add(
        "python_run",
        "Run main.py or __main__.py",
        T.SAFE_WRITE,
        payload={
            "cmd": "timeout 30 python3 main.py 2>&1 | tail -20 || timeout 30 python3 -m . 2>&1 | tail -20 || echo 'no runnable entry point'"
        },
    )
    sp.add(
        "pytest_run",
        "Run test suite with pytest",
        T.SAFE_WRITE,
        payload={
            "cmd": "timeout 60 python3 -m pytest -x -q 2>&1 | tail -20 || echo 'pytest not available or no tests'"
        },
    )
    sp.add(
        "go_build",
        "go build",
        T.SAFE_WRITE,
        payload={"cmd": "go build ./... 2>&1 | head -20 || echo 'no go modules'"},
    )
    sp.add(
        "npm_install",
        "npm install",
        T.SAFE_WRITE,
        payload={"cmd": "npm install 2>&1 | tail -10 || echo 'no package.json'"},
    )
    sp.add(
        "npm_build",
        "npm run build",
        T.SAFE_WRITE,
        payload={
            "cmd": "npm run build 2>&1 | tail -20 || echo 'no package.json or no build script'"
        },
    )

    # ── Tier 2: PROCESS ───────────────────────────────────────────────────────
    # Creates, signals, or manages processes.

    sp.add(
        "kill_zombie",
        "Kill zombie/defunct processes",
        T.PROCESS,
        payload={
            "cmd": "ps aux | awk '$8 ~ /Z/ {print $2}' | xargs -r kill -9 2>/dev/null; echo 'done'"
        },
    )
    sp.add(
        "bg_python_test",
        "Run quick Python smoke test in bg",
        T.PROCESS,
        payload={
            "cmd": "nohup python3 -c 'import sys; print(sys.version)' > /tmp/noosphere_py_test.out 2>&1 &"
        },
    )
    sp.add(
        "reload_systemd",
        "Reload systemd daemon config",
        T.PROCESS,
        payload={
            "cmd": "systemctl daemon-reload 2>/dev/null || echo 'systemd not available'"
        },
    )
    sp.add(
        "restart_service",
        "Restart noosphere service (if exists)",
        T.PROCESS,
        payload={
            "cmd": "systemctl restart noosphere 2>/dev/null || echo 'noosphere service not found'"
        },
    )
    sp.add(
        "docker_restart",
        "Restart stopped containers",
        T.PROCESS,
        payload={
            "cmd": "docker ps -aq --filter status=exited | head -3 | xargs docker start 2>/dev/null || echo 'no stopped containers'"
        },
    )

    # ── Tier 3: NETWORK ───────────────────────────────────────────────────────
    # Network I/O.

    sp.add(
        "ping_gateway",
        "Ping default gateway",
        T.NETWORK,
        payload={
            "cmd": "ping -c 3 -W 2 $(ip route | awk '/default/{print $3; exit}') 2>/dev/null || echo 'ping failed'"
        },
    )
    sp.add(
        "ping_dns",
        "Ping DNS resolver",
        T.NETWORK,
        payload={"cmd": "ping -c 3 -W 2 8.8.8.8 2>/dev/null || echo 'ping failed'"},
    )
    sp.add(
        "curl_health",
        "HTTP health check (localhost)",
        T.NETWORK,
        payload={
            "cmd": "curl -sf http://localhost/health 2>/dev/null || curl -sf http://localhost:8080/health 2>/dev/null || curl -sf http://localhost:3000 2>/dev/null || echo 'no local HTTP service'"
        },
    )
    sp.add(
        "curl_ip",
        "Get public IP",
        T.NETWORK,
        payload={
            "cmd": "curl -sf --max-time 5 https://ifconfig.me 2>/dev/null || echo 'no internet'"
        },
    )
    sp.add(
        "wget_check",
        "Check URL reachability",
        T.NETWORK,
        payload={
            "cmd": "wget -q --spider --timeout=5 https://pypi.org 2>/dev/null && echo 'PyPI reachable' || echo 'PyPI unreachable'"
        },
    )
    sp.add(
        "git_fetch",
        "git fetch (check for updates)",
        T.NETWORK,
        payload={"cmd": "git fetch --dry-run 2>&1 | head -10 || echo 'not a git repo'"},
    )
    sp.add(
        "git_pull",
        "git pull",
        T.NETWORK,
        payload={"cmd": "git pull 2>&1 | head -10 || echo 'not a git repo'"},
    )
    sp.add(
        "pip_outdated",
        "Check for outdated pip packages",
        T.NETWORK,
        payload={
            "cmd": "pip list --outdated 2>/dev/null | head -15 || echo 'pip not available'"
        },
    )
    sp.add(
        "nix_update",
        "nix-channel --update",
        T.NETWORK,
        payload={
            "cmd": "nix-channel --update 2>&1 | head -10 || echo 'nix not available'"
        },
    )

    # ── Tier 4: SYSTEM ────────────────────────────────────────────────────────
    # System-level changes.

    sp.add(
        "nix_install",
        "nix-env -i (requires package name)",
        T.SYSTEM,
        payload={"cmd": "echo 'specify package: nix-env -iA nixpkgs.PACKAGE'"},
    )
    sp.add(
        "pip_upgrade",
        "Upgrade outdated pip packages",
        T.SYSTEM,
        payload={
            "cmd": "pip list --outdated --format=freeze 2>/dev/null | grep -v '^\\-e' | cut -d = -f 1 | head -5 | xargs pip install -U 2>&1 | tail -10"
        },
    )
    sp.add(
        "nixos_rebuild",
        "nixos-rebuild switch (dry-run)",
        T.SYSTEM,
        payload={
            "cmd": "nixos-rebuild dry-build 2>&1 | tail -10 || echo 'not NixOS or no config'"
        },
    )
    sp.add(
        "clear_cache",
        "Clear pip/nix/docker caches",
        T.SYSTEM,
        payload={
            "cmd": "pip cache purge 2>/dev/null; docker system prune -f 2>/dev/null; nix-collect-garbage 2>/dev/null; echo 'cache clear attempted'"
        },
    )
    sp.add(
        "set_timezone",
        "Set timezone to UTC",
        T.SYSTEM,
        payload={
            "cmd": "timedatectl set-timezone UTC 2>/dev/null || echo 'timedatectl not available'"
        },
    )

    # ── Tier 5: DESTRUCTIVE ───────────────────────────────────────────────────
    # Irreversible. Require explicit allow_tiers=[5] or allow_all=True.

    sp.add(
        "rm_build_artifacts",
        "Remove build artifacts",
        T.DESTRUCTIVE,
        payload={
            "cmd": "rm -rf build/ dist/ __pycache__/ .pytest_cache/ *.egg-info/ 2>/dev/null; echo 'artifacts removed'"
        },
    )
    sp.add(
        "git_reset_hard",
        "git reset --hard HEAD",
        T.DESTRUCTIVE,
        payload={"cmd": "git reset --hard HEAD 2>/dev/null || echo 'not a git repo'"},
    )
    sp.add(
        "docker_prune_all",
        "Remove all stopped containers/images",
        T.DESTRUCTIVE,
        payload={
            "cmd": "docker system prune -af 2>/dev/null || echo 'docker not available'"
        },
    )

    return sp


# ── Digital state observation ─────────────────────────────────────────────────


class DigitalStateObserver:
    """
    Captures structured system state at each agent step.
    Passed as obs["structured"] (shape: (N_STATE_DIMS,) float32) so the
    world model learns to predict how actions change the digital environment,
    not just scalar reward.

    Dimensions (64-dim total):
        [0-5]   Memory: total, used, free, cached, swap_total, swap_used (GB)
        [6-10]  CPU: load_1m, load_5m, load_15m, n_cores, cpu_freq_MHz
        [11-15] Disk: total_GB, used_GB, free_GB, read_MB_s, write_MB_s
        [16-20] Processes: n_total, n_running, n_sleeping, n_zombie, n_threads
        [21-25] Network: rx_MB_s, tx_MB_s, n_tcp_conns, n_listening, n_established
        [26-30] GPU: util%, mem_util%, mem_used_MB, temp_C, n_gpus
        [31-35] Git: has_repo, n_modified, n_untracked, n_ahead, n_behind
        [36-40] Docker: n_running, n_stopped, n_images, n_volumes, n_networks
        [41-45] Python: has_venv, n_packages, has_cuda, torch_ok, n_ml_procs
        [46-50] Files: n_py, n_config, n_log, n_binary, n_dirs (in cwd)
        [51-55] Last command: exit_code, stdout_bytes, stderr_bytes, duration_s, n_lines
        [56-60] Time: hour/24, day_of_week/7, is_weekend, uptime_h (norm), epoch_s (norm)
        [61-63] Reserved / future
    """

    N_DIMS = 64

    def observe(
        self, last_result: Optional[Dict] = None, timeout_s: float = 1.0
    ) -> np.ndarray:
        """Collect state vector. Runs fast parallel probes with short timeouts."""
        v = np.zeros(self.N_DIMS, dtype=np.float32)
        t_start = time.time()

        # Memory [0-5]
        try:
            with open("/proc/meminfo") as f:
                mi = {}
                for line in f:
                    k, val = line.split(":")
                    mi[k.strip()] = int(val.strip().split()[0]) / 1024**2  # GB
            v[0] = mi.get("MemTotal", 0)
            v[1] = (
                mi.get("MemTotal", 0)
                - mi.get("MemFree", 0)
                - mi.get("Buffers", 0)
                - mi.get("Cached", 0)
            )
            v[2] = mi.get("MemAvailable", 0)
            v[3] = mi.get("Cached", 0) + mi.get("Buffers", 0)
            v[4] = mi.get("SwapTotal", 0)
            v[5] = mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)
        except Exception:
            pass

        # CPU [6-10]
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
                v[6], v[7], v[8] = float(parts[0]), float(parts[1]), float(parts[2])
            v[9] = float(os.cpu_count() or 1)
            with open("/proc/cpuinfo") as f:
                freqs = [float(l.split(":")[1].strip()) for l in f if "cpu MHz" in l]
                v[10] = float(np.mean(freqs)) if freqs else 0.0
        except Exception:
            pass

        # Disk [11-15]
        try:
            import shutil

            du = shutil.disk_usage(".")
            v[11] = du.total / 1e9
            v[12] = du.used / 1e9
            v[13] = du.free / 1e9
        except Exception:
            pass

        # Processes [16-20]
        try:
            r = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=timeout_s
            )
            lines = r.stdout.strip().splitlines()[1:]
            stats = [l.split()[7] for l in lines if len(l.split()) > 7]
            v[16] = len(stats)
            v[17] = sum(1 for s in stats if s == "R")
            v[18] = sum(1 for s in stats if s == "S")
            v[19] = sum(1 for s in stats if "Z" in s)
        except Exception:
            pass

        # Network [21-25]
        try:
            r = subprocess.run(
                ["ss", "-tnp"], capture_output=True, text=True, timeout=timeout_s
            )
            lines = r.stdout.strip().splitlines()[1:]
            v[23] = len(lines)
            v[25] = sum(1 for l in lines if "ESTAB" in l)
            r2 = subprocess.run(
                ["ss", "-tlnp"], capture_output=True, text=True, timeout=timeout_s
            )
            v[24] = len(r2.stdout.strip().splitlines()) - 1
        except Exception:
            pass

        # GPU [26-30]
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,count",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split(",")
                if len(parts) >= 4:
                    v[26] = float(parts[0].strip()) / 100.0
                    v[27] = float(parts[1].strip()) / 100.0
                    v[28] = float(parts[2].strip()) / 1024.0  # MB → GB
                    v[29] = float(parts[3].strip())
                    v[30] = 1.0
        except Exception:
            pass

        # Git [31-35]
        try:
            r = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=".",
            )
            if r.returncode == 0:
                v[31] = 1.0
                lines = r.stdout.splitlines()
                v[32] = sum(1 for l in lines if l and l[0] in "MARD")
                v[33] = sum(1 for l in lines if l.startswith("??"))
            r2 = subprocess.run(
                ["git", "rev-list", "--count", "--left-right", "@{u}...HEAD"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if r2.returncode == 0:
                parts = r2.stdout.split()
                if len(parts) == 2:
                    v[34] = float(parts[1])
                    v[35] = float(parts[0])
        except Exception:
            pass

        # Docker [36-40]
        try:
            r = subprocess.run(
                ["docker", "ps", "-aq"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            all_c = len(r.stdout.strip().splitlines()) if r.returncode == 0 else 0
            r2 = subprocess.run(
                ["docker", "ps", "-q"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            run_c = len(r2.stdout.strip().splitlines()) if r2.returncode == 0 else 0
            v[36] = run_c
            v[37] = all_c - run_c
        except Exception:
            pass

        # Python env [41-45]
        v[41] = (
            1.0
            if os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_DEFAULT_ENV")
            else 0.0
        )
        try:
            r = subprocess.run(
                ["pip", "list"], capture_output=True, text=True, timeout=timeout_s
            )
            if r.returncode == 0:
                v[42] = float(len(r.stdout.strip().splitlines()) - 2)
        except Exception:
            pass

        # Files in cwd [46-50]
        try:
            entries = os.listdir(".")
            v[46] = sum(1 for e in entries if e.endswith(".py"))
            v[47] = sum(
                1
                for e in entries
                if e.endswith((".toml", ".yaml", ".yml", ".json", ".conf"))
            )
            v[48] = sum(1 for e in entries if e.endswith(".log"))
            v[50] = sum(1 for e in entries if os.path.isdir(e))
        except Exception:
            pass

        # Last command result [51-55]
        if last_result:
            v[51] = float(np.clip(last_result.get("exit_code", 0) / 128.0, -1, 1))
            v[52] = min(float(len(last_result.get("stdout", ""))), 65536.0)
            v[53] = min(float(len(last_result.get("stderr", ""))), 4096.0)
            v[54] = min(float(last_result.get("duration_s", 0.0)), 60.0)
            v[55] = min(float(last_result.get("stdout", "").count("\n")), 1000.0)

        # Time [56-60]
        t = time.localtime()
        v[56] = t.tm_hour / 24.0
        v[57] = t.tm_wday / 7.0
        v[58] = 1.0 if t.tm_wday >= 5 else 0.0
        try:
            with open("/proc/uptime") as f:
                v[59] = min(float(f.read().split()[0]) / 3600.0, 1000.0)
        except Exception:
            pass

        return v


# ── Rich feature encoding for command output ──────────────────────────────────


class ShellOutputEncoder:
    """
    Encodes shell command output as a 32-dim feature vector for the world model.
    Was 10-dim (v1.0). Now 32-dim with structured parsing.

    Dimensions:
        [0]    exit_code normalised (-1 to +1)
        [1]    stdout length (log-normalised)
        [2]    stderr length (log-normalised)
        [3]    n_lines (log-normalised)
        [4]    n_numeric_values found in output
        [5]    n_file_paths found
        [6]    n_ip_addresses found
        [7]    n_processes found (PID-like patterns)
        [8]    is JSON output
        [9]    is table output (header + rows)
        [10]   success flag
        [11]   permission denied flag
        [12]   not found flag
        [13]   timeout flag
        [14]   empty output flag
        [15]   long output flag (>1KB)
        [16-19] numeric stats from output (mean, max, min, std) normalised
        [20]   n_error_keywords (error, fail, exception, traceback)
        [21]   n_warning_keywords (warn, deprecated, skip)
        [22]   n_ok_keywords (ok, success, done, complete, pass)
        [23]   output changed vs previous same-command run
        [24]   n_unique_words / total_words (lexical diversity)
        [25]   fraction of lines with numbers
        [26]   fraction of lines with paths (/)
        [27-31] reserved
    """

    N_DIMS = 32

    _ERR_KW = re.compile(r"error|fail|exception|traceback|critical|fatal", re.I)
    _WARN_KW = re.compile(r"warn|deprecated|skip|ignored", re.I)
    _OK_KW = re.compile(r"\bok\b|success|done|complete|pass|finished", re.I)
    _NUM = re.compile(r"\b\d+\.?\d*\b")
    _PATH = re.compile(r"(/[\w./-]+)")
    _IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    _PID = re.compile(r"\b\d{4,6}\b")

    def __init__(self):
        self._prev_outputs: Dict[str, str] = {}

    def encode(
        self, cmd_name: str, exit_code: int, stdout: str, stderr: str
    ) -> np.ndarray:
        v = np.zeros(self.N_DIMS, dtype=np.float32)
        combined = stdout + stderr
        lines = combined.splitlines()

        v[0] = float(np.clip(exit_code / 128.0, -1.0, 1.0))
        v[1] = math.log1p(len(stdout)) / 12.0
        v[2] = math.log1p(len(stderr)) / 10.0
        v[3] = math.log1p(len(lines)) / 8.0

        nums = [float(m) for m in self._NUM.findall(combined)]
        v[4] = math.log1p(len(nums)) / 6.0
        v[5] = math.log1p(len(self._PATH.findall(combined))) / 5.0
        v[6] = math.log1p(len(self._IP.findall(combined))) / 4.0
        v[7] = math.log1p(len(self._PID.findall(combined))) / 5.0

        try:
            json.loads(stdout.strip())
            v[8] = 1.0
        except Exception:
            pass
        if len(lines) > 2 and all("\t" in l or "  " in l for l in lines[:3]):
            v[9] = 1.0

        v[10] = float(exit_code == 0)
        v[11] = float("permission denied" in combined.lower())
        v[12] = float(
            "not found" in combined.lower() or "no such file" in combined.lower()
        )
        v[13] = float("timeout" in combined.lower())
        v[14] = float(len(stdout.strip()) == 0)
        v[15] = float(len(stdout) > 1024)

        if nums:
            arr = np.array(nums, dtype=np.float32)
            norm = max(abs(arr.max()), 1.0)
            v[16] = float(arr.mean()) / norm
            v[17] = float(arr.max()) / norm
            v[18] = float(arr.min()) / norm
            v[19] = float(arr.std()) / norm

        v[20] = math.log1p(len(self._ERR_KW.findall(combined))) / 4.0
        v[21] = math.log1p(len(self._WARN_KW.findall(combined))) / 4.0
        v[22] = math.log1p(len(self._OK_KW.findall(combined))) / 4.0

        prev = self._prev_outputs.get(cmd_name, "")
        v[23] = 0.0 if stdout == prev else 1.0
        self._prev_outputs[cmd_name] = stdout

        words = re.findall(r"\w+", combined)
        if words:
            v[24] = len(set(words)) / max(len(words), 1)
        if lines:
            v[25] = sum(1 for l in lines if self._NUM.search(l)) / len(lines)
            v[26] = sum(1 for l in lines if "/" in l) / len(lines)

        return v


# ── Shell executor ────────────────────────────────────────────────────────────


class ShellExecutor:
    """
    Executes shell commands with tier enforcement and rich output encoding.

    allow_tiers controls which tiers are permitted. Default: [0] (read-only).
    Expand as the world model demonstrates reliable consequence prediction:
        executor.allow_tiers = {0, 1}           # allow file writes
        executor.allow_tiers = {0, 1, 2, 3}     # allow process + network
        executor.allow_tiers = set(range(6))    # allow all (use with care)
    """

    def __init__(
        self,
        working_dir: str = ".",
        timeout_s: float = 30.0,
        allow_tiers: Optional[Set[int]] = None,
        allow_all: bool = False,
        max_output: int = 8192,
    ):
        self.cwd = os.path.abspath(working_dir)
        self.timeout = timeout_s
        self.allow_tiers = (
            set(allow_tiers) if allow_tiers is not None else {Tier.SAFE_READ}
        )
        self.allow_all = allow_all
        self.max_output = max_output
        self._encoder = ShellOutputEncoder()
        self._state_obs = DigitalStateObserver()

    def can_execute(self, action: Action) -> bool:
        if action.payload is None or action.payload.get("cmd") is None:
            return True  # wait action
        if self.allow_all:
            return True
        return action.tier in self.allow_tiers

    def execute(self, action: Action) -> Dict[str, Any]:
        payload = action.payload or {}
        cmd = payload.get("cmd")
        t_start = time.time()

        if cmd is None:
            return {
                "success": True,
                "outcome": "wait",
                "reward": 0.0,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "structured": self._encoder.encode(action.name, 0, "", ""),
                "digital_state": self._state_obs.observe(),
            }

        if not self.can_execute(action):
            return {
                "success": False,
                "outcome": f"Tier {action.tier} not allowed (allowed: {sorted(self.allow_tiers)})",
                "reward": -0.2,
                "stdout": "",
                "stderr": "tier denied",
                "exit_code": -1,
                "structured": self._encoder.encode(action.name, -1, "", "tier denied"),
            }

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                stdin=subprocess.DEVNULL,
            )
            duration = time.time() - t_start
            stdout = result.stdout[: self.max_output]
            stderr = result.stderr[:1024]
            exit_code = result.returncode
            success = exit_code == 0
            reward = self._compute_reward(
                exit_code, stdout, stderr, duration, action.tier
            )

            feats = self._encoder.encode(action.name, exit_code, stdout, stderr)
            digital_state = self._state_obs.observe(
                {
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration_s": duration,
                }
            )

            return {
                "success": success,
                "outcome": f"[{exit_code}] {stdout[:200]}",
                "reward": reward,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "duration_s": duration,
                "structured": feats,
                "digital_state": digital_state,
            }

        except subprocess.TimeoutExpired:
            feats = self._encoder.encode(action.name, -1, "", "timeout")
            return {
                "success": False,
                "outcome": "timeout",
                "reward": -0.5,
                "stdout": "",
                "stderr": "timeout",
                "exit_code": -1,
                "structured": feats,
                "digital_state": self._state_obs.observe(),
            }
        except Exception as e:
            feats = self._encoder.encode(action.name, -1, "", str(e))
            return {
                "success": False,
                "outcome": str(e),
                "reward": -0.3,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "structured": feats,
                "digital_state": self._state_obs.observe(),
            }

    @staticmethod
    def _compute_reward(
        exit_code: int, stdout: str, stderr: str, duration: float, tier: int
    ) -> float:
        """
        Shaped reward for digital task execution.

        Positive signals:
            Exit code 0 → base reward depends on tier (lower tier = higher reward)
            Output contains useful information → small bonus
            Fast execution → small bonus

        Negative signals:
            Non-zero exit code → penalty
            Permission denied → strong penalty (planning failure)
            Timeout → strong penalty
            High tier action failed → heavier penalty (risky action, no payoff)
        """
        if exit_code == 0:
            base = max(0.3, 1.0 - tier * 0.1)  # tier 0: 1.0, tier 5: 0.5
            info = 0.1 if len(stdout.strip()) > 20 else 0.0
            fast = 0.05 if duration < 1.0 else 0.0
            return float(min(base + info + fast, 1.5))
        if "permission denied" in (stdout + stderr).lower():
            return -0.5
        if "not found" in (stdout + stderr).lower():
            return -0.1
        tier_penalty = 0.1 * tier
        return float(-0.2 - tier_penalty)


# ── Other executors ───────────────────────────────────────────────────────────


class Executor(ABC):
    @abstractmethod
    def execute(self, action: Action) -> Dict[str, Any]: ...
    @abstractmethod
    def can_execute(self, action: Action) -> bool: ...


class NullExecutor(Executor):
    def execute(self, action: Action) -> Dict[str, Any]:
        return {
            "success": True,
            "outcome": f"[NullExecutor] {action.name}",
            "reward": 0.0,
        }

    def can_execute(self, action: Action) -> bool:
        return True


class ApparatusExecutor(Executor):
    def __init__(self, movement_executor=None, hardware=None):
        self._mex = movement_executor
        self._hw = hardware
        self._joints = [0.0] * 6

    def can_execute(self, action: Action) -> bool:
        return action.payload is not None and "joint" in action.payload

    def execute(self, action: Action) -> Dict[str, Any]:
        if not self.can_execute(action):
            return {
                "success": False,
                "outcome": "not an apparatus action",
                "reward": -0.1,
            }
        joint = action.payload["joint"]
        delta = action.payload["delta_deg"]
        self._joints[joint] = max(-90.0, min(90.0, self._joints[joint] + delta))
        if self._hw is not None:
            self._hw.set_all_angles(np.array(self._joints))
        return {
            "success": True,
            "outcome": f"joint_{joint} → {self._joints[joint]:.1f}°",
            "reward": 0.0,
            "joints": list(self._joints),
        }


# ── Act phase bridge ──────────────────────────────────────────────────────────


class ActBridge:
    """
    Translates MCTS integer → real-world command with dual confidence gate.
    s4_confidence and predicted_value both must exceed min_confidence.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        executor,
        min_confidence: float = 0.3,
        dry_run: bool = False,
    ):
        self.space = action_space
        self.executor = executor
        self.min_conf = min_confidence
        self.dry_run = dry_run
        self._history: List[Dict] = []

    def act(
        self,
        action_idx: int,
        predicted_value: float = 1.0,
        s4_confidence: Optional[float] = None,
        info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if s4_confidence is None and info is not None:
            s4_confidence = info.get("s4_confidence")
        effective = (
            min(float(predicted_value), float(s4_confidence))
            if s4_confidence is not None
            else float(predicted_value)
        )

        if action_idx >= len(self.space):
            return {
                "executed": False,
                "reason": "invalid index",
                "reward": -0.1,
                "action": None,
                "result": None,
            }

        action = self.space[action_idx]

        if effective < self.min_conf:
            out = {
                "executed": False,
                "reason": f"conf {effective:.2f} < {self.min_conf}",
                "reward": 0.0,
                "action": action,
                "result": None,
            }
            self._history.append(out)
            return out

        if self.dry_run:
            out = {
                "executed": False,
                "reason": "dry_run",
                "reward": 0.0,
                "action": action,
                "result": None,
            }
            self._history.append(out)
            return out

        exec_result = self.executor.execute(action)
        out = {
            "executed": True,
            "action": action,
            "result": exec_result,
            "reward": exec_result.get("reward", 0.0),
            "outcome": exec_result.get("outcome", ""),
            "confidence": effective,
        }
        if "structured" in exec_result:
            out["structured"] = exec_result["structured"]
        if "digital_state" in exec_result:
            out["digital_state"] = exec_result["digital_state"]
        self._history.append(out)
        return out

    def last_n(self, n: int = 5) -> List[Dict]:
        return self._history[-n:]
