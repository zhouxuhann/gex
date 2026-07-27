#!/usr/bin/env python3
"""Generate a private IBC config.ini from the bundled template and Keychain."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def read_keychain(service: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    value = result.stdout.rstrip("\n")
    if not value:
        raise SystemExit(f"missing Keychain value for service: {service}")
    return value


def rewrite_ini(template: str, replacements: dict[str, str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for line in template.splitlines():
        stripped = line.strip()
        key = stripped.split("=", 1)[0] if "=" in stripped and not stripped.startswith("#") else ""
        if key in replacements:
            out.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            out.append(line.rstrip("\r"))

    missing = [key for key in replacements if key not in seen]
    if missing:
        out.append("")
        out.append("# Added by gex IBC setup")
        for key in missing:
            out.append(f"{key}={replacements[key]}")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--username-service", default="ibkr-username")
    parser.add_argument("--password-service", default="ibkr-password")
    parser.add_argument("--trading-mode", default="paper", choices=("live", "paper"))
    parser.add_argument("--api-port", default="4002")
    args = parser.parse_args()

    username = read_keychain(args.username_service)
    password = read_keychain(args.password_service)
    template = Path(args.template).read_text(encoding="utf-8")

    replacements = {
        "IbLoginId": username,
        "IbPassword": password,
        "TradingMode": args.trading_mode,
        "AcceptNonBrokerageAccountWarning": "yes",
        "ReloginAfterSecondFactorAuthenticationTimeout": "yes",
        "SecondFactorAuthenticationExitInterval": "60",
        "ExistingSessionDetectedAction": "primary",
        "OverrideTwsApiPort": args.api_port,
        "ReadOnlyLogin": "no",
        "ReadOnlyApi": "no",
        "AcceptIncomingConnectionAction": "reject",
        "LogStructureWhen": "never",
        "CommandServerPort": "0",
        "FIX": "no",
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700)

    content = rewrite_ini(template, replacements)
    fd, tmp_name = tempfile.mkstemp(prefix=".config.", dir=str(output.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, output)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    print(
        "wrote IBC config: "
        f"{output} mode={args.trading_mode} api_port={args.api_port} "
        f"username_len={len(username)} password_len={len(password)}"
    )


if __name__ == "__main__":
    main()
