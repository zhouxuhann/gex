#!/bin/bash
# Start IB Gateway through IBC using credentials stored in macOS Keychain.

set -euo pipefail

ROOT_DIR="${GEX_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
IBC_PATH="${IBC_PATH:-$HOME/.local/share/ibc/current}"
IBC_CONFIG="${IBC_CONFIG:-$HOME/ibc/config.ini}"
IBC_LOG_PATH="${IBC_LOG_PATH:-$HOME/ibc/logs}"
IBC_CONFIG_WRITER="${IBC_CONFIG_WRITER:-$HOME/.scripts/write_ibc_config.py}"
IBC_TWS_MAJOR_VRSN="${IBC_TWS_MAJOR_VRSN:-10.46}"
TWS_PATH="${TWS_PATH:-$HOME/Applications}"
TWS_SETTINGS_PATH="${TWS_SETTINGS_PATH:-$HOME/Jts}"
TRADING_MODE="${TRADING_MODE:-paper}"
IB_GATEWAY_PORT="${IB_GATEWAY_PORT:-4002}"
IB_USERNAME_KEYCHAIN_SERVICE="${IB_USERNAME_KEYCHAIN_SERVICE:-ibkr-username}"
IB_PASSWORD_KEYCHAIN_SERVICE="${IB_PASSWORD_KEYCHAIN_SERVICE:-ibkr-password}"
TWOFA_TIMEOUT_ACTION="${TWOFA_TIMEOUT_ACTION:-restart}"

if [ ! -d "$IBC_PATH" ]; then
  echo "IBC path not found: $IBC_PATH" >&2
  exit 1
fi

mkdir -p "$IBC_LOG_PATH" "$TWS_SETTINGS_PATH"

"$IBC_CONFIG_WRITER" \
  --template "$IBC_PATH/config.ini" \
  --output "$IBC_CONFIG" \
  --username-service "$IB_USERNAME_KEYCHAIN_SERVICE" \
  --password-service "$IB_PASSWORD_KEYCHAIN_SERVICE" \
  --trading-mode "$TRADING_MODE" \
  --api-port "$IB_GATEWAY_PORT"

export IBC_VRSN
IBC_VRSN="$(cat "$IBC_PATH/version" 2>/dev/null || basename "$(readlink "$IBC_PATH" 2>/dev/null || echo "$IBC_PATH")")"

exec "$IBC_PATH/scripts/ibcstart.sh" "$IBC_TWS_MAJOR_VRSN" \
  --gateway \
  --tws-path="$TWS_PATH" \
  --tws-settings-path="$TWS_SETTINGS_PATH" \
  --ibc-path="$IBC_PATH" \
  --ibc-ini="$IBC_CONFIG" \
  --mode="$TRADING_MODE" \
  --on2fatimeout="$TWOFA_TIMEOUT_ACTION"
