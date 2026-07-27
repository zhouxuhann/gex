#!/bin/bash
# Restart IB Gateway before the US cash open.
#
# The launch agent can run this every few minutes. The script itself only
# restarts inside the New York premarket window and records one restart per
# New York trading date.

set -u

ROOT_DIR="${GEX_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
AUTOMATION_DIR="${IB_RESTART_HOME:-$HOME/.scripts/ib_gateway_restart}"

IB_APP_PATH="${IB_APP_PATH:-$HOME/Applications/IB Gateway 10.46/IB Gateway 10.46.app}"
IB_APP_NAME="${IB_APP_NAME:-IB Gateway 10.46}"
IB_GATEWAY_HOST="${IB_GATEWAY_HOST:-127.0.0.1}"
IB_GATEWAY_PORT="${IB_GATEWAY_PORT:-4002}"
IB_START_METHOD="${IB_START_METHOD:-ibc}"
IBC_START_SCRIPT="${IBC_START_SCRIPT:-$HOME/.scripts/start_ib_gateway_ibc.sh}"
IBC_CONFIG="${IBC_CONFIG:-$HOME/ibc/config.ini}"
IBC_AGENT_LABEL="${IBC_AGENT_LABEL:-com.fanzhouxu.ibgateway.ibc}"
IBC_AGENT_PLIST="${IBC_AGENT_PLIST:-$HOME/Library/LaunchAgents/$IBC_AGENT_LABEL.plist}"
IB_AUTO_LOGIN="${IB_AUTO_LOGIN:-0}"
IB_AUTO_LOGIN_ALLOW_UI_PASTE="${IB_AUTO_LOGIN_ALLOW_UI_PASTE:-0}"
IB_AUTO_LOGIN_DRY_RUN="${IB_AUTO_LOGIN_DRY_RUN:-0}"
IB_AUTO_LOGIN_CLICK="${IB_AUTO_LOGIN_CLICK:-1}"
IB_USERNAME_KEYCHAIN_SERVICE="${IB_USERNAME_KEYCHAIN_SERVICE:-ibkr-username}"
IB_PASSWORD_KEYCHAIN_SERVICE="${IB_PASSWORD_KEYCHAIN_SERVICE:-ibkr-password}"
IB_LOGIN_WAIT_SEC="${IB_LOGIN_WAIT_SEC:-20}"
WINDOW_START_MIN="${WINDOW_START_MIN:-535}"  # 08:55 ET
WINDOW_END_MIN="${WINDOW_END_MIN:-550}"      # 09:10 ET
QUIT_WAIT_SEC="${QUIT_WAIT_SEC:-25}"
PORT_WAIT_SEC="${PORT_WAIT_SEC:-180}"
NOTIFY_EMAIL_ENABLED="${NOTIFY_EMAIL_ENABLED:-0}"
EMAIL_SENDER="${EMAIL_SENDER:-fzhouxu615@gmail.com}"
EMAIL_PASSWORD_ENV="${EMAIL_PASSWORD_ENV:-GMAIL_APP_PASSWORD}"
EMAIL_RECIPIENTS="${EMAIL_RECIPIENTS:-wenyi.hann@gmail.com}"
LOG_DIR="$AUTOMATION_DIR/logs"
STATE_DIR="$AUTOMATION_DIR/state"
STATE_FILE="$STATE_DIR/last_premarket_restart_ny_date"
LOG_FILE="$LOG_DIR/ib_gateway_premarket_restart_$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR" "$STATE_DIR"

if { [ "${LOAD_GEX_ENV:-0}" = "1" ] || [ "${LOAD_GEX_ENV:-0}" = "true" ]; } && [ -r "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$LOG_FILE"
}

send_email() {
  local subject="$1"
  local body="$2"
  if [ "$NOTIFY_EMAIL_ENABLED" != "1" ] && [ "$NOTIFY_EMAIL_ENABLED" != "true" ]; then
    return 0
  fi
  python3 - "$subject" "$body" <<'PY' 2>>"$LOG_FILE" || true
import os
import smtplib
import sys
from email.mime.text import MIMEText
from email.utils import formatdate

subject, body = sys.argv[1], sys.argv[2]
sender = os.environ.get("EMAIL_SENDER", "fzhouxu615@gmail.com")
password_env = os.environ.get("EMAIL_PASSWORD_ENV", "GMAIL_APP_PASSWORD")
password = os.environ.get(password_env, "")
recipients = [x.strip() for x in os.environ.get("EMAIL_RECIPIENTS", "wenyi.hann@gmail.com").split(",") if x.strip()]
if not password or not recipients:
    raise SystemExit(0)
msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = f"[IB Gateway] {subject}"
msg["From"] = sender
msg["To"] = ", ".join(recipients)
msg["Date"] = formatdate(localtime=True)
with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
    s.login(sender, password)
    s.send_message(msg)
PY
}

keychain_password() {
  local service="$1"
  security find-generic-password -s "$service" -w 2>/dev/null || true
}

auto_login_ib() {
  if [ "$IB_AUTO_LOGIN" != "1" ] && [ "$IB_AUTO_LOGIN" != "true" ]; then
    log "auto-login disabled"
    return 0
  fi
  if [ "$IB_AUTO_LOGIN_ALLOW_UI_PASTE" != "1" ] && [ "$IB_AUTO_LOGIN_ALLOW_UI_PASTE" != "true" ]; then
    log "auto-login blocked: IB Gateway password field rejected reliable UI automation; leaving login for manual/IBC handling"
    return 1
  fi

  local username password
  username="$(keychain_password "$IB_USERNAME_KEYCHAIN_SERVICE")"
  password="$(keychain_password "$IB_PASSWORD_KEYCHAIN_SERVICE")"
  if [ -z "$username" ] || [ -z "$password" ]; then
    log "auto-login skipped: missing Keychain entries $IB_USERNAME_KEYCHAIN_SERVICE / $IB_PASSWORD_KEYCHAIN_SERVICE"
    return 1
  fi

  if [ "$IB_AUTO_LOGIN_DRY_RUN" = "1" ] || [ "$IB_AUTO_LOGIN_DRY_RUN" = "true" ]; then
    log "auto-login dry-run: inspecting login fields only; no credentials will be entered"
  else
    log "auto-login: filling credentials from Keychain service=$IB_USERNAME_KEYCHAIN_SERVICE/$IB_PASSWORD_KEYCHAIN_SERVICE"
  fi
  sleep "$IB_LOGIN_WAIT_SEC"

  IB_APP_NAME="$IB_APP_NAME" \
  IBKR_USERNAME="$username" \
  IBKR_PASSWORD="$password" \
  IB_AUTO_LOGIN_DRY_RUN="$IB_AUTO_LOGIN_DRY_RUN" \
  IB_AUTO_LOGIN_CLICK="$IB_AUTO_LOGIN_CLICK" \
  osascript <<'APPLESCRIPT' >>"$LOG_FILE" 2>&1
set appName to system attribute "IB_APP_NAME"
set ibUser to system attribute "IBKR_USERNAME"
set ibPass to system attribute "IBKR_PASSWORD"
set dryRun to system attribute "IB_AUTO_LOGIN_DRY_RUN"
set shouldClick to system attribute "IB_AUTO_LOGIN_CLICK"

tell application appName to activate
delay 1
tell application "System Events"
  if not (exists process appName) then error "process not found: " & appName
  tell process appName
    set frontmost to true
    delay 0.5
    if (count of windows) is 0 then error "no windows for " & appName
    set loginWindow to front window
    set loginFields to text fields of loginWindow
    set userField to missing value
    set passField to missing value
    set fieldDescriptions to {}
    repeat with e in loginFields
      set fieldDescription to ""
      try
        set fieldDescription to description of e as text
      end try
      set end of fieldDescriptions to fieldDescription
      if fieldDescription is "用户名" or fieldDescription is "Username" or fieldDescription is "User name" then
        set userField to e
      else if fieldDescription is "密码" or fieldDescription is "Password" then
        set passField to e
      end if
    end repeat
    if userField is missing value then error "username field not found; field descriptions=" & fieldDescriptions
    if passField is missing value then error "password field not found; field descriptions=" & fieldDescriptions
    log "login field inspection: text fields=" & (count of loginFields as text) & ", descriptions=" & fieldDescriptions
    if dryRun is "1" or dryRun is "true" then return

    set savedClipboard to the clipboard
    try
      click userField
      delay 0.3
      keystroke "a" using command down
      delay 0.1
      set the clipboard to ibUser
      delay 0.1
      keystroke "v" using command down
      delay 0.3
      click passField
      delay 0.3
      keystroke "a" using command down
      delay 0.1
      set the clipboard to ibPass
      delay 0.1
      keystroke "v" using command down
      delay 0.3
      set the clipboard to savedClipboard
    on error errMsg number errNum
      try
        set the clipboard to savedClipboard
      end try
      error errMsg number errNum
    end try

    set loginClicked to false
    set allButtons to buttons of loginWindow
    repeat with b in allButtons
      set buttonName to ""
      set buttonDescription to ""
      try
        set buttonName to name of b as text
      end try
      try
        set buttonDescription to description of b as text
      end try
      if (shouldClick is "1" or shouldClick is "true") and (buttonName is "Log In" or buttonName is "Login" or buttonName is "登录" or buttonDescription is "Log In" or buttonDescription is "Login" or buttonDescription is "登录" or buttonDescription is "模拟登录" or buttonDescription is "纸账户登录") then
        click b
        set loginClicked to true
        exit repeat
      end if
    end repeat
    if loginClicked is false and (shouldClick is "1" or shouldClick is "true") then key code 36
  end tell
end tell
APPLESCRIPT
}

ny_date="$(TZ=America/New_York date '+%Y-%m-%d')"
ny_hour="$(TZ=America/New_York date '+%H')"
ny_minute="$(TZ=America/New_York date '+%M')"
ny_weekday="$(TZ=America/New_York date '+%u')"  # 1=Mon ... 7=Sun
ny_total_min=$((10#$ny_hour * 60 + 10#$ny_minute))

force="${FORCE:-0}"
check_only="${CHECK_ONLY:-0}"

if [ "$force" != "1" ] && [ "$force" != "true" ]; then
  if [ "$ny_weekday" -gt 5 ]; then
    log "skip: NY date=$ny_date weekend weekday=$ny_weekday"
    exit 0
  fi
  if [ "$ny_total_min" -lt "$WINDOW_START_MIN" ] || [ "$ny_total_min" -gt "$WINDOW_END_MIN" ]; then
    log "skip: NY time $(TZ=America/New_York date '+%H:%M') outside restart window"
    exit 0
  fi
  if [ -f "$STATE_FILE" ] && [ "$(cat "$STATE_FILE")" = "$ny_date" ]; then
    log "skip: already restarted for NY date $ny_date"
    exit 0
  fi
fi

if [ "$check_only" = "1" ] || [ "$check_only" = "true" ]; then
  log "check_only: would restart $IB_APP_NAME with method=$IB_START_METHOD at $IB_APP_PATH for NY date $ny_date"
  exit 0
fi

if [ "$IB_START_METHOD" = "ibc" ]; then
  ib_gateway_home="$(dirname "$IB_APP_PATH")"
  if [ ! -d "$ib_gateway_home/jars" ]; then
    message="IB Gateway installation not found: $ib_gateway_home"
    log "error: $message"
    send_email "Premarket restart failed" "$message"
    exit 1
  fi
elif [ ! -d "$IB_APP_PATH" ]; then
  message="IB app path not found: $IB_APP_PATH"
  log "error: $message"
  send_email "Premarket restart failed" "$message"
  exit 1
fi

log "restart begin: app=$IB_APP_PATH host=$IB_GATEWAY_HOST port=$IB_GATEWAY_PORT NY=$ny_date $(TZ=America/New_York date '+%H:%M %Z')"

if [ "$IB_START_METHOD" = "ibc" ]; then
  launchctl kill TERM "gui/$(id -u)/$IBC_AGENT_LABEL" >/dev/null 2>&1 || true
  pkill -TERM -f "ibcalpha.ibc.IbcGateway.*$IBC_CONFIG" >/dev/null 2>&1 || true
  pkill -TERM -f "ibcstart.sh.*--ibc-ini=$IBC_CONFIG" >/dev/null 2>&1 || true
  deadline=$((SECONDS + QUIT_WAIT_SEC))
  while pgrep -f "ibcalpha.ibc.IbcGateway.*$IBC_CONFIG" >/dev/null 2>&1 || pgrep -f "ibcstart.sh.*--ibc-ini=$IBC_CONFIG" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      log "IBC quit timeout; force killing $IBC_AGENT_LABEL"
      pkill -KILL -f "ibcalpha.ibc.IbcGateway.*$IBC_CONFIG" >/dev/null 2>&1 || true
      pkill -KILL -f "ibcstart.sh.*--ibc-ini=$IBC_CONFIG" >/dev/null 2>&1 || true
      break
    fi
    sleep 1
  done
else
  osascript -e "tell application \"$IB_APP_NAME\" to quit" >/dev/null 2>&1 || true
  deadline=$((SECONDS + QUIT_WAIT_SEC))
  while pgrep -f "$IB_APP_PATH/Contents/MacOS/JavaApplicationStub" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      log "quit timeout; force killing $IB_APP_NAME"
      pkill -f "$IB_APP_PATH/Contents/MacOS/JavaApplicationStub" >/dev/null 2>&1 || true
      break
    fi
    sleep 1
  done
fi

sleep 3
if [ "$IB_START_METHOD" = "ibc" ]; then
  if [ ! -x "$IBC_START_SCRIPT" ]; then
    message="IBC start script not executable: $IBC_START_SCRIPT"
    log "error: $message"
    send_email "Premarket restart failed" "$message"
    exit 1
  fi
  log "starting via IBC: $IBC_START_SCRIPT"
  if [ -r "$IBC_AGENT_PLIST" ]; then
    launchctl bootout "gui/$(id -u)" "$IBC_AGENT_PLIST" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$(id -u)" "$IBC_AGENT_PLIST" >/dev/null 2>&1 || true
    launchctl kickstart -k "gui/$(id -u)/$IBC_AGENT_LABEL" >/dev/null 2>&1
    log "IBC launch agent kickstarted: $IBC_AGENT_LABEL"
  else
    nohup "$IBC_START_SCRIPT" >>"$LOG_FILE" 2>&1 &
    log "IBC launched pid=$!"
  fi
else
  open "$IB_APP_PATH"
  log "app opened"
  auto_login_ib || true
fi
log "waiting for $IB_GATEWAY_HOST:$IB_GATEWAY_PORT"

deadline=$((SECONDS + PORT_WAIT_SEC))
while true; do
  if nc -z "$IB_GATEWAY_HOST" "$IB_GATEWAY_PORT" >/dev/null 2>&1; then
    echo "$ny_date" > "$STATE_FILE"
    message="IB Gateway restarted and port $IB_GATEWAY_HOST:$IB_GATEWAY_PORT is reachable for NY date $ny_date."
    log "success: $message"
    send_email "Premarket restart succeeded" "$message"
    exit 0
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    message="IB Gateway start method=$IB_START_METHOD ran, but port $IB_GATEWAY_HOST:$IB_GATEWAY_PORT did not become reachable within ${PORT_WAIT_SEC}s. Check login/session."
    log "error: $message"
    send_email "Premarket restart needs attention" "$message"
    exit 1
  fi
  sleep 5
done
