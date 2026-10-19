#!/usr/bin/env bash
# Supply an independently obtained QAIRT 2.47 SDK; no vendor code is downloaded.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the QAIRT 2.47 SDK}"
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to NDK r30}"
OUT=${1:?usage: android/qnn/build.sh OUTPUT_JNILIBS_DIRECTORY}
INCLUDE="$QNN_SDK_ROOT/include/QNN"
[[ -d "$INCLUDE" ]] || INCLUDE="$QNN_SDK_ROOT/include"
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
FLAGS=(-std=gnu11 -Wall -Wextra -Werror -I "$INCLUDE")
cc "${FLAGS[@]}" -fPIC -shared "$HERE/test_graph_tuning.c" -DFAKE_BACKEND -o "$BUILD/libQnnHtp.so"
cc "${FLAGS[@]}" -fPIC -shared "$HERE/graph_tuning.c" -ldl -pthread -o "$BUILD/libJetlinkQnnHtp.so"
cc "${FLAGS[@]}" "$HERE/test_graph_tuning.c" -ldl -o "$BUILD/test-tuning"
for setting in stock:0 memory:1 accurate:14 memory-accurate:15 no-fusion:2 no-short-hmx:4 high-sigmoid:8 invalid:0; do
  JETLINK_QNN_TUNING="${setting%:*}" EXPECTED_FLAGS="${setting#*:}" "$BUILD/test-tuning" "$BUILD/libJetlinkQnnHtp.so"
done
TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
mkdir -p "$OUT/arm64-v8a"
"$TOOLCHAIN/aarch64-linux-android31-clang" "${FLAGS[@]}" -fPIC -shared "$HERE/graph_tuning.c" \
  -ldl -llog -pthread -Wl,-z,relro,-z,now -o "$OUT/arm64-v8a/libJetlinkQnnHtp.so"
python3 - "$QNN_SDK_ROOT" "$OUT/arm64-v8a" "$HERE/qairt-2.47-sha256.json" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
sdk, out, manifest = map(Path, sys.argv[1:])
for name, expected in json.loads(manifest.read_text()).items():
    matches = [p for p in sdk.rglob(name) if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expected]
    if not matches:
        raise SystemExit(f"No byte-matched QAIRT 2.47 library found: {name}")
    shutil.copyfile(matches[0], out / name)
print("Eight byte-matched vendor libraries and the tested graph adapter are ready.")
PY
