#!/usr/bin/env bash
# Downloads the third-party JS the editor depends on into editor/vendor/,
# instead of committing it to the repo. Pinned to the exact versions that
# were vendored before (three.js r160, cannon-es 0.20.0) -- bump the
# version numbers below if you want to intentionally upgrade.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/vendor"

THREE_VERSION="0.160.0"
CANNON_ES_VERSION="0.20.0"

mkdir -p "$VENDOR/three/addons/controls" "$VENDOR/cannon-es"

fetch() {
  echo "fetching $1"
  curl -fsSL "$1" -o "$2"
}

fetch "https://unpkg.com/three@${THREE_VERSION}/build/three.module.js" \
  "$VENDOR/three/three.module.js"
fetch "https://unpkg.com/three@${THREE_VERSION}/examples/jsm/controls/OrbitControls.js" \
  "$VENDOR/three/addons/controls/OrbitControls.js"
fetch "https://unpkg.com/three@${THREE_VERSION}/examples/jsm/controls/PointerLockControls.js" \
  "$VENDOR/three/addons/controls/PointerLockControls.js"
fetch "https://unpkg.com/three@${THREE_VERSION}/examples/jsm/controls/TransformControls.js" \
  "$VENDOR/three/addons/controls/TransformControls.js"

fetch "https://unpkg.com/cannon-es@${CANNON_ES_VERSION}/dist/cannon-es.js" \
  "$VENDOR/cannon-es/cannon-es.js"
fetch "https://unpkg.com/cannon-es@${CANNON_ES_VERSION}/LICENSE" \
  "$VENDOR/cannon-es/LICENSE"

echo "done -- editor/vendor/ populated"
