#!/usr/bin/env bash
# Builds the Avalonia demo with its native library and installs it for this user, with a launcher entry and icon.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install.sh [--cuda | --cpu] [--profile release|debug] [--prefix DIR] [--uninstall]

  --cuda          build the engine with CUDA (the default when nvidia-smi finds a GPU)
  --cpu           build the engine without CUDA
  --profile P     the cargo profile to build and install (default: release)
  --prefix DIR    where to install (default: ~/.local): DIR/lib/inference-rs-demo, DIR/bin/inference-rs-demo,
                  and the launcher entry and icon under DIR/share
  --uninstall     remove an installed demo (the settings in ~/.config/inference-rs-demo stay)
EOF
}

APP_ID="inference-rs-demo"
APP_NAME="inference.rs demo"
# the class Avalonia gives the window (the assembly name), which docks match to the launcher entry
WM_CLASS="InferenceRs.AvaloniaDemo"
EXECUTABLE="InferenceRs.AvaloniaDemo"
NATIVE_LIB="libinference_ffi.so"

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../../../.." && pwd)"

prefix="$HOME/.local"
profile="release"
accelerator=""
uninstall=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) accelerator="cuda" ;;
        --cpu) accelerator="cpu" ;;
        --profile) profile="${2:?--profile needs a value}"; shift ;;
        --prefix) prefix="${2:?--prefix needs a value}"; shift ;;
        --uninstall) uninstall=1 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

prefix="$(realpath -m "$prefix")"
app_dir="$prefix/lib/$APP_ID"
bin_link="$prefix/bin/$APP_ID"
icon_file="$prefix/share/icons/hicolor/256x256/apps/$APP_ID.png"
desktop_file="$prefix/share/applications/$APP_ID.desktop"

refresh_caches() {
    if command -v update-desktop-database >/dev/null; then
        update-desktop-database -q "$prefix/share/applications" || true
    fi
    if command -v gtk-update-icon-cache >/dev/null; then
        gtk-update-icon-cache -q -t "$prefix/share/icons/hicolor" || true
    fi
}

remove_app_dir() {
    # only ever a directory this script made, never whatever a mistyped prefix points at
    if [[ -d "$app_dir" ]]; then
        if [[ ! -f "$app_dir/$EXECUTABLE" ]]; then
            echo "$app_dir does not hold the demo; leaving it alone" >&2
            exit 1
        fi
        rm -rf -- "$app_dir"
    fi
}

if [[ $uninstall -eq 1 ]]; then
    remove_app_dir
    rm -f -- "$bin_link" "$icon_file" "$desktop_file"
    refresh_caches
    echo "Removed the demo from $prefix"
    exit 0
fi

for tool in cargo dotnet python3; do
    command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 1; }
done

if [[ -z "$accelerator" ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        accelerator="cuda"
    else
        accelerator="cpu"
    fi
fi

features=()
[[ "$accelerator" == "cuda" ]] && features=(--features cuda)
# cargo builds the dev profile into target/debug
cargo_profile="$profile"
[[ "$profile" == "debug" ]] && cargo_profile="dev"
[[ "$profile" == "dev" ]] && profile="debug"
target_dir="$REPO_ROOT/target/$profile"

echo "Building $NATIVE_LIB ($profile, $accelerator)"
(cd "$REPO_ROOT" && cargo build --profile "$cargo_profile" -p inference-ffi "${features[@]}")

staging="$(mktemp -d)"
trap 'rm -rf -- "$staging"' EXIT

echo "Publishing the demo"
dotnet publish "$DEMO_DIR/AvaloniaDemo.csproj" -c Release -o "$staging/app" --use-current-runtime --self-contained false \
    --nologo -v quiet

cp "$target_dir/$NATIVE_LIB" "$staging/app/"
# the CUDA kernel libraries this build loads, in kernels/sm<arch>/ beside the library, where it looks first
python3 - "$target_dir/$NATIVE_LIB" "$staging/app" "$REPO_ROOT/scripts/release" <<'EOF'
import shutil, sys
from pathlib import Path

library, app, helpers = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, helpers)
from kernel_libraries import KERNEL_SUBDIR, built_kernels

for kernel in built_kernels(library):
    target = app / KERNEL_SUBDIR / kernel.relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(kernel.path, target)
EOF

echo "Installing into $prefix"
remove_app_dir
mkdir -p "$(dirname "$app_dir")" "$(dirname "$bin_link")" "$(dirname "$icon_file")" "$(dirname "$desktop_file")"
mv "$staging/app" "$app_dir"
ln -sfn "$app_dir/$EXECUTABLE" "$bin_link"
cp "$DEMO_DIR/Assets/icon.png" "$icon_file"

cat >"$desktop_file" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Chat, speech, images and documents on the inference.rs engine
Exec="$app_dir/$EXECUTABLE"
Path=$app_dir
Icon=$APP_ID
Terminal=false
Categories=Development;
StartupWMClass=$WM_CLASS
EOF

refresh_caches
echo "Installed. Run $APP_ID, or open \"$APP_NAME\" from your application launcher."
case ":$PATH:" in
    *":$prefix/bin:"*) ;;
    *) echo "Note: $prefix/bin is not on PATH" ;;
esac
