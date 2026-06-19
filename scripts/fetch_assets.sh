#!/usr/bin/env bash
# Download a small set of texture-free Khronos glTF sample models into assets/.
# Run once from anywhere in the repo.

set -euo pipefail

REPO="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets"

fetch() {
	local src="$1" dst="$DEST/$2"
	mkdir -p "$(dirname "$dst")"
	printf "  %s\n" "$2"
	echo "curl -fsSL \"$src\" -o \"$dst\""
	curl -fsSL "$src" -o "$dst"
}

echo "Fetching glTF sample models -> assets/"

# Box — trivial mesh, good sanity check
fetch "$REPO/Box/glTF/Box.gltf"               "Box/Box.gltf"
fetch "$REPO/Box/glTF/Box0.bin"               "Box/Box0.bin"

# Suzanne — Blender monkey, ~500 tris, no textures
fetch "$REPO/Suzanne/glTF/Suzanne.gltf"      "Suzanne/Suzanne.gltf"
fetch "$REPO/Suzanne/glTF/Suzanne.bin"       "Suzanne/Suzanne.bin"

# Sponza — large scene for stress testing (fetch last)
fetch "$REPO/Sponza/glTF/Sponza.gltf"         "Sponza/Sponza.gltf"
fetch "$REPO/Sponza/glTF/Sponza.bin"         "Sponza/Sponza.bin"

echo "Done."
