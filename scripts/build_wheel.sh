#!/usr/bin/env bash
# Builds the Metrion wheel in a manylinux_2_28 container, so it installs on any x86-64 Linux
# with glibc 2.28 or newer (Ubuntu 20.04+, Debian 11+, RHEL 8+). The wheel lands in dist/.
#
#   scripts/build_wheel.sh          # with docker access, else: sudo scripts/build_wheel.sh
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
image="quay.io/pypa/manylinux_2_28_x86_64"
# Under sudo the wheel still belongs to whoever called it.
owner="${SUDO_UID:-$(id -u)}:${SUDO_GID:-$(id -g)}"

docker run --rm -v "$repo:/io" -w /io -e OWNER="$owner" "$image" bash -euo pipefail -c '
    # clang compiles the BPF programs; libbpf links against libelf and zlib.
    dnf install -y -q clang elfutils-libelf-devel zlib-devel
    curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
    source "$HOME/.cargo/env"
    /opt/python/cp312-cp312/bin/pip install -q "maturin>=1,<2"
    # Outside /io, so the container leaves no root-owned build files in the repository.
    export CARGO_TARGET_DIR=/tmp/target
    # maturin copies libelf and zlib into the wheel, as they are not part of manylinux.
    /opt/python/cp312-cp312/bin/maturin build --release --compatibility manylinux_2_28 \
        --interpreter /opt/python/cp312-cp312/bin/python --out dist
    chown -R "$OWNER" dist
'
ls -l "$repo"/dist/*.whl
