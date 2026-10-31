#!/bin/sh
# Native unit tests for header-only DXMT logic. Usage: tests/unit/run.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${TMPDIR:-/tmp}/dxmt-unit
mkdir -p "$out"
for t in "$here"/*_test.cpp; do
  name=$(basename "$t" .cpp)
  xcrun clang++ -std=c++20 -O1 -g -Wall -Wextra -fsanitize=address,undefined \
    -I"$root/src/dxmt" -I"$root/src/util" "$t" -o "$out/$name"
  echo "== $name"
  "$out/$name"
done
