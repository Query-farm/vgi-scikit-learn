#!/usr/bin/env bash
# Copyright 2026 Query Farm LLC - https://query.farm
#
# Verify the package version matches a GitHub Release tag, so the version the
# worker advertises over VGI (implementation_version), the published PyPI wheel,
# and the published container image all equal the release. Run on the `release`
# event before building/publishing anything.
#
# Usage: ci/check-version.sh <release-tag>      # e.g. v0.1.0 or 0.1.0
set -euo pipefail

TAG="${1:?usage: check-version.sh <release-tag>}"
TAG="${TAG#v}"  # accept an optional leading 'v'

HERE="$(cd "$(dirname "$0")" && pwd)"
INIT="$HERE/../vgi_sklearn/__init__.py"
VERSION="$(sed -nE 's/^__version__ = "([^"]+)".*/\1/p' "$INIT")"

if [ -z "$VERSION" ]; then
  echo "::error::could not read __version__ from $INIT" >&2
  exit 1
fi

if [ "$TAG" != "$VERSION" ]; then
  echo "::error::release tag ($TAG) does not match package __version__ ($VERSION); bump vgi_sklearn/__init__.py before tagging." >&2
  exit 1
fi

echo "version OK: $VERSION matches release tag"
