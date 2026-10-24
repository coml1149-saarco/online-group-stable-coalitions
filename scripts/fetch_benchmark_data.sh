#!/bin/sh
# Download the benchmark data used for the games of Appendix L into data/benchmarks/raw/.
# The files are not redistributed with the code: Karate Club and Jazz carry no stated data licence
# (KONECT), and Iris is fetched from the UCI repository (CC BY 4.0).
set -e
D="$(cd "$(dirname "$0")/.." && pwd)/data/benchmarks/raw"
mkdir -p "$D/konect_zachary" "$D/konect_jazz" "$D/networkx" "$D/uci_iris"
curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors -o "$D/konect_zachary/download.tar.bz2" http://konect.cc/files/download.tsv.ucidata-zachary.tar.bz2
curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors -o "$D/konect_jazz/download.tar.bz2" http://konect.cc/files/download.tsv.arenas-jazz.tar.bz2
curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors -o "$D/networkx/social.py" https://raw.githubusercontent.com/networkx/networkx/networkx-3.3/networkx/generators/social.py
curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors -o "$D/uci_iris/iris.data" https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
(cd "$D/konect_zachary" && tar -xjf download.tar.bz2)
(cd "$D/konect_jazz" && tar -xjf download.tar.bz2)
# Check that the files are those used for the paper.
(cd "$D" && shasum -a 256 -c "$D/../SHA256SUMS")
