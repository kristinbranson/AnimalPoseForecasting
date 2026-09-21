#!/bin/bash
# Export JAABA .jab classifiers to the plain .mat the Python port reads.
#
# A .jab holds a MATLAB object scipy cannot read, so each one is exported once to a
# flat .mat via export_classifier.m. This exports every jab whose .mat is missing or
# older than the jab, so it can be re-run after training in the JAABA GUI and will
# only redo what changed.
#
# Usage:
#   export_classifiers.sh [jab ...]        specific jabs
#   export_classifiers.sh                  every .jab in DIR
#
# Environment overrides:
#   DIR    directory to scan when no jabs are given
#   FORCE  set to 1 to re-export even when the .mat is up to date
#
# <name>.jab produces <name>.classifier.mat beside it. Editor backups (*.jab~) are
# skipped.

set -uo pipefail

MATLAB=${MATLAB:-/groups/branson/home/bransonk/software/matlab-2023b/bin/matlab}
JD=${JD:-/groups/branson/home/bransonk/behavioranalysis/code/APF_main/jaaba_detect}
DIR=${DIR:-/groups/branson/home/bransonk/behavioranalysis/code/MABe2022}
FORCE=${FORCE:-0}

# This MATLAB install needs GTK_PATH even headless.
export GTK_PATH=${GTK_PATH:-/usr/lib/x86_64-linux-gnu/gtk-2.0}

if [ "$#" -gt 0 ]; then
  jabs=("$@")
else
  mapfile -t jabs < <(find "$DIR" -maxdepth 1 -name '*.jab' ! -name '*~' | sort)
fi

if [ "${#jabs[@]}" -eq 0 ]; then
  echo "no .jab files found in $DIR"
  exit 0
fi

exported=0 skipped=0 failed=0
for jab in "${jabs[@]}"; do
  if [ ! -f "$jab" ]; then
    echo "  missing   $jab"; failed=$((failed + 1)); continue
  fi
  mat="${jab%.jab}.classifier.mat"
  if [ "$FORCE" != "1" ] && [ -f "$mat" ] && [ "$mat" -nt "$jab" ]; then
    echo "  current   $(basename "$mat")"; skipped=$((skipped + 1)); continue
  fi
  echo "  exporting $(basename "$jab") ..."
  if "$MATLAB" -nodisplay -batch \
       "addpath('$JD'); export_classifier('$jab','$mat')" > /tmp/export_$$.log 2>&1; then
    echo "            -> $(basename "$mat")"
    exported=$((exported + 1))
  else
    echo "            FAILED, last lines:"
    tail -5 /tmp/export_$$.log | sed 's/^/              /'
    failed=$((failed + 1))
  fi
  rm -f /tmp/export_$$.log
done

echo
echo "exported $exported, already current $skipped, failed $failed"
