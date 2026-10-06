#!/bin/bash
# Reproduce the full validation of the jaaba_detect port.
# Regenerates MATLAB ground truth (test/*_gt.mat, ~150MB total) then runs the
# Python comparisons. Ground-truth .mat files are large and regeneratable; safe
# to delete after.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(dirname "$HERE")"
REPO="$(dirname "$PKG")"
JAB="$REPO/courtship_v2pt3_apt.jab"
CLF="$PKG/courtship_v2pt3_apt.classifier.mat"
EXP="$REPO/data/CsChr_JRC_SS36564_RigB_20210902T075750"
PY=/home/bransonk@hhmi.org/miniforge3/envs/APT/bin/python
MAT="/misc/local/matlab-2025b/bin/matlab -nodisplay -nosplash -batch"
JPATH="/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe"

echo "### 0. export classifier (one-time MATLAB)"
$MAT "addpath('$PKG'); export_classifier('$JAB','$CLF')"
$PY "$PKG/jab_io.py" "$CLF"

echo "### 1. window features"
$MAT "addpath('$HERE'); gen_window_gt2('$HERE/window_gt2.mat')"
$PY "$HERE/check_window_features2.py" "$HERE/window_gt2.mat"
$MAT "addpath('$HERE'); gen_window_gt3('$HERE/window_gt3.mat')"
$PY "$HERE/check_window_features3.py" "$HERE/window_gt3.mat"
$MAT "addpath('$HERE'); gen_window_gt('$HERE/window_gt.mat')"
$PY "$HERE/check_window_features.py" "$HERE/window_gt.mat"

echo "### 2. APT per-frame features"
$MAT "addpath('$JPATH'); SetUpJAABAPath; addpath('$HERE'); apt_ref('$EXP/registered_trx.mat','$EXP/apttrk.mat','$HERE/apt_gt.mat')"
$PY "$HERE/check_apt.py" "$HERE/apt_gt.mat" "$EXP/registered_trx.mat" "$EXP/apttrk.mat"

echo "### 3. ellipse per-frame (simple)"
$MAT "addpath('$HERE'); ellipse_ref('$EXP/registered_trx.mat','$HERE/ellipse_gt.mat')"
$PY "$HERE/check_ellipse.py" "$HERE/ellipse_gt.mat" "$EXP/registered_trx.mat"

echo "### 4. ellipse per-frame (nose2ell / anglesub)"
$MAT "addpath('$JPATH'); SetUpJAABAPath; addpath('$HERE'); ellipse_ref2('$EXP/registered_trx.mat','$HERE/ellipse2_gt.mat')"
$PY "$HERE/check_ellipse2.py" "$HERE/ellipse2_gt.mat" "$EXP/registered_trx.mat"

echo "### 5. classifier core (myBoostClassify + PostProcess)"
$MAT "addpath('$HERE'); clf_core_ref('$JAB','$HERE/clf_core_gt.mat')"
$PY "$HERE/check_clf_core.py" "$HERE/clf_core_gt.mat" "$CLF"

echo "### 6. full trx/ellipse per-frame lexicon (vs cached MATLAB sublexicon; needs a completed JAABADetect run)"
# $PY "$HERE/compare_ellipse_lexicon.py" <registered_trx.mat> <perframedir_with_full_sublexicon>
echo "### DONE — all component checks above should print PASS"
