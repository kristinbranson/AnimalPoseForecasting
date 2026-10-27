function retrain_window_radius(jabfile, max_radius, outjabfile, varargin)
% RETRAIN_WINDOW_RADIUS  Retrain a JAABA classifier with a smaller window radius.
%
%   retrain_window_radius(jabfile, max_radius, outjabfile)
%   retrain_window_radius(..., 'exportmat', '.../out.classifier.mat', 'nworkers', 16)
%
% A window feature at frame t spans [t-radius+offset, t+radius+offset], and JAABA sets
% the offsets to {0, +-radius}, so a classifier reads up to 2*max_radius frames either
% side of the frame it scores (plus change_window_radii for the 'change' statistic).
% That reach is what limits scoring of simulated tracks: any frame within it of the
% start of a simulation is classified partly from the real frames that preceded it.
% Shrinking max_window_radius shrinks that contaminated region.
%
% Inputs:
%   jabfile: path to a trained .jab. Its labels and experiment list are reused; only
%     the window parameters change. Per-frame features are unaffected by window
%     parameters, so nothing is recomputed in the experiment directories.
%   max_radius: new max_window_radius, applied to every statistic. JAABA derives the
%     radii as unique(round(logspace(log10(min+1), log10(max_radius+1), n)))-1, and the
%     'change' statistic widens the span by a further change_window_radius (up to 3).
%     Measured on the exported classifier, max_radius 25 gives reach 53 and 10 gives
%     reach 23, so the reach is roughly 2*max_radius + 3.
%   outjabfile: where to write the retrained .jab.
%
% Optional name/value:
%   'exportmat': also export the retrained classifier to this plain .mat, for the
%     Python port. Default '' (skip).
%   'nworkers': size of the parallel pool to use. Default [] (MATLAB's default, which
%     takes the whole machine). Set it when running several retrains at once.
%
% Accuracy is not measured here. JAABA's GetGTPerformance drives its prediction path
% in a way that assumes GUI state and fails headless, and the held-out experiments are
% exactly the MABe test1/test2 splits, so accuracy is measured instead by scoring
% those splits with the exported classifier through the Python port -- the same code
% that scores the simulations.
%
% Side effects:
%   Writes outjabfile, and exportmat when given. Reads the experiment directories
%   named in the jab, which must exist with their per-frame features present.
%
% Example:
%   matlab -nodisplay -batch "addpath('.../jaaba_detect'); \
%     retrain_window_radius('.../chase_apt_split.jab', 5, \
%                           '.../chase_apt_split_r5.jab', ...
%                           'exportmat', '.../chase_apt_r5.classifier.mat')"

parser = inputParser;
parser.addParameter('exportmat', '', @ischar);
parser.addParameter('nworkers', [], @(x) isempty(x) || isscalar(x));
parser.parse(varargin{:});
exportmat = parser.Results.exportmat;
nworkers = parser.Results.nworkers;

%% JAABA path
jaabaperframe = '/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe';
addpath(jaabaperframe);
SetUpJAABAPath;

%% bound the parallel pool so several retrains can share the machine
% Each process also needs its own JobStorageLocation. The default is shared per user
% and per MATLAB release, so concurrent retrains delete each other's job files and the
% pool fails to start.
if ~isempty(nworkers)
  pool = gcp('nocreate');
  if ~isempty(pool) && pool.NumWorkers ~= nworkers
    delete(pool);
    pool = [];
  end
  if isempty(pool)
    cluster = parcluster('Processes');
    jobdir = tempname;
    mkdir(jobdir);
    cluster.JobStorageLocation = jobdir;
    parpool(cluster, nworkers);
    fprintf('Parallel pool: %d workers, job storage %s\n', nworkers, jobdir);
  end
end

%% load the project and rewrite its window parameters
fprintf('Loading %s\n', jabfile);
loaded = loadAnonymous(jabfile);

% These jabs store a plain struct, but openJabFile only accepts a Macguffin when the
% project is handed to it directly. Convert exactly as openJabFile would when it loads
% a jab itself, so nothing else about the project changes.
if isstruct(loaded)
  if isfield(loaded, 'fromAPT') && loaded.fromAPT
    macguffin = Macguffin(loaded, loaded.aptInfo, true);
  else
    macguffin = Macguffin(loaded);
  end
else
  macguffin = loaded;
end

% Discard the stored classifier before changing the radii. Its stumps address window
% features by position in the feature list, and changing the radii rebuilds that list:
% the old indices no longer name the same features, and once the new list is shorter
% than the largest stored index they overflow it outright, which is how radius 5 fails
% while radius 10 does not. Training rebuilds the ensemble from the labels regardless,
% so clearing it loses nothing.
for i = 1:numel(macguffin.classifierStuff)
  macguffin.classifierStuff(i).params = [];
  macguffin.classifierStuff(i).featureNames = {};
  macguffin.classifierStuff(i).scoreNorm = [];
end

[macguffin.windowFeaturesParams, nchanged] = ...
    set_max_window_radius(macguffin.windowFeaturesParams, max_radius);
fprintf('Set max_window_radius = %d in %d places\n', max_radius, nchanged);
if nchanged == 0
  error('No max_window_radius fields found; the jab layout is not what was expected.');
end

%% train on the project's own labels
% isInteractive must be false: changing the radii invalidates the feature names the
% stored classifier was trained against, and JLabelData reports that through warndlg
% when interactive, which cannot open under -nodisplay. Non-interactively it warns and
% drops the stale classifier, which is what retraining wants anyway.
data = JLabelData('isInteractive', false, ...
                  'setstatusfn', @(s) fprintf('  [%s]\n', s), ...
                  'clearstatusfn', @() []);
fprintf('Opening project for training\n');
data.openJabFile(jabfile, false, 'macguffin', macguffin);
fprintf('%d experiments\n', data.nexps);

trainstart = tic;
data.Train();
fprintf('Trained in %.1f s\n', toc(trainstart));

data.saveJabFile(outjabfile);
fprintf('Wrote %s\n', outjabfile);
clear data;

%% export for the Python port
if ~isempty(exportmat)
  export_classifier(outjabfile, exportmat);
  fprintf('Exported %s\n', exportmat);
end

end
