function dump_window_features(expdir, jabfile, fly, outmat)
% Dump the window-feature matrix a trained classifier actually uses, for one fly,
% computed by JAABA's own ComputeWindowFeatures, so the Python port can be compared
% column by column.
%
% This mirrors the fast-prediction path in JLabelData: the classifier's stumps index
% into featureNames, those names are grouped by per-frame feature, and
% WindowFeatureName2Params turns each group back into ComputeWindowFeatures
% arguments. Only the columns the classifier references are kept, in the order of
% the classifier's unique feature list, so column k here is column k there.
%
% Inputs:
%   expdir   JAABA experiment directory; its perframe/ cache supplies the inputs
%   jabfile  trained .jab
%   fly      1-based fly index within the experiment
%   outmat   output path
%
% Output .mat contains:
%   X        single (nframes, nfeatures) window-feature matrix, columns ordered as keys
%   keys     1 x nfeatures cell of char, "pff|stat|trans|radius|offset|extra=val,..."
%   pffs     1 x npff cell of the per-frame features used
%   fly      the fly index dumped
%
% Side effects: writes outmat. Nothing is written inside expdir.
addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe');
SetUpJAABAPath;

Q = loadAnonymous(jabfile);
cs = Q.classifierStuff;
assert(numel(cs) == 1, 'Expected a single (2-class) classifier');
featureNames = cs.featureNames;

% the unique window features the stumps reference, in first-use order
dims = [cs.params.dim];
wfidx = unique(dims, 'stable');
wfs = featureNames(wfidx);                       % 1 x nfeatures cell of name cells
nfeat = numel(wfs);
fprintf('%d stumps referencing %d unique window features\n', numel(dims), nfeat);

% group by per-frame feature, as JLabelData does for fast prediction
wf2pff = cellfun(@(x) x{1}, wfs, 'UniformOutput', false);
[pffs, ~, wf2pffidx] = unique(wf2pff);
fprintf('%d per-frame features\n', numel(pffs));

keys = cellfun(@name2key, wfs, 'UniformOutput', false);

% Per-frame features have different natural lengths (a velocity is one frame shorter
% than a position), so every column is computed over frames 1..nframes of the fly and
% NaN-padded where the per-frame vector runs out, which is what JLabelData does.
T = load(fullfile(expdir, 'registered_trx.mat'));
nframes = T.trx(fly).nframes;
X = nan(nframes, nfeat, 'single');

for pfi = 1:numel(pffs)
  pf = pffs{pfi};
  mine = find(wf2pffidx == pfi);                 % columns of wfs from this pff
  params = WindowFeatureName2Params(wfs(mine));

  d = load(fullfile(expdir, 'perframe', [pf '.mat']));
  x = d.data{fly};                               % 1 x n per-frame vector

  % ComputeWindowFeatures sees the whole trajectory, so the `relative` percentile
  % bins are taken over all of it -- the same scope the Python port uses. t0/t1
  % select which frames come back, exactly as in JLabelData.ComputeWindowDataChunk.
  n = min(nframes, numel(x));
  [y, names] = ComputeWindowFeatures(x, params{:}, 't0', 1, 't1', n);   % y is (ncols, n)
  if n < nframes
    y(:, end+1:nframes) = nan;
  end

  % WindowFeatureName2Params returns the cartesian product of the grouped
  % stats/windows/transforms, so y holds more columns than the classifier uses.
  % Keep only the requested ones, matched by name.
  produced = cellfun(@(s) name2key([{pf}, s]), names, 'UniformOutput', false);
  for k = mine(:)'
    j = find(strcmp(keys{k}, produced), 1);
    assert(~isempty(j), 'ComputeWindowFeatures did not produce %s', keys{k});
    X(:, k) = single(y(j, :))';
  end
  fprintf('  %-44s %4d of %4d columns, %d/%d frames\n', ...
          pf, numel(mine), numel(names), n, nframes);
end

save(outmat, 'X', 'keys', 'pffs', 'fly', '-v7');
fprintf('wrote %s  (%d frames x %d features)\n', outmat, size(X, 1), size(X, 2));
end


function k = name2key(c)
% Canonical string for one window-feature name cell, so MATLAB and Python columns
% can be matched. c is {pff,'stat',stat,'trans',trans,'radius',r,'offset',o,...}.
% Extra key/value pairs beyond offset are appended sorted, as key=value.
pff = c{1}; stat = c{3}; trans = c{5}; radius = c{7}; offset = c{9};
extra = {};
for j = 10:2:numel(c)
  extra{end+1} = sprintf('%s=%g', c{j}, c{j+1}); %#ok<AGROW>
end
extra = sort(extra);
k = sprintf('%s|%s|%s|%g|%g|%s', pff, stat, trans, radius, offset, strjoin(extra, ','));
end
