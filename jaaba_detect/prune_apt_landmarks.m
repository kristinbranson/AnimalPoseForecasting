function prune_apt_landmarks(injab, outjab, landmarks)
% PRUNE_APT_LANDMARKS  Copy an APT-based jab without the per-frame features that
% reference some landmarks, ready to be retrained.
%
%   prune_apt_landmarks(injab, outjab, landmarks)
%
% For scoring tracks that lack some landmarks -- e.g. forecasting-model simulations,
% which have no outer wing tips (APT landmarks 19 and 21) -- the classifier must not
% depend on them. This removes every APT per-frame feature whose name lists one of
% `landmarks` (pair, triad, body, global, social and socialpair families) from
% everywhere JAABA reads the feature set, and clears the trained classifier, whose
% stumps index into the old feature list. Labels, experiments and every other setting
% are kept, so the output can be opened and retrained directly.
%
% Features that use the landmarks implicitly are NOT removed: apt_view*_social_*_k
% takes the closest landmark on another fly among all landmarks, so it can still
% depend on the pruned ones.
%
% When JAABA opens an APT jab, the feature lexicon comes from aptInfo.featureLexicon
% (Macguffin.initFromFeatureLexiconName), the top-level featureLexicon is ignored, and
% the selection comes from sublexiconPFNames and windowFeaturesParams. All four are
% pruned. aptInfo.pairs/triads are only read by the APT project dialog when a project
% is created; they are pruned too so the settings match the features.
%
% Inputs:
%   injab      trained APT jab to copy from (not modified)
%   outjab     path for the pruned copy; must not exist yet
%   landmarks  APT landmark indices (1-based) to remove, e.g. [19 21]
%
% Side effects: writes outjab. Per-frame files are untouched: the remaining features'
% cached values do not depend on the removed ones.

addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe');
SetUpJAABAPath;

% never overwrite a jab, including injab itself
assert(~exist(outjab, 'file'), '%s already exists; not overwriting it', outjab);

Q = loadAnonymous(injab);                   % struct(Macguffin), as saveJabFile writes it
assert(isstruct(Q) && isfield(Q, 'fromAPT') && Q.fromAPT, '%s is not an APT jab', injab);

% per-frame features to drop, taken from the lexicon JAABA actually loads
lexnames = fieldnames(Q.aptInfo.featureLexicon.perframe);
drop = lexnames(cellfun(@(n) uses_landmarks(n, landmarks), lexnames));
fprintf('%s: removing %d of %d per-frame features that reference landmark(s) %s\n', ...
        injab, numel(drop), numel(lexnames), mat2str(landmarks));

% feature lexicons: the one JAABA loads, and the top-level copy kept for consistency
Q.aptInfo.featureLexicon.perframe = rmfield(Q.aptInfo.featureLexicon.perframe, drop);
Q.featureLexicon.perframe = rmfield(Q.featureLexicon.perframe, ...
    intersect(drop, fieldnames(Q.featureLexicon.perframe)));

% selection: one entry per classifier when sublexiconPFNames is a cell of cells
if iscell(Q.sublexiconPFNames) && ~isempty(Q.sublexiconPFNames) && iscell(Q.sublexiconPFNames{1})
  for i = 1:numel(Q.sublexiconPFNames)
    Q.sublexiconPFNames{i} = setdiff(Q.sublexiconPFNames{i}, drop, 'stable');
  end
else
  Q.sublexiconPFNames = setdiff(Q.sublexiconPFNames, drop, 'stable');
end
if iscell(Q.windowFeaturesParams)
  for i = 1:numel(Q.windowFeaturesParams)
    Q.windowFeaturesParams{i} = rmfield(Q.windowFeaturesParams{i}, ...
        intersect(drop, fieldnames(Q.windowFeaturesParams{i})));
  end
else
  Q.windowFeaturesParams = rmfield(Q.windowFeaturesParams, ...
      intersect(drop, fieldnames(Q.windowFeaturesParams)));
end

% APT project settings: pairs and triads that include a removed landmark
keep = @(sets) sets(cellfun(@(p) ~any(ismember(p, landmarks)), sets));
npairs = numel(Q.aptInfo.pairs);
ntriads = numel(Q.aptInfo.triads);
Q.aptInfo.pairs = keep(Q.aptInfo.pairs);
Q.aptInfo.triads = keep(Q.aptInfo.triads);
fprintf('  aptInfo.pairs %d -> %d, aptInfo.triads %d -> %d\n', ...
        npairs, numel(Q.aptInfo.pairs), ntriads, numel(Q.aptInfo.triads));

% trained classifier: its stumps and feature encodings refer to the old feature list
for i = 1:numel(Q.classifierStuff)
  Q.classifierStuff(i).params = [];
  Q.classifierStuff(i).featureNames = {};
  Q.classifierStuff(i).scoreNorm = [];
  Q.classifierStuff(i).windowdata = [];
  Q.classifierStuff(i).selFeatures = SelFeatures.createEmpty();
end

saveAnonymous(outjab, Q);
fprintf('  wrote %s (classifier cleared; retrain before use)\n', outjab);
end


function tf = uses_landmarks(name, landmarks)
% True if an APT per-frame feature name lists any of `landmarks`.
% Names are apt_view<v>_<family>_<kind>_<i>[_<j>[_<k>]], so every token after the
% fourth is a landmark index. Non-APT features never match.
tokens = strsplit(name, '_');
if numel(tokens) < 5 || ~strcmp(tokens{1}, 'apt')
  tf = false;
  return;
end
tf = any(ismember(str2double(tokens(5:end)), landmarks));
end
