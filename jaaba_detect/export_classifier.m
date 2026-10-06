function export_classifier(jabfile, outfile)
% EXPORT_CLASSIFIER  One-time export of a trained JAABA classifier to a plain
% (scipy-readable) v7 .mat, so the Python port can run inference without MATLAB.
%
%   export_classifier(jabfile, outfile)
%
% Writes, for the classifier's boosted-stump ensemble, parallel arrays indexed
% by stump (dim/dir/tr/alpha/error), the per-stump window-feature descriptor
% parsed into pff/stat/trans/radius/offset(+extra), plus scoreNorm, the
% post-processing parameters, aptInfo (n_pts/head_tail/pairs/triads), and the
% behavior/score-file names. Nothing here is classifier-specific in the code;
% whatever the ensemble references is what gets exported.
%
% Run headless, e.g.:
%   matlab -nodisplay -batch "addpath('.../MABe2022/jaaba_detect'); \
%       export_classifier('.../courtship_v2pt3_apt.jab', '.../courtship_v2pt3_apt.classifier.mat')"

if nargin < 2
  error('Usage: export_classifier(jabfile, outfile)');
end

%% JAABA path
jaabaperframe = '/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe';
addpath(jaabaperframe);
SetUpJAABAPath;

%% load the jab and get the classifier object
Q = loadAnonymous(jabfile);
if isstruct(Q) && isfield(Q,'classifierStuff')
  x = Q;                                  % plain struct form
elseif isobject(Q) || isfield(Q,'classifierStuff')
  x = Q;
else
  error('Unexpected jab contents: could not find classifierStuff');
end
cs = x.classifierStuff;                    % ClassifierStuff object (scalar)
assert(numel(cs) == 1, 'Expected a single (2-class) classifier');

params = cs.params;                        % 1 x nstumps struct array
fn     = cs.featureNames;                  % 1 x Nwf cell of descriptors
nstumps = numel(params);
fprintf('nstumps = %d, numel(featureNames) = %d\n', nstumps, numel(fn));

%% per-stump arrays
dim   = double([params.dim]);
dir   = double([params.dir]);
tr    = double([params.tr]);
alpha = double([params.alpha]);
if isfield(params,'error'), err = double([params.error]); else, err = nan(1,nstumps); end

%% parse each stump's window-feature descriptor
pff    = cell(1,nstumps);
stat   = cell(1,nstumps);
trans  = cell(1,nstumps);
radius = nan(1,nstumps);
offset = nan(1,nstumps);
extra  = cell(1,nstumps);                  % leftover {key,val,...} (e.g. num_harmonic, hist_edges)
for i = 1:nstumps
  [pff{i},stat{i},trans{i},radius(i),offset(i),extra{i}] = parse_desc(fn{dim(i)});
end

%% scoreNorm and post-processing params
scoreNorm = double(cs.scoreNorm);
pp = cs.postProcessParams;
ppmethod = ''; hyst_hi = 0; hyst_lo = 0; filt_size = 1; blen = 1;
if ~isempty(pp)
  if isfield(pp,'method'), ppmethod = pp.method; end
  if isfield(pp,'hystopts') && ~isempty(pp.hystopts)
    hv = pp.hystopts;
    hyst_hi = getopt(hv,{'hthres','high'}, hv(1).value);
    if numel(hv) >= 2, hyst_lo = getopt(hv,{'lthres','low'}, hv(2).value); else, hyst_lo = hyst_hi; end
  end
  if isfield(pp,'filtopts') && ~isempty(pp.filtopts)
    filt_size = getopt(pp.filtopts,{'size'}, pp.filtopts(1).value);
  end
  if isfield(pp,'blen'), blen = double(pp.blen); end
end

%% aptInfo (skeleton geometry the APT features need)
apt_npts = []; apt_headtail = []; apt_pairs = {}; apt_triads = {}; apt_projname = '';
if isfield(x,'aptInfo') && ~isempty(x.aptInfo)
  ai = x.aptInfo;
  if isfield(ai,'n_pts'),        apt_npts     = double(ai.n_pts); end
  if isfield(ai,'head_tail'),    apt_headtail = double(ai.head_tail); end
  if isfield(ai,'pairs'),        apt_pairs    = tocellofdouble(ai.pairs); end
  if isfield(ai,'triads'),       apt_triads   = tocellofdouble(ai.triads); end
  if isfield(ai,'projname'),     apt_projname = ai.projname; end
end

%% per-frame computation params (fov, angle-range max) from the feature lexicon
pf_fov = pi; pf_maxdnose2ell = 127; pf_nbodylengths_near = 2.5;
if isfield(x,'featureLexicon') && isfield(x.featureLexicon,'perframe_params')
  pp2 = x.featureLexicon.perframe_params;
  if isfield(pp2,'fov'), pf_fov = double(pp2.fov); end
  if isfield(pp2,'max_dnose2ell_anglerange'), pf_maxdnose2ell = double(pp2.max_dnose2ell_anglerange); end
  if isfield(pp2,'nbodylengths_near'), pf_nbodylengths_near = double(pp2.nbodylengths_near); end
end

%% behavior / file names
behavior = ''; scorefilename = ''; trxfilename = ''; perframedir = 'perframe'; trkfilename = '';
if isfield(x,'behaviors') && isfield(x.behaviors,'names'), behavior = x.behaviors.names{1}; end
if isfield(x,'file')
  if isfield(x.file,'scorefilename'), scorefilename = firstchar(x.file.scorefilename); end
  if isfield(x.file,'trxfilename'),   trxfilename   = firstchar(x.file.trxfilename); end
  if isfield(x.file,'perframedir'),   perframedir   = firstchar(x.file.perframedir); end
end
if isfield(x,'aptInfo') && isfield(x.aptInfo,'trkfilename'), trkfilename = firstchar(x.aptInfo.trkfilename); end

%% save (v7 so scipy.io.loadmat can read it)
S = struct();
S.nstumps = nstumps;
S.dim = dim; S.dir = dir; S.tr = tr; S.alpha = alpha; S.error = err;
S.pff = pff; S.stat = stat; S.trans = trans; S.radius = radius; S.offset = offset; S.extra = extra;
S.scoreNorm = scoreNorm;
S.pp_method = ppmethod; S.pp_hyst_hi = hyst_hi; S.pp_hyst_lo = hyst_lo;
S.pp_filt_size = filt_size; S.pp_blen = blen;
S.apt_npts = apt_npts; S.apt_headtail = apt_headtail;
S.apt_pairs = apt_pairs; S.apt_triads = apt_triads; S.apt_projname = apt_projname;
S.pf_fov = pf_fov; S.pf_max_dnose2ell_anglerange = pf_maxdnose2ell;
S.pf_nbodylengths_near = pf_nbodylengths_near;
S.behavior = behavior; S.scorefilename = scorefilename;
S.trxfilename = trxfilename; S.perframedir = perframedir; S.trkfilename = trkfilename;
S.jabfile = jabfile;

save(outfile, '-struct', 'S', '-v7');
fprintf('Wrote %s\n', outfile);

% quick summary
upff = unique(pff);
fprintf('unique per-frame features (%d):\n', numel(upff));
fprintf('  %s\n', strjoin(upff, ', '));
fprintf('stats: %s\n', strjoin(unique(stat), ', '));
fprintf('trans: %s\n', strjoin(unique(trans), ', '));
fprintf('radii: %s\n', mat2str(unique(radius)));
fprintf('offsets: %s\n', mat2str(unique(offset)));
fprintf('scoreNorm=%g  pp_method=%s  blen=%g  filt_size=%g  hyst=[%g %g]\n', ...
  scoreNorm, ppmethod, blen, filt_size, hyst_hi, hyst_lo);

end

% ------------------------------------------------------------------
function [pff,stat,trans,radius,offset,extra] = parse_desc(c)
% c = { pffname, 'stat', statname, 'trans', transname, 'radius', r, 'offset', o, [key,val,...] }
pff = c{1};
stat = ''; trans = ''; radius = nan; offset = nan; extra = {};
k = 2;
while k < numel(c)
  key = c{k}; val = c{k+1};
  switch lower(key)
    case 'stat',   stat  = val;
    case 'trans',  trans = val;
    case 'radius', radius = double(val);
    case 'offset', offset = double(val);
    otherwise,     extra(end+1:end+2) = {key, val}; %#ok<AGROW>
  end
  k = k + 2;
end
end

% ------------------------------------------------------------------
function v = getopt(optstruct, tagnames, defaultval)
% pull .value from a 1xN struct array whose element has .tag in tagnames
v = defaultval;
for j = 1:numel(optstruct)
  if isfield(optstruct(j),'tag') && any(strcmpi(optstruct(j).tag, tagnames))
    v = double(optstruct(j).value); return;
  end
end
end

% ------------------------------------------------------------------
function C = tocellofdouble(p)
% normalize pairs/triads (cell of vectors, or an Nxk matrix) to a cell of double row vectors
if iscell(p)
  C = cellfun(@(z) double(z(:)'), p, 'uni', 0);
elseif isnumeric(p)
  C = cell(1,size(p,1));
  for j = 1:size(p,1), C{j} = double(p(j,:)); end
else
  C = {};
end
end

% ------------------------------------------------------------------
function s = firstchar(v)
if iscell(v), s = v{1}; else, s = v; end
if ~ischar(s), s = ''; end
end
