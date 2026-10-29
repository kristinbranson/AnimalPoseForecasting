function [params, nchanged] = set_max_window_radius(params, max_radius)
% SET_MAX_WINDOW_RADIUS  Rewrite max_window_radius throughout JAABA window parameters.
%
%   [params, nchanged] = set_max_window_radius(params, max_radius)
%
% A jab's windowFeaturesParams is a cell wrapping a struct with one field per
% per-frame feature (503 of them for these projects). Each of those holds a
% max_window_radius, and so does each of its per-statistic sub-structs (mean, min,
% max, change, std, harmonic, diff_neighbor_*, zscore_neighbors). Every copy has to
% agree or the radii end up differing between statistics, so this walks the whole
% nested structure rather than setting the top-level value alone.
%
% The radii JAABA derives from this are
%   unique(round(logspace(log10(min_window_radius+1), log10(max_radius+1), n))) - 1
% and the offsets are {0, +-radius}, so a classifier reads up to 2*max_radius frames
% either side of the frame it is scoring.
%
% Inputs:
%   params: cell, struct or struct array of window feature parameters, nested to any
%     depth. Anything else is returned untouched.
%   max_radius: scalar value written into every max_window_radius field.
%
% Outputs:
%   params: the input with every max_window_radius replaced.
%   nchanged: number of fields set, for the caller to sanity-check against zero.

nchanged = 0;
if iscell(params)
  for i = 1:numel(params)
    [params{i}, sub] = set_max_window_radius(params{i}, max_radius);
    nchanged = nchanged + sub;
  end
  return;
end
if ~isstruct(params)
  return;
end
for i = 1:numel(params)
  fields = fieldnames(params(i));
  for j = 1:numel(fields)
    name = fields{j};
    if strcmp(name, 'max_window_radius')
      params(i).(name) = max_radius;
      nchanged = nchanged + 1;
    elseif isstruct(params(i).(name)) || iscell(params(i).(name))
      [params(i).(name), sub] = set_max_window_radius(params(i).(name), max_radius);
      nchanged = nchanged + sub;
    end
  end
end
end
