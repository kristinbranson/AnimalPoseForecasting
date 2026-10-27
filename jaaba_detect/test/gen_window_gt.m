function gen_window_gt(outfile)
% Generate ground-truth window features from JAABA for a random per-frame vector,
% to validate the Python window_features.py port.
addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe');
SetUpJAABAPath;

rng(12345);
% two test vectors: a longer clean one and a shorter one with NaNs and negatives
X = {};
X{1} = randn(1,600) .* 3 + 0.5;
x2 = randn(1,220);
x2(50:60) = nan;           % interior NaN block
x2(1:3) = nan;             % leading NaN
X{2} = x2;
X{3} = cumsum(randn(1,400));  % correlated, wider range

radii   = [6 9 50];
offsets = [-50 -9 0 6 9 50];
W = [];
for r = radii, for o = offsets, W(end+1,:) = [r o]; end, end %#ok<AGROW>

stats = {'mean','min','max','std'};
S = struct();
for xi = 1:numel(X)
  x = X{xi};
  N = numel(x);
  % relative bins exactly as ComputeWindowFeatures computes them
  resolution = 2; prcBins = 0:resolution:100; nsamples = 5000;
  nx = x(~isnan(x));
  if numel(nx)>nsamples
    sx = round(linspace(1,numel(nx),nsamples)); relativeBins = prctile(nx(sx),prcBins);
  elseif isempty(nx)
    relativeBins = linspace(0,1,numel(prcBins));
  elseif min(nx)==max(nx)
    relativeBins = linspace(min(nx)-.1,max(nx)+.1,numel(prcBins));
  else
    relativeBins = prctile(nx,prcBins);
  end
  RP.relativeBins = relativeBins; RP.prcBins = prcBins;

  xs = struct('x',x,'relativeBins',relativeBins);
  for si = 1:numel(stats)
    st = stats{si};
    fn = str2func(sprintf('Compute%sWindowFeatures', [upper(st(1)) st(2:end)]));
    [y,names] = fn(x,'windows',W,'trans_types',uint8(15),...
                   'sanitycheck',false,'docache',true,'relativeParams',RP);
    % parse names -> parallel arrays
    ntr = numel(names);
    trans = cell(1,ntr); rad = nan(1,ntr); off = nan(1,ntr);
    for k = 1:ntr
      c = names{k};
      for j = 1:2:numel(c)
        switch c{j}
          case 'trans',  trans{k} = c{j+1};
          case 'radius', rad(k) = c{j+1};
          case 'offset', off(k) = c{j+1};
        end
      end
    end
    xs.(st) = struct('Y',y,'trans',{trans},'radius',rad,'offset',off);
  end
  S.(sprintf('x%d',xi)) = xs;
end
save(outfile,'-struct','S','-v7');
fprintf('wrote %s\n', outfile);
end
