function gen_window_gt3(outfile)
% Ground truth for prctile and hist window stats.
addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe'); SetUpJAABAPath;
rng(999);
X = {randn(1,600).*3+0.5, cumsum(randn(1,400)), randn(1,220)};
X{3}(50:60)=nan; X{3}(1:3)=nan;
radii=[1 4 6]; offsets=[-6 -4 0 4 6]; W=[]; for r=radii, for o=offsets, W(end+1,:)=[r o]; end, end %#ok<AGROW>
S=struct();
for xi=1:numel(X)
  x=X{xi}; N=numel(x);
  prcBins=0:2:100; nx=x(~isnan(x));
  if numel(nx)>5000, sx=round(linspace(1,numel(nx),5000)); rb=prctile(nx(sx),prcBins);
  elseif isempty(nx), rb=linspace(0,1,numel(prcBins));
  elseif min(nx)==max(nx), rb=linspace(min(nx)-.1,max(nx)+.1,numel(prcBins));
  else, rb=prctile(nx,prcBins); end
  RP.relativeBins=rb; RP.prcBins=prcBins;
  xs=struct('x',x);
  % prctile: trans none/abs/relative(/flip) via uint8(15); percentiles 25/50/75
  [yp,namesp]=ComputePrctileWindowFeatures(x,'windows',W,'trans_types',uint8(15),'prctile',[25 50 75],'sanitycheck',false,'relativeParams',RP);
  xs.prctile=pack(yp,namesp);
  % hist: trans none + relative; 4 bins
  hedges=[-5 -1 0 1 5];
  [yh,namesh]=ComputeHistWindowFeatures(x,'windows',W,'trans_types',uint8(15),'hist_edges',hedges,'sanitycheck',false,'relativeParams',RP);
  xs.hist=pack(yh,namesh); xs.hist_edges_full=hedges;
  S.(sprintf('x%d',xi))=xs;
end
save(outfile,'-struct','S','-v7'); fprintf('wrote %s\n',outfile);
end

function P=pack(y,names)
ntr=numel(names); trans=cell(1,ntr); rad=nan(1,ntr); off=nan(1,ntr); prc=nan(1,ntr); he=nan(2,ntr);
for k=1:ntr
  c=names{k};
  for j=1:2:numel(c)
    switch c{j}
      case 'trans', trans{k}=c{j+1};
      case 'radius', rad(k)=c{j+1};
      case 'offset', off(k)=c{j+1};
      case 'prctile', prc(k)=c{j+1};
      case 'hist_edges', he(:,k)=c{j+1}(:);
    end
  end
end
P=struct('Y',y,'trans',{trans},'radius',rad,'offset',off,'prc',prc,'he',he);
end
