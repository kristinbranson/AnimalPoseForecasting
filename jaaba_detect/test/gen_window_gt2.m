function gen_window_gt2(outfile)
% Ground truth for the ADDED window stats (change, harmonic, diff_neighbor_*,
% zscore_neighbors), to validate the wingextension-driven additions.
addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe'); SetUpJAABAPath;
rng(12345);
X = {randn(1,600).*3+0.5, cumsum(randn(1,400)), randn(1,220)};
X{3}(50:60)=nan; X{3}(1:3)=nan;
radii=[1 4 6]; offsets=[-6 -4 0 4 6]; W=[]; for r=radii, for o=offsets, W(end+1,:)=[r o]; end, end %#ok<AGROW>
S=struct();
for xi=1:numel(X)
  x=X{xi}; N=numel(x);
  resolution=2; prcBins=0:resolution:100; nx=x(~isnan(x));
  if numel(nx)>5000, sx=round(linspace(1,numel(nx),5000)); rb=prctile(nx(sx),prcBins);
  elseif isempty(nx), rb=linspace(0,1,numel(prcBins));
  elseif min(nx)==max(nx), rb=linspace(min(nx)-.1,max(nx)+.1,numel(prcBins));
  else, rb=prctile(nx,prcBins); end
  RP.relativeBins=rb; RP.prcBins=prcBins;
  xs=struct('x',x);
  calls={ ...
    'change',        {'change_window_radii',[0 1]}; ...
    'harmonic',      {'num_harmonic',3}; ...
    'diff_neighbor_mean', {}; ...
    'diff_neighbor_min',  {}; ...
    'diff_neighbor_max',  {}; ...
    'zscore_neighbors',   {} };
  for ci=1:size(calls,1)
    st=calls{ci,1}; ex=calls{ci,2};
    fn=str2func(sprintf('Compute%sWindowFeatures',regexprep(st,'(^|_)(\w)','${upper($2)}')));
    [y,names]=fn(x,'windows',W,'trans_types',uint8(15),'sanitycheck',false,'docache',true,'relativeParams',RP,ex{:});
    ntr=numel(names); trans=cell(1,ntr); rad=nan(1,ntr); off=nan(1,ntr); cwr=nan(1,ntr); nh=nan(1,ntr);
    for k=1:ntr
      c=names{k};
      for j=1:2:numel(c)
        switch c{j}
          case 'trans', trans{k}=c{j+1};
          case 'radius', rad(k)=c{j+1};
          case 'offset', off(k)=c{j+1};
          case 'change_window_radius', cwr(k)=c{j+1};
          case 'num_harmonic', nh(k)=c{j+1};
        end
      end
    end
    xs.(matlab.lang.makeValidName(st))=struct('Y',y,'trans',{trans},'radius',rad,'offset',off,'cwr',cwr,'nh',nh);
  end
  S.(sprintf('x%d',xi))=xs;
end
save(outfile,'-struct','S','-v7'); fprintf('wrote %s\n',outfile);
end
