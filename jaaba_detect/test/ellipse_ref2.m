function ellipse_ref2(trxfile, outfile)
% Reference for the 7 nose2ell / anglesub ellipse features, calling JAABA's
% on-path geometry (dnose2ell_pair, dnose2ell_bounds, dnose2center_pair,
% dnose2ell_anglerange_pair, anglesubtended) and replicating the closest-fly
% orchestration + simple base features from the compute_*.m sources.
Lt = load(trxfile); trxin = Lt.trx; nflies = numel(trxin);
trx = struct('x_mm',{},'y_mm',{},'theta_mm',{},'a_mm',{},'b_mm',{},...
             'dt',{},'nframes',{},'firstframe',{},'endframe',{},'off',{});
for i = 1:nflies
  trx(i).x_mm=trxin(i).x_mm(:)'; trx(i).y_mm=trxin(i).y_mm(:)';
  trx(i).theta_mm=trxin(i).theta_mm(:)'; trx(i).a_mm=trxin(i).a_mm(:)'; trx(i).b_mm=trxin(i).b_mm(:)';
  trx(i).dt=trxin(i).dt(:)'; trx(i).nframes=double(trxin(i).nframes);
  trx(i).firstframe=double(trxin(i).firstframe); trx(i).endframe=double(trxin(i).endframe);
  trx(i).off=double(trxin(i).off);
end
fov = pi; MAXVALUE = 127; roi = ones(1,nflies); flies = 1:nflies;

% ----- closestfly_nose2ell -----
cf_n2e = cell(1,nflies); angleon = cell(1,nflies);
for i1 = 1:nflies
  flies2 = flies(roi(i1)==roi);
  [mindupper,dlower] = dnose2ell_bounds(trx,i1,flies2);
  mind = inf(1,trx(i1).nframes); closesti = ones(1,trx(i1).nframes); ang = nan(1,trx(i1).nframes);
  for k = 1:numel(flies2)
    fly2 = flies2(k); if i1==fly2, continue; end
    idx1try = find(mindupper >= dlower(k,:));
    [dcurr,anglecurr] = dnose2ell_pair(trx,i1,fly2,idx1try);
    idx = dcurr < mind; mind(idx)=dcurr(idx); closesti(idx)=k; ang(idx)=anglecurr(idx);
  end
  cf = flies2(closesti); cf(isnan(mind)|isinf(mind)) = nan;
  cf_n2e{i1} = cf; angleon{i1} = ang;
end

% ----- closestfly_anglesub -----
cf_as = cell(1,nflies);
for i1 = 1:nflies
  flies2 = flies(roi(i1)==roi);
  asub = nan(numel(flies2),trx(i1).nframes);
  for k = 1:numel(flies2)
    fly2 = flies2(k); if i1==fly2, continue; end
    asub(k,:) = anglesub_pair_local(trx,i1,fly2,fov);
  end
  [maxa,ci] = max(asub,[],1);
  cf = flies2(ci); cf(isnan(maxa)) = nan; cf_as{i1} = cf;
end

% ----- phi -----
phi = cell(1,nflies);
for i = 1:nflies, phi{i} = compute_phi_local(trx(i)); end

% ----- features -----
S = struct();
S.feat_names = {'angleonclosestfly','veltoward_nose2ell','absanglefrom1to2_nose2ell',...
  'absthetadiff_anglesub','absphidiff_anglesub','dnose2ell_angle_min20to20','dnose2ell_angle_min30to30'};
S.f1 = angleon;
S.f2 = veltoward(trx,cf_n2e,flies,roi);
S.f3 = cellfun(@(a) abs(a), anglefrom1to2(trx,cf_n2e,flies,roi), 'uni',0);
S.f4 = absthetadiff(trx,cf_as,flies,roi);
S.f5 = absphidiff(trx,cf_as,phi,flies,roi);
S.f6 = dnose2ell_anglerange(trx,[-20 20]*pi/180,MAXVALUE,roi,flies);
S.f7 = dnose2ell_anglerange(trx,[-30 30]*pi/180,MAXVALUE,roi,flies);
save(outfile,'-struct','S','-v7');
fprintf('wrote %s\n', outfile);
end

% ================= replicated base features =================
function data = veltoward(trx,closest_all,flies,roi)
nflies=numel(trx); data=cell(1,nflies);
for i1=1:nflies
  closestfly=closest_all{i1};
  dx1=diff(trx(i1).x_mm,1,2); dy1=diff(trx(i1).y_mm,1,2);
  x1=trx(i1).x_mm; y1=trx(i1).y_mm; data{i1}=zeros(1,trx(i1).nframes);
  for i2=1:nflies
    if i1==i2, continue; end
    idx=find(closestfly(1:end-1)==i2);
    off=trx(i1).firstframe-trx(i2).firstframe;
    idx(idx+off==trx(i2).nframes)=[];
    if isempty(idx), continue; end
    dx2=trx(i2).x_mm(off+idx)-x1(idx); dy2=trx(i2).y_mm(off+idx)-y1(idx);
    dz=sqrt(dx2.^2+dy2.^2); dx2=dx2./dz; dy2=dy2./dz; dx2(dz==0)=0; dy2(dz==0)=0;
    data{i1}(idx)=dx1(idx).*dx2+dy1(idx).*dy2;
  end
end
end

function data = anglefrom1to2(trx,closest_all,flies,roi)
nflies=numel(trx); data=cell(1,nflies);
for i1=1:nflies
  closestfly=closest_all{i1};
  xnose=trx(i1).x_mm+2*trx(i1).a_mm.*cos(trx(i1).theta_mm);
  ynose=trx(i1).y_mm+2*trx(i1).a_mm.*sin(trx(i1).theta_mm);
  data{i1}=zeros(1,trx(i1).nframes);
  for i2=1:nflies
    if i1==i2, continue; end
    idx=find(closestfly==i2); if isempty(idx), continue; end
    off=trx(i1).firstframe-trx(i2).firstframe;
    dx2=trx(i2).x_mm(off+idx)-xnose(idx); dy2=trx(i2).y_mm(off+idx)-ynose(idx);
    data{i1}(idx)=modrange(atan2(dy2,dx2)-trx(i1).theta_mm(idx),-pi,pi);
  end
end
end

function data = absthetadiff(trx,closest_all,flies,roi)
nflies=numel(trx); data=cell(1,nflies);
for i1=1:nflies
  closestfly=closest_all{i1}; th1=trx(i1).theta_mm; data{i1}=zeros(1,trx(i1).nframes);
  for i2=1:nflies
    if i1==i2, continue; end
    idx=find(closestfly==i2); if isempty(idx), continue; end
    off=trx(i1).firstframe-trx(i2).firstframe;
    data{i1}(idx)=abs(modrange(trx(i2).theta_mm(off+idx)-th1(idx),-pi,pi));
  end
end
end

function data = absphidiff(trx,closest_all,phi,flies,roi)
nflies=numel(trx); data=cell(1,nflies);
for i1=1:nflies
  closestfly=closest_all{i1}; phi1=phi{i1}; data{i1}=zeros(1,trx(i1).nframes);
  for i2=1:nflies
    if i1==i2, continue; end
    idx=find(closestfly(1:end-1)==i2);
    off=trx(i1).firstframe-trx(i2).firstframe;
    idx(idx+off==trx(i2).nframes)=[];
    if isempty(idx), continue; end
    data{i1}(idx)=abs(modrange(phi{i2}(off+idx)-phi1(idx),-pi,pi));
  end
end
end

function data = compute_phi_local(t)
if t.nframes<2, data=t.theta_mm; return; end
dy1=[t.y_mm(2)-t.y_mm(1),(t.y_mm(3:end)-t.y_mm(1:end-2))/2,t.y_mm(end)-t.y_mm(end-1)];
dx1=[t.x_mm(2)-t.x_mm(1),(t.x_mm(3:end)-t.x_mm(1:end-2))/2,t.x_mm(end)-t.x_mm(end-1)];
badidx=dy1==0 & dx1==0; data=atan2(dy1,dx1); data(badidx)=t.theta_mm(badidx);
end

function anglesub = anglesub_pair_local(trx,fly1,fly2,fov)
anglesub = nan(1,trx(fly1).nframes);
t0=max(trx(fly1).firstframe,trx(fly2).firstframe);
t1=min(trx(fly1).endframe,trx(fly2).endframe); if t1<t0, return; end
off1=trx(fly1).off; off2=trx(fly2).off;
for t=t0:t1
  i=t+off1; j=t+off2;
  anglesub(i)=anglesubtended(trx(fly1).x_mm(i),trx(fly1).y_mm(i),2*trx(fly1).a_mm(i),2*trx(fly1).b_mm(i),trx(fly1).theta_mm(i),...
    trx(fly2).x_mm(j),trx(fly2).y_mm(j),2*trx(fly2).a_mm(j),2*trx(fly2).b_mm(j),trx(fly2).theta_mm(j),fov);
end
end

function data = dnose2ell_anglerange(trx,anglerange,MAXVALUE,roi,flies)
logmaxvalue=log(MAXVALUE); issmooth=~isinf(MAXVALUE);
nflies=numel(trx); data=cell(1,nflies);
a1=anglerange(1); a2=modrange(anglerange(2),anglerange(1),anglerange(1)+2*pi);
for i1=1:nflies
  flies2=flies(roi(i1)==roi);
  mindupper=inf(1,trx(i1).nframes); dlower=nan(numel(flies2),trx(i1).nframes); weights=nan(numel(flies2),trx(i1).nframes);
  for k=1:numel(flies2)
    fly2=flies2(k); if i1==fly2, continue; end
    [dnc,off10,off11,off20,off21,~,~,af]=dnose2center_pair(trx,i1,fly2);
    if off10>off11, continue; end
    af=modrange(af,a1,a1+2*pi); idx=af>=a1 & af<=a2;
    idx1in=find(idx)+off10-1; idx2in=find(idx)+off20-1;
    if issmooth
      u=zeros(size(af));
      u(~idx)=min(abs(modrange(af(~idx)-a1,-pi,pi)),abs(modrange(af(~idx)-a2,-pi,pi)))/pi;
      w=exp(logmaxvalue.*u);
      mindupper(off10:off11)=min(mindupper(off10:off11),w.*(dnc(off10:off11)+2*trx(fly2).a_mm(off20:off21)));
      dlower(k,off10:off11)=w.*(dnc(off10:off11)-2*trx(fly2).a_mm(off20:off21));
      weights(k,off10:off11)=w;
    else
      mindupper(idx1in)=min(mindupper(idx1in),dnc(idx1in)+2*trx(fly2).a_mm(idx2in));
      dlower(k,idx1in)=dnc(idx1in)-2*trx(fly2).a_mm(idx2in);
    end
  end
  d=nan(numel(flies2),trx(i1).nframes);
  for k=1:numel(flies2)
    fly2=flies2(k); if i1==fly2, continue; end
    istry=find(dlower(k,:)<=mindupper);
    d(k,:)=weights(k,:).*dnose2ell_anglerange_pair(trx,i1,fly2,anglerange,istry);
  end
  mind=min(d,[],1); mind(isnan(mind))=MAXVALUE; data{i1}=mind;
end
end
