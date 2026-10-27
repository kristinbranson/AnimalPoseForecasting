function apt_ref(trxfile, trkfile, outfile)
% Reference APT per-frame features, using compute_apt.m's exact arithmetic
% (subfunctions copied verbatim from
%  JAABA/perframe/compute_perframe_features/compute_apt.m and compute_apt_social.m)
% driven directly from real registered_trx.mat + apttrk.mat. Validates the Python
% port perframe_apt.py on real data using keypoint indices <= n_pts.

Lt = load(trxfile);                 % registered_trx.mat (v7.3, loads in MATLAB)
trxin = Lt.trx;
nflies = numel(trxin);
Lk = load(trkfile);                 % apttrk.mat
pTrk = Lk.pTrk;
npts = size(pTrk{1},1);

% build per-fly struct array with the fields the copied code needs
trx = struct('x',{},'y',{},'theta',{},'dt',{},'nframes',{},...
             'firstframe',{},'endframe',{},'off',{},'kpts',{});
for i = 1:nflies
  trx(i).x = trxin(i).x(:)';
  trx(i).y = trxin(i).y(:)';
  trx(i).theta = trxin(i).theta(:)';
  trx(i).dt = trxin(i).dt(:)';
  trx(i).nframes = double(trxin(i).nframes);
  trx(i).firstframe = double(trxin(i).firstframe);
  trx(i).endframe = double(trxin(i).endframe);
  trx(i).off = double(trxin(i).off);
  trx(i).kpts = reshape(pTrk{i}, npts*2, trx(i).nframes);   % JAABA storage layout
end
pxpermm = double(trxin(1).pxpermm);
roi = ones(1,nflies);               % single ROI (all flies), matches Python default

feats = {
  'apt_view1_body_x_1','apt_view1_body_y_3','apt_view1_body_cos_9',...
  'apt_view1_body_sin_2','apt_view1_body_dtheta_1','apt_view1_body_dx_15',...
  'apt_view1_body_dy_7','apt_view1_body_distcenter_12','apt_view1_body_ddistcenter_11',...
  'apt_view1_global_velmag_13',...
  'apt_view1_pair_dist_6_10','apt_view1_pair_x_15_16','apt_view1_pair_sin_6_7',...
  'apt_view1_pair_areaswept_10_11',...
  'apt_view1_triad_area_6_5_12','apt_view1_triad_cos_6_4_16','apt_view1_triad_sin_6_4_16',...
  'apt_view1_triad_dangle_6_4_17','apt_view1_triad_dlen_6_4_16',...
  'apt_view1_social_dist_5','apt_view1_social_cos_10','apt_view1_social_sin_15',...
  'apt_view1_social_ddist_9',...
  'apt_view1_socialpair_cos_1_7','apt_view1_socialpair_dist_12_7'};

S = struct(); S.feat_names = feats;
for fi = 1:numel(feats)
  fn = feats{fi};
  data = compute_one(trx, fn, pxpermm, npts, roi);
  S.(sprintf('f%d',fi)) = data;   % cell {1 x nflies}
end
save(outfile,'-struct','S','-v7');
fprintf('wrote %s (%d features, %d flies)\n', outfile, numel(feats), nflies);
end

% ================= driver mirroring compute_apt.m dispatch =================
function all_data = compute_one(trx, fn, pxpermm, n_parts, roi)
fn = regexprep(fn,'^apt_','');       % compute_apt.m receives the name without 'apt_' (keeps view1)
parts = strsplit(fn,'_');
fn_type = parts{2};
comp_type = parts{3};
nflies = numel(trx);
all_data = cell(1,nflies);

if startsWith(fn_type,'social')
  all_data = compute_social(trx, fn, pxpermm, n_parts, roi);
  return;
end

for fndx = 1:nflies
  x = trx(fndx).x; y = trx(fndx).y; theta = trx(fndx).theta;
  dt = trx(fndx).dt; nframes = trx(fndx).nframes;
  mod_apt_data = reshape(trx(fndx).kpts, n_parts, [], nframes);
  switch fn_type
    case 'global'
      part = str2double(parts{end});
      px = permute(mod_apt_data(part,1,:),[1,3,2]);
      py = permute(mod_apt_data(part,2,:),[1,3,2]);
      data = velmag(px,py,dt,pxpermm);   % only velmag used here
    case 'body'
      part = str2double(parts{end});
      px = permute(mod_apt_data(part,1,:),[1,3,2]);
      py = permute(mod_apt_data(part,2,:),[1,3,2]);
      [bx,by] = convert_to_body(px,py,x,y,theta);
      data = body_comp(bx,by,dt,comp_type,pxpermm);
    case 'pair'
      p2 = str2double(parts{end}); p1 = str2double(parts{end-1});
      x1 = permute(mod_apt_data(p1,1,:),[1,3,2]); y1 = permute(mod_apt_data(p1,2,:),[1,3,2]);
      x2 = permute(mod_apt_data(p2,1,:),[1,3,2]); y2 = permute(mod_apt_data(p2,2,:),[1,3,2]);
      [x1,y1] = convert_to_body(x1,y1,x,y,theta);
      [x2,y2] = convert_to_body(x2,y2,x,y,theta);
      data = pair_fn(x1,y1,x2,y2,dt,comp_type,pxpermm);
    case 'triad'
      p3 = str2double(parts{end}); p2 = str2double(parts{end-1}); p1 = str2double(parts{end-2});
      x1 = permute(mod_apt_data(p1,1,:),[1,3,2]); y1 = permute(mod_apt_data(p1,2,:),[1,3,2]);
      x2 = permute(mod_apt_data(p2,1,:),[1,3,2]); y2 = permute(mod_apt_data(p2,2,:),[1,3,2]);
      x3 = permute(mod_apt_data(p3,1,:),[1,3,2]); y3 = permute(mod_apt_data(p3,2,:),[1,3,2]);
      data = triad_fn(x1,x2,x3,y1,y2,y3,dt,comp_type,pxpermm);
  end
  all_data{fndx} = data;
end
end

function data = body_comp(x,y,dt,comp_type,pxpermm)
relative_fns = {'x','y','dx','dy','sin','cos','dtheta'};
if any(strcmp(relative_fns,comp_type))
  data = relative(x,y,zeros(size(x)),zeros(size(y)),dt,comp_type,pxpermm);
elseif strcmp(comp_type,'distcenter')
  data = dist_center(x,y,zeros(size(x)),zeros(size(y)),pxpermm);
elseif strcmp(comp_type,'ddistcenter')
  data = ddist_center(x,y,zeros(size(x)),zeros(size(y)),dt,pxpermm);
elseif strcmp(comp_type,'velmag')
  data = velmag(x,y,dt,pxpermm);
end
end

% ================= copied verbatim from compute_apt.m =================
function [b_x,b_y] = convert_to_body(x,y,trx_x,trx_y,theta)
b_x = nan(size(x)); b_y = nan(size(y));
for t = 1:numel(x)
  T = [1,0,0; 0,1,0; -trx_x(t),-trx_y(t),1];
  R = [cos(theta(t)-pi/2),-sin(theta(t)-pi/2),0
    sin(theta(t)-pi/2),cos(theta(t)-pi/2),0
    0,0,1];
  A = T*R; M = [x(t) y(t) 1]*A;
  b_x(t) = M(1); b_y(t) = M(2);
end
end

function data = velmag(x,y,dt,pxpermm)
if numel(x) == 1, data = 0; else
  dx = diff(x,1,2); dy = diff(y,1,2); data = sqrt(dx.^2 + dy.^2)./dt;
end
data = data/pxpermm;
end

function data = dist_center(x,y,x_body,y_body,pxpermm)
data = sqrt((x-x_body).^2 + (y-y_body).^2); data = data/pxpermm;
end

function data = ddist_center(x,y,x_body,y_body,dt,pxpermm)
if numel(x)>1, dist = sqrt((x-x_body).^2 + (y-y_body).^2); data = diff(dist)./dt;
else, data = 0; end
data = data/pxpermm;
end

function data = relative(x1,y1,x2,y2,dt,fn,pxpermm)
switch fn
  case 'x', data = (x1-x2)/pxpermm;
  case 'y', data = (y1-y2)/pxpermm;
  case 'dx', data = (diff(x1-x2)./dt)/pxpermm;
  case 'dy', data = (diff(y1-y2)./dt)/pxpermm;
  case 'cos'
    len = sqrt((x1-x2).^2 + (y1-y2).^2); data = (x1-x2)./len; data(len==0) = 0;
  case 'sin'
    len = sqrt((x1-x2).^2 + (y1-y2).^2); data = (y1-y2)./len; data(len==0) = 0;
  case 'dtheta'
    if numel(x1) > 2, theta = atan2(y1-y2, x1-x2); data = modrange(diff(theta,1,2),-pi,pi)./dt;
    else, data = []; end
end
end

function data = pair_fn(x1,y1,x2,y2,dt,fn,pxpermm)
relative_fns = {'x','y','dx','dy','sin','cos','dtheta'};
if any(strcmp(relative_fns,fn)), data = relative(x1,y1,x2,y2,dt,fn,pxpermm);
elseif strcmp(fn,'dist'), data = dist_center(x1,y1,x2,y2,pxpermm);
elseif strcmp(fn,'ddist'), data = ddist_center(x1,y1,x2,y2,dt,pxpermm);
elseif strcmp(fn,'areaswept')
  area = zeros(1,numel(x1));
  for ndx = 1:numel(x1)-1
    curx = [x1(ndx),x1(ndx+1),x2(ndx+1),x2(ndx)];
    cury = [y1(ndx), y1(ndx+1), y2(ndx+1), y2(ndx)];
    area(ndx) = polyarea(curx,cury);
  end
  data = area/pxpermm/pxpermm;
end
end

function data = triad_fn(x1,x2,x3,y1,y2,y3,dt,fn,pxpermm)
theta1 = atan2(y1-y2,x1-x2); theta2 = atan2(y3-y2,x3-x2);
theta = modrange(theta1-theta2,-pi,pi);
switch fn
  case 'cos', data = cos(theta);
  case 'sin', data = sin(theta);
  case 'dangle', data = diff(theta)./dt;
  case 'area'
    area = zeros(1,numel(x1));
    for ndx = 1:numel(x1), area(ndx) = polyarea([x1(ndx),x2(ndx),x3(ndx)],[y1(ndx),y2(ndx),y3(ndx)]); end
    data = area/pxpermm/pxpermm;
  case 'dlen'
    len1 = sqrt((x1-x2).^2 + (y1-y2).^2); len2 = sqrt((x3-x2).^2 + (y3-y2).^2);
    data = (len1-len2)/pxpermm;
end
end

% ================= copied from compute_apt_social.m =================
function all_data = compute_social(trx, fn, pxpermm, npts, roi)
parts = strsplit(fn,'_');
fn_type = parts{2}; comp_type = parts{3};
pt1 = str2double(parts{4});
nflies = numel(trx);
if strcmp(fn_type,'socialpair'), pt2 = str2double(parts{5}); else, pt2 = 1:npts; end
[dist,xy] = compute_distclosest(trx, pt1, pt2, npts, roi);
all_data = cell(1,nflies);
switch comp_type
  case 'dist'
    all_data = dist;
  case 'ddist'
    for ndx = 1:numel(dist), all_data{ndx} = diff(dist{ndx},1,2); end
  case {'sin','cos'}
    for ndx = 1:numel(dist)
      xyangle = atan2(xy{ndx}(2,:),xy{ndx}(1,:));
      dangle = xyangle - trx(ndx).theta;
      if strcmp(comp_type,'sin'), all_data{ndx} = sin(dangle); else, all_data{ndx} = cos(dangle); end
    end
end
end

function [data,dxyclosest_all] = compute_distclosest(trx, pt1, pt2, n_parts, roi)
flies = 1:numel(trx); nflies = numel(flies);
dclosest_all = cell(1,nflies); dxyclosest_all = cell(1,nflies);
for i1 = 1:nflies
  fly1 = flies(i1);
  flies2 = flies(roi(fly1)==roi(flies));
  dclosest = nan(numel(flies2),trx(fly1).nframes);
  dxy = nan(numel(flies2),2,trx(fly1).nframes);
  for i2 = 1:numel(flies2)
    fly2 = flies2(i2);
    if fly1 == fly2, continue; end
    [dcur,dxycur] = dapt_pair(trx,fly1,fly2,pt1,pt2,n_parts);
    dclosest(i2,:) = dcur; dxy(i2,:,:) = dxycur;
  end
  [dclosest_all{i1},closesti] = min(dclosest,[],1,'omitnan');
  dxycur = nan(2,trx(fly1).nframes);
  for ndx = 1:trx(fly1).nframes
    if isnan(dclosest_all{i1}(ndx)), continue; end
    dxycur(:,ndx) = dxy(closesti(ndx),:,ndx);
  end
  dxyclosest_all{i1} = dxycur;
end
data = dclosest_all;
end

function [dist_apt,dxy_apt] = dapt_pair(trx,fly1,fly2,pt1,pt2,n_pts)
dist_apt = nan(1,trx(fly1).nframes); dxy_apt = nan(2,trx(fly1).nframes);
t0 = max(trx(fly1).firstframe,trx(fly2).firstframe);
t1 = min(trx(fly1).endframe,trx(fly2).endframe);
if t1 < t0, return; end
i0 = t0 + trx(fly1).off; i1 = t1 + trx(fly1).off;
j0 = t0 + trx(fly2).off; j1 = t1 + trx(fly2).off;
nframes = trx(fly1).nframes; apt_data1 = reshape(trx(fly1).kpts,n_pts,[],nframes);
nframes2 = trx(fly2).nframes; apt_data2 = reshape(trx(fly2).kpts,n_pts,[],nframes2);
a1 = apt_data1(pt1,:,:); a2 = apt_data2(pt2,:,:);
if numel(pt2) > 1, a1 = repmat(a1,[numel(pt2),1,1]); end
dd = a2(:,:,j0:j1)-a1(:,:,i0:i1);
d = sqrt(sum(dd.^2,2));
[d,ix] = min(d,[],1,'omitnan');
if size(dd,1)>1
  for ndx = j0:j1, dxy_apt(:,i0+(ndx-j0)) = dd(ix(ndx-j0+1),:,ndx-j0+1); end
else
  dxy_apt(:,i0:i1) = reshape(dd,[size(dd,2),size(dd,3)]);
end
dist_apt(:,i0:i1) = d(1,:);
end
