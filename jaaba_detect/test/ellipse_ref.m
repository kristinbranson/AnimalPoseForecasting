function ellipse_ref(trxfile, outfile)
% Reference for the tractable ellipse per-frame features (b_mm, ecc, dv_tail,
% dcenter, dnose2tail), copying the exact JAABA arithmetic, on real registered_trx.
Lt = load(trxfile); trxin = Lt.trx; nflies = numel(trxin);
trx = struct('x_mm',{},'y_mm',{},'theta_mm',{},'theta',{},'a_mm',{},'b_mm',{},...
             'dt',{},'nframes',{},'firstframe',{},'endframe',{},'off',{});
for i = 1:nflies
  trx(i).x_mm = trxin(i).x_mm(:)'; trx(i).y_mm = trxin(i).y_mm(:)';
  trx(i).theta_mm = trxin(i).theta_mm(:)'; trx(i).theta = trxin(i).theta(:)';
  trx(i).a_mm = trxin(i).a_mm(:)'; trx(i).b_mm = trxin(i).b_mm(:)';
  trx(i).dt = trxin(i).dt(:)'; trx(i).nframes = double(trxin(i).nframes);
  trx(i).firstframe = double(trxin(i).firstframe); trx(i).endframe = double(trxin(i).endframe);
  trx(i).off = double(trxin(i).off);
end
roi = ones(1,nflies);

S = struct(); S.feat_names = {'b_mm','ecc','dv_tail','dcenter','dnose2tail'};
S.f1 = arrayfun(@(t) {t.b_mm}, trx); S.f1 = {trx.b_mm};
S.f2 = cellfun(@(a,b) b./a, {trx.a_mm}, {trx.b_mm}, 'uni', 0);
S.f3 = arrayfun(@(t) {dv_tail(t)}, trx); S.f3 = cellfun(@(t) t, S.f3, 'uni', 0);
% dcenter, dnose2tail: min over flies2
dc = cell(1,nflies); dn = cell(1,nflies);
for i1 = 1:nflies
  flies2 = find(roi(i1)==roi);
  Dc = inf(numel(flies2),trx(i1).nframes);
  Dn = nan(numel(flies2),trx(i1).nframes);
  for k = 1:numel(flies2)
    i2 = flies2(k); if i1==i2, continue; end
    Dc(k,:) = dcenter_pair(trx,i1,i2);
    Dn(k,:) = dnose2tail_pair(trx,i1,i2);
  end
  dc{i1} = min(Dc,[],1);
  dn{i1} = min(Dn,[],1);
end
S.f4 = dc; S.f5 = dn;
save(outfile,'-struct','S','-v7');
fprintf('wrote %s\n', outfile);
end

function data = dv_tail(t)
if t.nframes < 2, data = []; return; end
tailx = t.x_mm + 2*cos(-t.theta).*t.a_mm;
taily = t.y_mm + 2*sin(-t.theta).*t.a_mm;
dx = diff(tailx,1,2); dy = diff(taily,1,2);
data = dx.*cos(t.theta_mm(1:end-1)+pi/2) + dy.*sin(t.theta_mm(1:end-1)+pi/2)./t.dt;
end

function dcenter = dcenter_pair(trx,fly1,fly2)
dcenter = nan(1,trx(fly1).nframes);
t0 = max(trx(fly1).firstframe,trx(fly2).firstframe);
t1 = min(trx(fly1).endframe,trx(fly2).endframe);
if t1 < t0, return; end
i0 = t0 + trx(fly1).off; i1 = t1 + trx(fly1).off;
j0 = t0 + trx(fly2).off; j1 = t1 + trx(fly2).off;
dx = trx(fly2).x_mm(j0:j1)-trx(fly1).x_mm(i0:i1);
dy = trx(fly2).y_mm(j0:j1)-trx(fly1).y_mm(i0:i1);
dcenter(i0:i1) = sqrt(dx.^2 + dy.^2);
end

function d = dnose2tail_pair(trx,fly1,fly2)
d = nan(1,trx(fly1).nframes);
t0 = max(trx(fly1).firstframe,trx(fly2).firstframe);
t1 = min(trx(fly1).endframe,trx(fly2).endframe);
if t1 < t0, return; end
i0 = t0 + trx(fly1).off; i1 = t1 + trx(fly1).off;
j0 = t0 + trx(fly2).off; j1 = t1 + trx(fly2).off;
xnose = trx(fly1).x_mm(i0:i1) + 2*trx(fly1).a_mm(i0:i1).*cos(trx(fly1).theta_mm(i0:i1));
ynose = trx(fly1).y_mm(i0:i1) + 2*trx(fly1).a_mm(i0:i1).*sin(trx(fly1).theta_mm(i0:i1));
xtail = trx(fly2).x_mm(j0:j1) - 2*trx(fly2).a_mm(j0:j1).*cos(trx(fly2).theta_mm(j0:j1));
ytail = trx(fly2).y_mm(j0:j1) - 2*trx(fly2).a_mm(j0:j1).*sin(trx(fly2).theta_mm(j0:j1));
d(i0:i1) = sqrt((xtail-xnose).^2 + (ytail-ynose).^2);
end
