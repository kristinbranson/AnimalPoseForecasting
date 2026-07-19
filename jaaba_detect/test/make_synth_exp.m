function make_synth_exp(outdir)
% Create a small synthetic 21-landmark experiment (registered_trx.mat + apttrk.mat)
% so the full detect.py pipeline can be run end-to-end (this jab needs 21 landmarks).
if ~exist(outdir,'dir'), mkdir(outdir); end
rng(2024);
nflies = 2; nframes = 600; npts = 21; pxpermm = 18.9; fps = 150;
trx = struct([]);
kp = cell(1,nflies);
for i = 1:nflies
  t = (0:nframes-1);
  cx = 500 + 120*cos(t/80 + i) + cumsum(randn(1,nframes))*0.5;
  cy = 500 + 120*sin(t/80 + i) + cumsum(randn(1,nframes))*0.5;
  th = atan2([diff(cy) 0],[diff(cx) 0]) + 0.05*randn(1,nframes);
  a = 13 + 0.5*randn(1,nframes); b = 5 + 0.2*randn(1,nframes);
  trx(i).x = cx; trx(i).y = cy; trx(i).theta = th;
  trx(i).a = a; trx(i).b = b;
  trx(i).x_mm = cx/pxpermm; trx(i).y_mm = cy/pxpermm; trx(i).theta_mm = th;
  trx(i).a_mm = a/pxpermm; trx(i).b_mm = b/pxpermm;
  trx(i).dt = ones(1,nframes-1)/fps; trx(i).fps = fps; trx(i).pxpermm = pxpermm;
  trx(i).firstframe = 1; trx(i).endframe = nframes; trx(i).nframes = nframes; trx(i).off = 0;
  trx(i).id = i; trx(i).sex = repmat({'?'},1,nframes);
  trx(i).xwingl=cx; trx(i).xwingr=cx; trx(i).ywingl=cy; trx(i).ywingr=cy;
  trx(i).timestamps = t/fps; trx(i).moviename=''; trx(i).matname=''; trx(i).moviefile='';
  % keypoints: npts around the fly, in the body frame then rotated to lab
  base = [linspace(-1.5,1.5,npts); 0.6*sin(linspace(0,pi,npts))];   % 2 x npts (in a-units)
  P = zeros(npts,2,nframes);
  for f = 1:nframes
    R = [cos(th(f)) -sin(th(f)); sin(th(f)) cos(th(f))];
    pts = R*(base.*a(f)) + [cx(f);cy(f)];
    P(:,:,f) = pts' + 0.3*randn(npts,2);
  end
  kp{i} = P;
end
timestamps = (0:nframes-1)/fps;
save(fullfile(outdir,'registered_trx.mat'),'trx','timestamps','-v7.3');
pTrk = kp; startframes = ones(1,nflies); endframes = nframes*ones(1,nflies);
trkInfo = struct('params',struct('n_classes',npts,'imsz',{{1024,1024}}));
save(fullfile(outdir,'apttrk.mat'),'pTrk','startframes','endframes','trkInfo','-v7.3');
fprintf('wrote synthetic experiment to %s (%d flies, %d frames, %d landmarks)\n', outdir, nflies, nframes, npts);
end
