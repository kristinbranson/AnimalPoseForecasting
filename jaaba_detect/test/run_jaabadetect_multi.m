function run_jaabadetect_multi(expdir, jabfiles, outmat)
% Run real JAABADetect for several jabs on one experiment (shared per-frame
% features computed once), and save each classifier's allScores for comparison.
addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe');
SetUpJAABAPath;
t = tic;
[classifierinfo, allScores] = JAABADetect(expdir, 'jabfiles', jabfiles, 'forcecompute', true);
fprintf('JAABADetect (%d jabs) done in %.1f s\n', numel(jabfiles), toc(t));
S = struct();
S.classifierinfo = classifierinfo;
for k = 1:size(allScores, 1)
  S.(sprintf('allScores_%d', k)) = allScores{k, 1};
  S.(sprintf('behavior_%d', k)) = classifierinfo(k).behavior;
end
save(outmat, '-struct', 'S', '-v7.3');
fprintf('wrote %s\n', outmat);
end
