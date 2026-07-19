function run_jaabadetect(expdir, jabfile, outmat)
% Run the real JAABADetect to generate ground-truth scores for the Python port.
addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe');
SetUpJAABAPath;
fprintf('Running JAABADetect on %s\n', expdir);
t = tic;
[classifierinfo, allScores] = JAABADetect(expdir, 'jabfiles', {jabfile}, 'forcecompute', true);
fprintf('JAABADetect done in %.1f s\n', toc(t));
S = struct();
S.allScores = allScores{1,1};
S.classifierinfo = classifierinfo;
save(outmat, '-struct', 'S', '-v7.3');
fprintf('wrote %s\n', outmat);
end
