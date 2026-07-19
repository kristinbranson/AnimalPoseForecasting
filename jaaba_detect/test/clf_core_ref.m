function clf_core_ref(jabfile, outfile)
% Ground truth for the classifier core: myBoostClassify + PostProcessor.PostProcess
% on a random feature matrix, using the real trained model.
addpath('/groups/branson/home/bransonk/behavioranalysis/code/JAABA/perframe'); SetUpJAABAPath;
Q = loadAnonymous(jabfile); cs = Q.classifierStuff;
params = cs.params;
dims = [params.dim];
udims = unique(dims);
rng(777);
N = 300;
Xu = randn(N, numel(udims)) * 2;          % random feature matrix over the used dims
% remap model dims to columns of Xu
model2 = params;
[~, loc] = ismember(dims, udims);
for j = 1:numel(model2), model2(j).dim = loc(j); end
scores = myBoostClassify(Xu, model2);
posts = PostProcessor.PostProcess(scores(:)', cs.postProcessParams, cs.scoreNorm);
save(outfile, 'Xu', 'udims', 'scores', 'posts', '-v7');
fprintf('wrote %s (N=%d, nudim=%d)\n', outfile, N, numel(udims));
end
