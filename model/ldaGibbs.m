%LDAGIBBS achieves gibbs sampling algorithm to learn 
%  model parameters.
%
%  Procedures:
%    1.Setting some model parameters and global variables;
%    2.Loading documents information;
%    3.Randomly initialization topic for each word;
%    4.Iterative gibbs Sampling;
%    5.Evaluation by perplexity and log-likelihood;
%    6.Topic explation.
%
%  @author:anthonylife
%  @date:11/2/2012


clear all;
rand('state',sum(100*clock));
% 1----------------------
% Setting model parameter
% =======================
global Model;
Model.maxIter = 1000;
Model.burnIter = 500;
Model.diff = 1;
Model.topword=15;
Model.T = 20; %Model.alpha = 50/Model.T;
Model.alpha = 0.01;
Model.beta = 0.01;

% Setting corpus paramter
% =======================
global Corp;
Corp.featurefile = '../features/feature.txt';
Corp.dictfile = '../features/dict.txt';
Corp.nd = 0;
Corp.nw = 0;
Corp.N = 0;

% 2------------------------------
% Loading data source information
% ===============================
Corp.triple = load(Corp.featurefile);
Corp.X = spconvert(Corp.triple);
Corp.nd = size(Corp.X, 1);
Corp.nw = size(Corp.X, 2);
N = size(Corp.triple, 1);
for i=1:N,
    for j=1:Corp.triple(i,3),
        Corp.N = Corp.N + 1;
        D(Corp.N) = Corp.triple(i,1);
        W(Corp.N) = Corp.triple(i,2);
    end
end

% Setting doc-topic matrix and topic-word matrix
% ==============================================
global Pz; global Pd_z; global Pw_z;
global dt_mat; global tw_mat;
dt_mat = repmat(0, Corp.nd, Model.T);
tw_mat = repmat(0, Model.T, Corp.nw);


% 3------------------------------------------
% Randomly initialization topic for each word
% ===========================================
Z = floor(Model.T*rand(Corp.N,1)) + 1; 
for i=1:Corp.N,
    dt_mat(D(i), Z(i)) = dt_mat(D(i), Z(i))+1;
    tw_mat(Z(i), W(i)) = tw_mat(Z(i), W(i))+1;
end
Nt = sum(dt_mat, 1);


% 4-----------------------
% Iterative gibbs Sampling
% ========================
tic;
for i=1:Model.maxIter,
    fprintf('Gibbs sampling one iteration...\n');
    for j=1:Corp.N,
        t = Z(j);
        dt_mat(D(j), t) = dt_mat(D(j), t) - 1;
        tw_mat(t, W(j)) = tw_mat(t, W(j)) - 1;
        Nt(t) = Nt(t) - 1;

        for t=1:Model.T,
            probs(t) = (tw_mat(t,W(j))+Model.beta) / (Nt(t)+...
                Corp.nw*Model.beta) * (dt_mat(D(j),t) + Model.alpha);
        end

        probs = probs/sum(probs);
        cumprobs = cumsum(probs);
        t = find(cumprobs>rand, 1);

        Z(j) = t;
        dt_mat(D(j), t) = dt_mat(D(j), t) + 1;
        tw_mat(t, W(j)) = tw_mat(t, W(j)) + 1;
        Nt(t) = Nt(t) + 1;
    end
    Pz = sum(dt_mat, 1)'/Corp.N;
    Pd_z = dt_mat * diag(1./sum(dt_mat, 1));
    Pw_z = tw_mat' * diag(1./sum(tw_mat, 2));
    fprintf('Calculate likelihood and perplexity...\n');
    loghood = compLoghood();
    perplex = compPerplex(loghood);
    fprintf('Current iteration number: %d; Loglikelihood: %f; Perplexity: %f...\n', i, loghood, perplex);
end
fprintf('Total Time Cost:\n');
toc;

% 6.Topic explanation
% ===================
fprintf('Topic list:\n');
explaTopic();

fprintf('Perplexity of test data: %f...\n', perplex);
fprintf('Log-likelihood of test data: %f...\n', loghood);
fprintf('Finish.\n');

