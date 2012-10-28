clear all

%--------------------------------------------
% parameters
%--------------------------------------------
T     = 8;     % number of topics
beta  = 0.01;  % Dirichlet prior on words
alpha = 0.1;   % Dirichlet prior on topics
rand('state',sum(100*clock));


%--------------------------------------------
% read corpus
%--------------------------------------------
[did,wid,cnt] = textread('docword.txt','%d%d%d','headerlines',3);
X = sparse(did,wid,cnt);
D = max(did);       % number of docs
W = max(wid);       % size of vocab
N = sum(cnt);       % total number of words
[word] = textread('vocab.txt','%s');
assert(length(word)==W)


%--------------------------------------------
% allocate memory
%--------------------------------------------
w = zeros(N,1);
d = zeros(N,1);
z = zeros(N,1);
Nwt = zeros(W,T);
Ndt = zeros(D,T);


%--------------------------------------------
% fill w and d
%--------------------------------------------
count = 1;
for j = 1:length(cnt)
  for i = count:count+cnt(j)-1 
    w(i) = wid(j);
    d(i) = did(j);
  end
  count = count + cnt(j);
end
assert(max(w)==max(wid))
assert(max(d)==max(did))
assert(count-1==N)


%--------------------------------------------
% random initial assignment
%--------------------------------------------
z = floor(T*rand(N,1)) + 1;
for n = 1:N
  Nwt(w(n),z(n)) = Nwt(w(n),z(n)) + 1;
  Ndt(d(n),z(n)) = Ndt(d(n),z(n)) + 1;
end
Nt    = sum(Nwt,1);
Ntchk = sum(Ndt,1);
assert(norm(Nt-Ntchk)==0)
assert(sum(Nt)==N)

%--------------------------------------------
% gibbs sampler
%--------------------------------------------
for iter = 1:200

  for i = 1:N
  
    t = z(i);
    Nwt(w(i),t) = Nwt(w(i),t) - 1;
    Ndt(d(i),t) = Ndt(d(i),t) - 1;
    Nt(t)       = Nt(t)       - 1;

    for t = 1:T
      probs(t) = (Nwt(w(i),t) + beta)/(Nt(t) + W*beta) * (Ndt(d(i),t) + alpha);
    end

    probs = probs/sum(probs);
    cumprobs = cumsum(probs);
    t = find(cumprobs>rand,1);

    z(i) = t;
    Nwt(w(i),t) = Nwt(w(i),t) + 1;
    Ndt(d(i),t) = Ndt(d(i),t) + 1;
    Nt(t)       = Nt(t)       + 1;
    
  end
  
  fprintf('iter %d \n', iter);
  
  %--------------------------------------------
  % topics
  %--------------------------------------------
  for t = 1:T
    fprintf('\t[%d] (%.3f) ', t, Nt(t)/N);
    [xsort,isort] = sort(-Nwt(:,t));
    for k = 1:8
      fprintf('%s ', word{ isort(k) } );
    end
    fprintf('\n');
  end
  
end
