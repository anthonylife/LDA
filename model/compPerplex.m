%COMPPERPLEX compute perplexity of whole corpus.

function perplex = compPerplex()
global Corp; global Model;
global Pz; global Pw_z; global Pd_z;

loghood = 0.0;
for i=1:Corp.nw
    loghood = loghood + Corp.X(:,i)'*log(Pd_z*diag(Pz)*Pw_z(i,:)');
end 

perplex = exp(-loghood/sum(sum(Corp.X)));
