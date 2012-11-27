%EXPLATOPIC explain topics by sorting words accroding to their occuring
%  probability given corresponding topic.

function explaTopic()
global Corp; global Model;
global Pw_z;

dict = textread();
if Model.K < 10,
for i=1:Model.K,
    [temp, order_idx] = sort(Pw_z(:,i), 'descend');
    fprintf('Topic %d:\n', i);

    for j=1:Model.topword,
        fprintf('%s : %f\n',dict(order_idx(j)).word,Pw_z(order_idx(j),i));
    end
end
end
