[input_data, target_data] = dataLoader();
[dim, len, num] = size(input_data);

seed = 3;

% shuffle the dataset
rng(seed)
shuffle = randperm(num);
input_data = input_data(:,:,shuffle);
target_data = target_data(:,shuffle);


dv_index = 1:floor(0.8*num); % from 1 to 70
tr_index = 1:floor(0.8*0.8*num); % from 1 to 56
vl_index = floor(0.8*0.8*num)+1:floor(0.8*num); % from 57 to 70
ts_index = floor(0.8*num)+1:num; % from 71 to 88

dv_in = input_data(:,:,dv_index);
tr_in = input_data(:,:,tr_index);
vl_in = input_data(:,:,vl_index);
ts_in = input_data(:,:,ts_index);

dv_tg = target_data(:,dv_index); 
tr_tg = target_data(:,tr_index);
vl_tg = target_data(:,vl_index);
ts_tg = target_data(:,ts_index);


% TRAINING
omega_in = 0.9;
Nh = 100;
rho = 0.9;

hidden_tr = zeros(Nh,0);
aux = eye(7);

for i=1:size(tr_in,3)
    [~, ~, pooler_tr] = rc(tr_in(:,:,i), omega_in, Nh, rho, seed);
    hidden_tr = cat(2,hidden_tr,pooler_tr);
end

hidden_tr(:,2)=ones(Nh,1); % MUST SOLVE!!!
W_out = trainOffline(hidden_tr,tr_tg);
y_tr = readout(hidden_tr,W_out);

[~, argmax_tr] = max(y_tr,[],1);
tr_pr = aux(:,argmax_tr);

loss_tr = immse(tr_tg, y_tr)
accuracy_tr = nnz(min(tr_tg==tr_pr,[],1)) / size(tr_tg,2)

% VALIDATION
hidden_vl = zeros(Nh,0);

for i=1:size(vl_in,3)
    [~, ~, pooler_vl] = rc(vl_in(:,:,i), omega_in, Nh, rho, seed);
    hidden_vl = cat(2,hidden_vl,pooler_vl);
end

hidden_vl(:,2)=ones(Nh,1);
W_out = trainOffline(hidden_vl,vl_tg);
y_vl = readout(hidden_vl,W_out);

[~, argmax_vl] = max(y_vl,[],1);
vl_pr = aux(:,argmax_vl);

loss_vl = immse(vl_tg, y_vl)
accuracy_vl = nnz(min(vl_tg==vl_pr,[],1)) / size(vl_tg,2)