% LOAD TEST SET
[input_data, target_data] = dataLoader();
[dim, len, num] = size(input_data);
[num_classes, ~] = size(target_data);
seed_shuffle = 13;
rng(seed_shuffle) %shuffle the dataset
shuffle = randperm(num);
input_data = input_data(:,:,shuffle);
target_data = target_data(:,shuffle);
ts_index = 65:88; % from 65 to 88
ts_in = input_data(:,:,ts_index);
ts_tg = target_data(:,ts_index);

[cls,~] = find(ts_tg);

ts_tg = kron(ts_tg, ones(1,len));
ts_tg = reshape(ts_tg, [num_classes, len, size(ts_in,3)]);

% LOAD MODEL
S=load('results/LIESN_readOutWeights.mat');
W_out = S.W_out_best;

% LOAD HYPERPARAMETERS
S=load('results/LIESN_hyperparameters.mat');
Nh = S.Nh_best;
a = S.a_best;
dns = S.dns_best;
lambda_r = S.lambda_r_best;
omega_in = S.omega_in_best;
rho = S.rho_best;
seed_esn = S.seed;
ws = S.ws_best;

% TEST DATA STREAMING
time_steps_perclass = 100;
real_time_target = zeros(1,0);
one_hot = eye(num_classes);
real_time_predict = zeros(1,0);
pooler = zeros(Nh, 1);

start = tic;
for k=1:num_classes
    sample=find(cls==k,1);
    for j=1:time_steps_perclass 
        rng('shuffle')
        time = randi([1,size(ts_in,2)]);
        input_t = ts_in(:,time,sample);
    
        [~, ~, pooler] = rc(input_t, seed_esn, omega_in, rho, Nh, dns, a, ws, pooler);
        y = readout(pooler, W_out);
        [~, argmax] = max(y,[],1);
        [predict_t, ~] = find(one_hot(:,argmax));
        real_time_predict = cat(2, real_time_predict, predict_t);
        
        [target_t, ~] = find(ts_tg(:,time,sample));
        real_time_target = cat(2, real_time_target, target_t);
    end
end
elapsed = toc(start);

time_per_op = elapsed / (num_classes * time_steps_perclass);

gcf = figure;
p1 = plot(real_time_target, '-k');
hold on
p2 = plot(real_time_predict, '-r');
hold off
legend('Target', 'Predict')
xlabel('Timestep')
ylabel('Class')
title('Real Time prediction')

saveas(gcf, fullfile('results', strcat('LIESN_realtime', '.png')))
save(fullfile('results', strcat('LIESN_timePerOp', '.mat')), 'time_per_op')