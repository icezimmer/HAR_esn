[input_data, target_data] = dataLoader();
[dim, len, num] = size(input_data);
[num_classes, ~] = size(target_data);

seed_shuffle = 13;

%shuffle the dataset
rng(seed_shuffle)
shuffle = randperm(num);
input_data = input_data(:,:,shuffle);
target_data = target_data(:,shuffle);

dv_index = 1:64; % from 1 to 64
tr_index = 1:50; % from 1 to 50
vl_index = 51:64; % from 51 to 64
ts_index = 65:88; % from 65 to 88


%[dv_index, tr_index, vl_index, ts_index] = holdOut();

dv_in = input_data(:,:,dv_index);
tr_in = input_data(:,:,tr_index);
vl_in = input_data(:,:,vl_index);
ts_in = input_data(:,:,ts_index);

dv_tg = target_data(:,dv_index);
dv_tg = kron(dv_tg, ones(1,len));
tr_tg = target_data(:,tr_index);
tr_tg = kron(tr_tg, ones(1,len));
vl_tg = target_data(:,vl_index);
vl_tg = kron(vl_tg, ones(1,len));
ts_tg = target_data(:,ts_index);
ts_tg = kron(ts_tg, ones(1,len));

% Hyper-parameters
omega_in = 0.4;
rho = 0.9;
Nh = [10, 50, 100, 300, 500];
dns = 0.1;
a = [0.1, 0.3, 0.5, 0.7, 1];
lambda_r = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000];
ws = [0, 24];

tot = length(omega_in)*length(rho)*length(Nh)*length(dns)*length(a)*length(lambda_r)*length(ws);
r_guesses = 5;
one_hot = eye(num_classes);

% Model selection (by Grid search)
disp('Grid Search: 0%')
MAK_vl = -Inf;
MAK_tr = -Inf;
config = 0;
for i = 1:length(omega_in)
    for j = 1:length(rho)
        for k = 1:length(Nh)
            for l = 1:length(dns)
                for m = 1:length(a)
                    for n = 1:length(lambda_r)
                        for o = 1:length(ws)
                            meanAccuracy_K_vl = 0;
                            meanAccuracy_K_tr = 0;
                            for seed = 1:r_guesses
                                % TRAINING
                                hidden_tr = zeros(Nh(k),0);
    
                                for sample=1:size(tr_in,3)
                                    [~, sequence_tr] = rc(tr_in(:,:,sample), seed, omega_in(i), rho(j), Nh(k), dns(l), a(m), ws(o));
                                    hidden_tr = cat(2,hidden_tr,sequence_tr);
                                end
                                W_out = trainOffline(hidden_tr,tr_tg, lambda_r(n), ws(o));
    
                                y_tr = readout(hidden_tr,W_out);
                                [~, argmax_tr] = max(y_tr,[],1);
                                tr_pr = one_hot(:,argmax_tr);
                                [~, accuracy_K_tr] = evaluation(washout(tr_tg,ws(o)), tr_pr);
    
                                meanAccuracy_K_tr = meanAccuracy_K_tr + (accuracy_K_tr / r_guesses);
                                
                                % VALIDATION
                                hidden_vl = zeros(Nh(k),0);
                                
                                for sample=1:size(vl_in,3)
                                    [~, sequence_vl] = rc(vl_in(:,:,sample), seed, omega_in(i), rho(j), Nh(k), dns(l), a(m));
                                    hidden_vl = cat(2,hidden_vl,sequence_vl);
                                end
    
                                y_vl = readout(hidden_vl, W_out);
                                [~, argmax_vl] = max(y_vl,[],1);
                                vl_pr = one_hot(:,argmax_vl);
                                
                                [~, accuracy_K_vl] = evaluation(vl_tg, vl_pr);
                                
                                meanAccuracy_K_vl = meanAccuracy_K_vl + (accuracy_K_vl / r_guesses);
                            end
                            config = config+1;
                            disp(['Grid Search: ',num2str(100*(config/tot)),'%'])
                            if meanAccuracy_K_vl > MAK_vl
                                MAK_tr = meanAccuracy_K_tr;
                                MAK_vl = meanAccuracy_K_vl;
                                omega_in_best = omega_in(i);
                                rho_best = rho(j);
                                Nh_best = Nh(k);
                                dns_best = dns(l);
                                a_best = a(m);
                                ws_best = ws(o);
                                lambda_r_best = lambda_r(n);
                            end
                        end
                    end
                end
            end
        end
    end
end

% Refit on the development set
disp('Refit')
hidden_dv = zeros(Nh_best,0);
for sample=1:size(dv_in,3)
    [~, sequence_dv] = rc(dv_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, dns_best, a_best, ws_best);
    hidden_dv = cat(2,hidden_dv, sequence_dv);
end
W_out_best = trainOffline(hidden_dv, dv_tg, lambda_r_best, ws_best);

% Assessment on the test set
disp('Assessment')
hidden_ts = zeros(Nh_best,0);
for sample=1:size(ts_in,3)
    [~, sequence_ts] = rc(ts_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, dns_best, a_best);
    hidden_ts = cat(2,hidden_ts, sequence_ts);
end
y_ts = readout(hidden_ts, W_out_best);
[~, argmax_ts] = max(y_ts,[],1);
ts_pr = one_hot(:,argmax_ts);
[~, accuracy_K_ts, accuracy_ts, accuracy_av_ts, F1_ts, F1_macro_ts] = evaluation(ts_tg, ts_pr);

% Plot Confusion Matrix
[classes_target, ~] = find(ts_tg);
[classes_predict, ~] = find(ts_pr);
gcf = figure;
confusionchart(classes_target, classes_predict);
title("Confusion Matrix (TS set)")

% Save plot and net structure
saveas(gcf, fullfile('results', strcat('LIESN_confusionMatrix', '.png')))
save(fullfile('results', strcat('LIESN_hyperparameters', '.mat')), 'seed', 'omega_in_best', 'rho_best', 'Nh_best', 'dns_best', 'a_best', 'lambda_r_best', 'ws_best')
save(fullfile('results', strcat('LIESN_readOutWeights', '.mat')), 'W_out_best')

% Save performance
save(fullfile('results', strcat('LIESN_performanceTR', '.mat')), 'MAK_tr')
save(fullfile('results', strcat('LIESN_performanceVL', '.mat')), 'MAK_vl')
save(fullfile('results', strcat('LIESN_performanceTS', '.mat')), 'accuracy_K_ts', 'accuracy_ts', 'accuracy_av_ts', 'F1_ts', 'F1_macro_ts')