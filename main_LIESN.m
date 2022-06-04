[input_data, target_data] = dataLoader();
[dim, len, num] = size(input_data);

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

dv_in = input_data(:,:,dv_index);
tr_in = input_data(:,:,tr_index);
vl_in = input_data(:,:,vl_index);
ts_in = input_data(:,:,ts_index);

dv_tg = target_data(:,dv_index); 
tr_tg = target_data(:,tr_index);
vl_tg = target_data(:,vl_index);
ts_tg = target_data(:,ts_index);

% Hyper-parameters

omega_in = 0.4;
rho = 0.9;
Nh = [10, 50, 100, 300, 500];
dns = 0.1;
a = [0.1, 0.3, 0.5, 0.7, 1];
lambda_r = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000];

%{
omega_in = 0.4;
rho = 0.9;
Nh = 100;
dns = 0.1;
a = 0.1;
lambda_r = 0.0001;
%}

tot = length(omega_in)*length(rho)*length(Nh)*length(dns)*length(a)*length(lambda_r);
r_guesses = 5;
%{
meanLoss_tr = 0;
meanAccuracy_K_tr = 0;
meanAccuracy_tr = zeros(7,1);
meanAccuracy_av_tr = 0;

meanLoss_vl = 0;
meanAccuracy_K_vl = 0;
meanAccuracy_vl = zeros(7,1);
meanAccuracy_av_vl = 0;
%}
aux = eye(7);

% Model selection (by Grid search)
minimum_vl = -Inf;
config = 0;
for i = 1:length(omega_in)
    for j = 1:length(rho)
        for k = 1:length(Nh)
            for l = 1:length(dns)
                for m = 1:length(a)
                    for n = 1:length(lambda_r)
                        meanAccuracy_K_vl = 0;
                        config = config+1;
                        disp([num2str(100*(config/tot)),'%'])
                        for seed = 1:r_guesses
                            % TRAINING
                            hidden_tr = zeros(Nh(k),0);
                            
                            for sample=1:size(tr_in,3)
                                [~, ~, pooler_tr] = rc(tr_in(:,:,sample), seed, omega_in(i), rho(j), Nh(k), dns(l), a(m));
                                hidden_tr = cat(2,hidden_tr,pooler_tr);
                            end
                            
                            W_out = trainOffline(hidden_tr,tr_tg, lambda_r(n));

                            y_tr = readout(hidden_tr,W_out);
                            [~, argmax_tr] = max(y_tr,[],1);
                            tr_pr = aux(:,argmax_tr);
                            [~, accuracy_K_tr, accuracy_tr, accuracy_av_tr, F1_tr, F1_macro_tr, support_tr, condusionMatrix_tr] = evaluation(tr_tg, tr_pr)
                            %[~, accuracy_K_tr] = evaluation(tr_tg, tr_pr);
                        
                            %meanLoss_tr = meanLoss_tr + (loss_tr / r_guesses);
                            %meanAccuracy_K_tr = meanAccuracy_K_tr + (accuracy_K_tr / r_guesses);
                            %meanAccuracy_tr = meanAccuracy_tr + (accuracy_tr / r_guesses);
                            %meanAccuracy_av_tr = meanAccuracy_av_tr + (accuracy_av_tr / r_guesses);
                            %meanF1_macro_tr = meanF1_macro_tr + (F_1_macro_tr / r_guesses);
                            
                            % VALIDATION
                            hidden_vl = zeros(Nh(k),0);
                            
                            for sample=1:size(vl_in,3)
                                [~, ~, pooler_vl] = rc(vl_in(:,:,sample), seed, omega_in(i), rho(j), Nh(k), dns(l), a(m));
                                hidden_vl = cat(2,hidden_vl,pooler_vl);
                            end

                            y_vl = readout(hidden_vl, W_out);
                            [~, argmax_vl] = max(y_vl,[],1);
                            vl_pr = aux(:,argmax_vl);
                            
                            [~, accuracy_K_vl] = evaluation(vl_tg, vl_pr);
                            
                            %meanLoss_vl = meanLoss_vl + (loss_vl / r_guesses);
                            meanAccuracy_K_vl = meanAccuracy_K_vl + (accuracy_K_vl / r_guesses);
                            %meanAccuracy_vl = meanAccuracy_vl + (accuracy_vl / r_guesses);
                            %meanAccuracy_av_vl = meanAccuracy_av_vl + (accuracy_av_vl / r_guesses);
                        end
                        if meanAccuracy_K_vl > minimum_vl
                            minimum_vl = meanAccuracy_K_vl;
                            omega_in_best = omega_in(i);
                            rho_best = rho(j);
                            Nh_best = Nh(k);
                            dns_best = dns(l);
                            a_best = a(m);
                            lambda_r_best = lambda_r(n);
                        end
                    end
                end
            end
        end
    end
end

% Refit on the development set
hidden_dv = zeros(Nh_best,0);
for sample=1:size(dv_in,3)
    [~, ~, pooler_dv] = rc(dv_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, dns_best, a_best);
    hidden_dv = cat(2,hidden_dv, pooler_dv);
end
W_out_refit = trainOffline(hidden_dv, dv_tg, lambda_r_best);

% Assessment on the test set
hidden_ts = zeros(Nh_best,0);
for sample=1:size(ts_in,3)
    [~, ~, pooler_ts] = rc(ts_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, dns_best, a_best);
    hidden_ts = cat(2,hidden_ts, pooler_ts);
end
y_ts = readout(hidden_ts, W_out_refit);
[~, argmax_ts] = max(y_ts,[],1);
ts_pr = aux(:,argmax_ts);
[~, accuracy_K_ts, accuracy_ts, accuracy_av_ts, F1_ts, F1_macro_ts, ~, confusionMatrix_ts] = evaluation(ts_tg, ts_pr)

% Plot Confusion Matrix 
figure
b = bar3(confusionMatrix_ts,0.5);      % Specify bar width in the third argument
for k = 1:length(b)
    zdata = b(k).ZData;                 % Use ZData property to create color gradient
    b(k).CData = zdata;                 % Set CData property to Zdata
    b(k).FaceColor = "interp";          % Set the FaceColor to 'interp' to enable the gradient 
end
title("Confusion Matrix (TS set)")
xlabel("Predict")
ylabel("Target")
zlabel("#Samples")

%{
disp('Estimation of Loss and Accuracy over different Reservoir guesses:')
disp('TRAINING:')
disp(['Loss: ', num2str(meanLoss_tr)]);
disp(['Accuracy: ', num2str(meanAccuracy_K_tr)]);
disp('Accuracy per class with support:')
[meanAccuracy_tr, support_tr]
disp(['Average Accuracy: ', num2str(meanAccuracy_av_tr)]);
disp('VALIDATION:')
disp(['Loss: ', num2str(meanLoss_vl)])
disp(['Accuracy: ', num2str(meanAccuracy_K_vl)]);
[meanAccuracy_vl, support_vl]
disp(['Average Accuracy: ', num2str(meanAccuracy_av_vl)]);
%}