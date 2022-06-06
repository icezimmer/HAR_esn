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

[dv_layer_in, dv_layer_tg] = dataProcess(dv_in, dv_tg);
[tr_layer_in, tr_layer_tg] = dataProcess(tr_in, tr_tg);
[vl_layer_in, vl_layer_tg] = dataProcess(vl_in, vl_tg);
[ts_layer_in, ts_layer_tg] = dataProcess(ts_in, ts_tg);

% Hyperparameters
len_window = 10;
Nh = [10, 50, 100, 500]; % best 100
eta = [0.0001, 0.002, 0.0025, 0.003, 0.005, 0.01, 0.1];
alpha = [0, 0.001, 0.0001];
weight_decay = [0, 0.01, 0.001, 0.0001, 0.00001]; % best 0.00001
max_epochs = 5000;
early_stopping = 10;

tot = length(len_window)*length(Nh)*length(eta)*length(alpha)*length(weight_decay)*length(max_epochs)*length(early_stopping);
num_runs = 5;

% Model selection (by Grid search)
disp('Grid Search: 0%')
minimum_vl = -Inf;
config = 0;
for i=1:length(len_window)
    for j=1:length(Nh)
        for k=1:length(eta)
            for l=1:length(alpha)
                for m=1:length(weight_decay)
                    for n=1:length(max_epochs)
                        for o=1:length(early_stopping)
                            meanAccuracy_K_tr = 0;
                            meanAccuracy_K_vl = 0;
                            for seed=1:num_runs
                                rng(seed)
                                wf = @(sz) 0.001*(2*rand(sz)-1);
                                layers = [ ...
                                    sequenceInputLayer(dim)
                                    convolution1dLayer(len_window(i), Nh(j), Padding="causal", WeightsInitializer=wf) %num. of filters = num. of hidden neurons
                                    tanhLayer
                                    fullyConnectedLayer(num_classes)
                                    softmaxLayer
                                    classificationLayer];
                                options = trainingOptions("sgdm", ...
                                    MiniBatchSize=size(tr_layer_tg, 1), ...
                                    InitialLearnRate=eta(k), ...
                                    Momentum=alpha(l), ...
                                    L2Regularization=weight_decay(m), ...
                                    MaxEpochs=max_epochs(n), ...
                                    ValidationPatience=early_stopping(o), ...
                                    OutputNetwork="best-validation-loss", ...
                                    shuffle='never', ...
                                    ValidationData={vl_layer_in,vl_layer_tg}, ...
                                    Verbose=0);
                                [net, info] = trainNetwork(tr_layer_in,tr_layer_tg,layers,options);
                                accuracy_K_tr = info.TrainingAccuracy(info.OutputNetworkIteration);
                                meanAccuracy_K_tr = meanAccuracy_K_tr + (accuracy_K_tr / num_runs);
                                accuracy_K_vl = info.FinalValidationAccuracy;
                                meanAccuracy_K_vl = meanAccuracy_K_vl + (accuracy_K_vl / num_runs);
                            end
                            config = config+1;
                            disp(['Grid Search: ',num2str(100*(config/tot)),'%'])
                            if meanAccuracy_K_vl > minimum_vl
                                minimum_tr = meanAccuracy_K_tr;
                                minimum_vl = meanAccuracy_K_vl;
                                layers_best = layers;
                                options_best = options;
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
options_best.MiniBatchSize = size(dv_layer_tg, 1);
options_best.ValidationData = [];
options_best.ValidationPatience = Inf;
options_best.OutputNetwork = 'last-iteration';
options_best.Verbose = 1;
options_best.VerboseFrequency = 1;
net_best = trainNetwork(dv_layer_in,dv_layer_tg,layers_best,options_best);

% Assessment on the test set
disp('Assessment')
ts_layer_pr = classify(net_best, ts_layer_in, SequencePaddingDirection="left");
[~, accuracy_K_ts, accuracy_ts, accuracy_av_ts, F1_ts, F1_macro_ts] = evaluation(ts_layer_tg, ts_layer_pr);

% Plot Confusion Matrix 
gcf = figure;
confusionchart([ts_layer_tg{:,:}], [ts_layer_pr{:,:}]);
title("Confusion Matrix (TS set)")

% Save plot and net structure
saveas(gcf, fullfile('results', strcat('1DCONV_confusionMatrix', '.png')))
save(fullfile('results', strcat('1DCONV_net', '.mat')), 'net_best')
save(fullfile('results', strcat('1DCONV_options', '.mat')), 'options_best')

% Save performance
save(fullfile('results', strcat('1DCONV_performanceTR', '.mat')), 'minimum_tr')
save(fullfile('results', strcat('1DCONV_performanceVL', '.mat')), 'minimum_vl')
save(fullfile('results', strcat('1DCONV_performanceTS', '.mat')), 'accuracy_K_ts', 'accuracy_ts', 'accuracy_av_ts', 'F1_ts', 'F1_macro_ts')
