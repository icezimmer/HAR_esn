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
Nh = 100;
% InitialLearnRate 0.001 (default)
% L2Regularization 0.0001 (default)
max_epochs = [50, 100];
early_stopping = 10;

tot = length(Nh)*length(max_epochs)*length(early_stopping);

% Model selection (by Grid search)
disp('Grid Search: 0%')
minimum_vl = -Inf;
config = 0;
for j=1:length(Nh)
    for n=1:length(max_epochs)
        for o=1:length(early_stopping)
            meanAccuracy_K_tr = 0;
            meanAccuracy_K_vl = 0;
            layers = [ ...
                sequenceInputLayer(dim)
                gruLayer(Nh(j), OutputMode="sequence")
                fullyConnectedLayer(num_classes)
                softmaxLayer
                classificationLayer];
            options = trainingOptions('adam', ...
                MiniBatchSize=size(tr_layer_tg, 1), ...
                MaxEpochs=max_epochs(n), ...
                GradientThreshold=2, ...
                shuffle='never', ...
                ValidationData={vl_layer_in,vl_layer_tg}, ...
                ValidationPatience=early_stopping(o), ...
                OutputNetwork="best-validation-loss", ...
                Verbose=0);
            [net, info] = trainNetwork(tr_layer_in,tr_layer_tg,layers,options);
            accuracy_K_tr = info.TrainingAccuracy(info.OutputNetworkIteration);
            accuracy_K_vl = info.FinalValidationAccuracy;
            config = config+1;
            disp(['Grid Search: ',num2str(100*(config/tot)),'%'])
            if accuracy_K_vl > minimum_vl
                minimum_tr = accuracy_K_tr;
                minimum_vl = accuracy_K_vl;
                layers_best = layers;
                options_best = options;
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
options_best.VerboseFrequency=1;
net_best = trainNetwork(dv_layer_in,dv_layer_tg,layers_best,options_best);

% Assessment on the test set
disp('Assessment')
ts_layer_pr = classify(net_best, ts_layer_in);
[~, accuracy_K_ts, accuracy_ts, accuracy_av_ts, F1_ts, F1_macro_ts] = evaluation(ts_layer_tg, ts_layer_pr);

% Plot Confusion Matrix 
gcf = figure;
confusionchart([ts_layer_tg{:,:}], [ts_layer_pr{:,:}]);
title("Confusion Matrix (TS set)")

% Save plot and net structure
saveas(gcf, fullfile('results', strcat('GRU_confusionMatrix', '.png')))
save(fullfile('results', strcat('GRU_net', '.mat')), 'net_best')
save(fullfile('results', strcat('GRU_options', '.mat')), 'options_best')

% Save performance
save(fullfile('results', strcat('GRU_performanceTR', '.mat')), 'minimum_tr')
save(fullfile('results', strcat('GRU_performanceVL', '.mat')), 'minimum_vl')
save(fullfile('results', strcat('GRU_performanceTS', '.mat')), 'accuracy_K_ts', 'accuracy_ts', 'accuracy_av_ts', 'F1_ts', 'F1_macro_ts')

