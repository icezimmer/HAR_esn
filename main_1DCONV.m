[input_data, target_data] = dataLoader();
[dim, len, num] = size(input_data);
[~, num_classes] = size(target_data);

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

tr_in=num2cell(tr_in,[1,2]);
tr_in=reshape(tr_in,[size(tr_in,1) * size(tr_in,3), 1]);

[tr_tg, ~] = find(tr_tg);
tr_tg = categorical(tr_tg);


%tr_tg = reshape(tr_tg,[size(tr_tg,1),1,size(tr_tg,2)]);

% Hyperparameters
len_window = 10;
eta = [0.0001, 0.002, 0.0025, 0.003, 0.005, 0.01, 0.1];
alpha = [0, 0.001, 0.0001];
weight_decay = [0, 0.01, 0.001, 0.0001, 0.00001]; % best 0.00001
Nh = [10, 50, 100, 500]; % best 100
max_epochs = 5000;
early_stopping = 10;

num_runs = 5;

layers = [ ...
    sequenceInputLayer(dim)
    convolution1dLayer(len_window, 100, Padding="causal") %num. of filters = num. of hidden neurons
    tanhLayer
    layerNormalizationLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer];

options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    SequencePaddingDirection="left", ...
    Verbose=1);

%{
options = trainingOptions('adam', ...
    'MaxEpochs',max_epochs, ...
    'Shuffle','never', ...
    'Verbose',0);
%}

net = trainNetwork(tr_in,tr_tg,layers,options);

tr_pr = classify(net, tr_in, SequencePaddingDirection="left");

confusionchart(tr_tg, tr_pr);