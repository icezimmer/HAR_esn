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

tr_tg = reshape(tr_tg,[size(tr_tg,1),1,size(tr_tg,2)]);

% Hyperparameters
len_window = 10;
eta = [0.0001, 0.002, 0.0025, 0.003, 0.005, 0.01, 0.1];
alpha = [0, 0.001, 0.0001];
weight_decay = [0, 0.01, 0.001, 0.0001, 0.00001]; % best 0.00001
Nh = [10, 50, 100, 500]; % best 100
max_epochs = 5000;
early_stopping = 10;

num_runs = 5;

for i=1:length(len_window)
    for j=1:length(eta)
        for k=1:length(alpha)
            for l=1:length(weight_decay)
                for m=1:length(Nh)
                    for n=1:length(max_epochs)
                        for o=1:length(early_stopping)
                            %size of input delay
                            inputDelays = 0:len_window(i);
                            % Create the net
                            net = timedelaynet(inputDelays);
                            %net
                            %net = configure(net, tr_in, tr_tg)
                            %net
                            net.numInputs = size(tr_in,3); %set  the number of inputs equal to the number of samples
                            net.divideFcn = 'dividetrain'; %all samples provided in train are used for training
                            net.trainParam.showWindow=0;
                            net.trainParam.lr = eta(j); %learning rate for gradient descent alg
                            net.trainParam.mc = alpha(k); %momentum constant
                            net.performParam.regularization = weight_decay(l); %weight decay regularization
                            net.layers{1}.size = Nh(m); %size of the (first) hidden layer
                            net.trainParam.epochs = max_epochs(n); %maximum number of epochs
                            net
                            %for sample=1:size(tr_in,3)
                                % Prepare the time series
                                [delayedInput, initialInput, initialStates, delayedTarget] = preparets(net,num2cell(tr_in(:,:,sample)),num2cell(tr_tg(:,sample)));
                                % Training
                                net = train(net, delayedInput, delayedTarget, initialInput);
                                tr_out = net(tr_in);
                                vl_out = net(vl_in);
                                tr_error = immse(tr_out, tr_tg);
                                vl_error = immse(vl_out, vl_tg);
                            %end
                        end
                    end
                end
            end
        end
    end
end
