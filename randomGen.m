function [omega_in, Nh, rho, Nw, lambda_r] = randomGen(is, hs, r, ws, lr, seed)

rng(seed)

% Random generation of the hyperparameters
omega_in = is*rand; % input scaling
Nh = randi([hs(1), hs(2)]); % number of hidden neurons
rho = r*rand; % spatial radius
Nw = randi([ws(1), ws(2)]);% transient
lambda_r = 10^(-randi([lr(1), lr(2)])); % weight decay regularization

end

