function [x, x_ws, pooler, W_in, W_hat] = rc(u, seed, omega_in, rho, Nh, dns, a, x0, Nw)

[Nu, time_steps] = size(u);

if nargin < 6 % no dns, a, x0, Nw
    dns = 1;
    a = 1;
    x = zeros(Nh,1);
    Nw = 0;
elseif nargin == 6 % no a, x0, Nw
    a = 1;
    x = zeros(Nh,1);
    Nw = 0;
elseif nargin == 7 % no x0, Nw
    x = zeros(Nh,1);
    Nw = 0;
elseif nargin == 8 % no Nw
    x = x0;
    Nw = 0;
else
    x = x0;
end

if a < 0 || a > 1
    error('The parameter a must be in [0, 1]')
else
    W_in = initInputMatrix(Nu, omega_in, Nh, seed, a);
    W_hat = initStateMatrix(Nh, rho, seed, dns, a);
    
    % Add ones for bias
    u = [u; ones(1, time_steps)];
    
    % LI-ESN
    for t=1:time_steps
        x = cat(2, x, (1-a)*x(:,end) + a*tanh(W_in*u(:,t) + W_hat*x(:,end)));
    end
    
    % Discard the initial state
    x = x(:, 2:end);
    
    % Discard the washout
    x_ws = x(:, Nw+1:end);
    
    pooler = x(:, end);
end

end

