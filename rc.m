function [x, x_ws, pooler, W_in, W_hat] = rc(u, omega_in, Nh, rho, seed, x0, Nw)

[Nu, time_steps] = size(u);

if nargin < 6
    x = zeros(Nh,1);
    Nw = 0;
elseif nargin == 6
    x = x0;
    Nw = 0;
else
    x = x0;
end

W_in = initInputMatrix(Nu, omega_in, Nh, seed);
W_hat = initStateMatrix(Nh, rho, seed);

% Add ones for bias
u = [u; ones(1, time_steps)];

for t=1:time_steps
    x = cat(2, x, tanh(W_in * u(:,t) + W_hat * x(:,end)));
end

% Discard the initial state
x = x(:, 2:end);

% Discard the washout
x_ws = x(:, Nw+1:end);

pooler = x(:, end);

end

