function W_hat = initStateMatrix(Nh, rho, seed)
rng(seed)

W_hat = 2*rand(Nh,Nh) - 1;
W_hat = rho * (W_hat / max(abs(eig(W_hat))));
end

