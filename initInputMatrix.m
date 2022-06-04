function W_in = initInputMatrix(Nu, omega_in, Nh, seed, a)
rng(seed)

if a > 0
    W_in = 2*rand(Nh,Nu+1) - 1;
    W_in = omega_in * W_in;
else
    W_in = zeros(Nh,Nu+1);
end

end

