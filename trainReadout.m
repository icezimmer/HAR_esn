function W_out = trainReadout(x, d, lambda_r)
[dim, len] = size(x);

% Discard the washout from target
d = d(:, end-len+1:end);

X = [x; ones(1, len)];

if lambda_r == 0
    W_out = d * pinv(X);
elseif lambda_r > 0
    W_out = d * X' * inv(X*X' + lambda_r * eye(dim+1));
end

end