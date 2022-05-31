function W_out = trainOffline(x, d, lambda_r)

if nargin < 3
    lambda_r = 0;
end

[Nh, num] = size(x);

X = [x; ones(1, num)]; % Nh+1 x num

%sX=size(X)
%sd=size(d)

if lambda_r == 0
    W_out = d * pinv(X); % (7 x num) * (Nh+1 x num)
elseif lambda_r > 0
    W_out = d * X' * inv(X*X' + lambda_r * eye(Nh+1));
end

end

