function y = readout(x, W_out)
% Discard the washout

[~, len] = size(x);
y = W_out * [x; ones(1, len)];
end

