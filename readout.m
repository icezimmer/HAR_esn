function y = readout(x, W_out)
% Discard the washout

[~, num] = size(x);
y = W_out * [x; ones(1, num)];
end

