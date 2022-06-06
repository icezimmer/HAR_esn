function [input,target] = dataProcess(input,target)
%DATAPROCESS Summary of this function goes here
%   Detailed explanation goes here
input=num2cell(input,1);
input=reshape(input,[size(input,2) * size(input,3), 1]);

[target, ~] = find(target);
target = categorical(target);
end

