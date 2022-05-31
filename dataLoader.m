function data = dataLoader()

data = zeros(6, 480, 0);
opts = detectImportOptions(fullfile('AReM','bending1','dataset1.csv'));
var_names = {'avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'};
opts.SelectedVariableNames = var_names;

for num = 1:7
    T = readtable(fullfile('AReM','bending1', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    data = cat(3,data,ts);
end

for num = 1:6
    T = readtable(fullfile('AReM','bending2', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    data = cat(3,data,ts);
end

for num = 1:15
    T = readtable(fullfile('AReM','cycling', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    data = cat(3,data,ts);
end

for num = 1:15
    T = readtable(fullfile('AReM','lying', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    data = cat(3,data,ts);
end

for num = 1:15
    T = readtable(fullfile('AReM','sitting', strcat('dataset', num2str(num), '.csv')),opts);
    if num == 8 % time step 13500 doesn't exist (line 60 / row 55)
        % insert the mean of the two adiacent rows
        T = [T(1:54,:);array2table((T{54,:}+T{55,:})/2, "VariableNames",var_names);T(55:end,:)];
    end
    ts = T{:,:}';
    data = cat(3,data,ts);
end

for num = 1:15
    T = readtable(fullfile('AReM','standing', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    data = cat(3,data,ts);
end

for num = 1:15
    T = readtable(fullfile('AReM','walking', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    data = cat(3,data,ts);
end

end
