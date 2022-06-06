function [dv_index, tr_index, vl_index, ts_index] = holdOut()

vl_index = [1,2, 8,9, 14,15, 29,30, 44,45, 59,60, 74,75]; %14 samples
tr_index = [3:6, 10:12, 16:23, 31:38, 46:54, 61:69, 76:84]; %50 samples
ts_index = [7, 13, 24:28, 39:43, 55:58, 70:73, 85:88];
dv_index = union(tr_index, vl_index); %64 samples

end