
data = ads_filtered;     % 输入数据: [N × 4]
seg_start = 2000;
seg_len   = 2945;
numCh = size(data,2);

sigma = 0.15;   % 峰-包络偏差惩罚
gamma = 16;      % dB → 0–1 映射斜率
alpha = 1.5;      % F 权重
beta  = 0.2;      % 能量权重


F = zeros(numCh,1);
E_ac = zeros(numCh,1);


for ch = 1:numCh

    %% -------- 取信号 --------
    x = data(seg_start:seg_start+seg_len-1, ch);

    %% -------- AC 能量 --------
    x_ac = x - mean(x);
    E_ac(ch) = mean(x_ac.^2);
    x = normalize(x);
    %% -------- xcorr --------
    [acf, lags] = xcorr(x,'coeff');
    N = length(x);
    mid = ceil(length(lags)/2);

    acf_p = acf(mid-1:end);
    lags_p = lags(mid-1:end);

    %% -------- 理论包络 --------
    R_env = (N - abs(lags_p)) / N;
    R_env(R_env < 0) = 0;

    %% -------- 正峰检测 --------
    [pks, locs] = findpeaks( ...
    acf_p, ...
    'MinPeakDistance', 200, ...
    'MinPeakHeight', 0.02);

    if isempty(pks)
        F(ch) = 0;
        continue;
    end

    r_k = pks;              
    e_k = R_env(locs);      

    w_k = exp(-(r_k' - e_k).^2 / sigma^2);
    F(ch) = sum(e_k .* w_k) / sum(e_k);

end


E_ac_dB = 10*log10(E_ac / max(E_ac));


E_norm = exp(E_ac_dB / gamma);


Q = (F.^alpha) .* (E_norm.^beta);


ResultTable = table( ...
    (1:numCh)', ...
    F, ...
    E_ac_dB, ...
    E_norm, ...
    Q, ...
    'VariableNames', ...
    {'Channel','F_envelope','E_ac_dB','E_energy_norm','Q_quality'} );

disp('===== Signal Quality Evaluation =====');
disp(ResultTable);


[Q_sorted, idx] = sort(Q,'descend');

disp('===== Channel Ranking (Best → Worst) =====');
disp(table(idx, Q_sorted, ...
    'VariableNames',{'Channel','Q'}));