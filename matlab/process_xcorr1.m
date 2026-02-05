close all; clc;


x = ads_filtered(2000:2000+2944,2);
x = normalize(x);

figure('Color',[1 1 1]);
plot(x,'LineWidth',1.5);
xlabel('sample point');
ylabel('Normalized amplitude (a.u.)');
title('Sliced pressure waveform');
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;


x_seg = x;
[acf, lags] = xcorr(x_seg,'coeff');

mid = round(length(lags)/2);

figure('Color',[1 1 1]);
plot(lags(mid:end), acf(mid:end),'k','LineWidth',1.5);
hold on;


N = length(x_seg);
R_env = (N - abs(lags)) / N;
R_env(R_env < 0) = 0;

plot(lags(mid:end), R_env(mid:end),'--r','LineWidth',1.8);


[pks_pos, locs_pos] = findpeaks( ...
    acf, ...
    'MinPeakDistance', 200, ...
    'MinPeakHeight', 0.02);

% 只保留 lag >= 0（避免对称重复）
valid = lags(locs_pos) >= 0;
r_k  = pks_pos(valid);          % 正峰值
locs = locs_pos(valid);
e_k  = R_env(locs);             % 对应理论包络值

% 可视化正峰
plot(lags(locs), acf(locs),'bo','MarkerFaceColor','b');


sigma = 0.15;   % 偏差惩罚参数（0.1~0.2 推荐）

w_k = exp(-(r_k' - e_k).^2 / sigma^2);
F = sum(e_k .* w_k) / sum(e_k);


legend('xcorr (coeff)','Peak envelope','Positive peaks','Location','best');
xlabel('lag (samples)');
ylabel('Autocorrelation coefficient');
title(['Autocorrelation and peak-envelope consistency,  F = ' num2str(F,'%.3f')]);

ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 12;
grid on;


disp(['Envelope consistency score F = ', num2str(F,'%.4f')]);
