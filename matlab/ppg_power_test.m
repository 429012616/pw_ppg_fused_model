clear; clc; close all;
load("ppg.mat")
fs = 500;

signal = inf1;

cutpoint = [3390,6710,10120,13693,17014,20822,24649,29225,34444];
offset   = 50;


segmentStarts = [1, cutpoint + offset + 1];
segmentEnds   = [cutpoint - offset, length(signal)];
numSeg = length(segmentStarts);


dcEnergy  = zeros(1,numSeg);
acEnergy  = zeros(1,numSeg);
acdcRatio = zeros(1,numSeg);
acdcDB    = zeros(1,numSeg);


segs_ori = cell(1,numSeg);
segs_low = cell(1,numSeg);
segs_high= cell(1,numSeg);
segs_high_ori = cell(1,numSeg);

for k = 1:numSeg
    idx1 = segmentStarts(k);
    idx2 = segmentEnds(k);
    
    if idx1 > idx2 || idx2 > length(signal)
        warning('Segment %d invalid after trimming — skip', k);
        dcEnergy(k) = NaN;
        acEnergy(k) = NaN;
        acdcRatio(k) = NaN;
        acdcDB(k) = NaN;
        continue;
    end
    

    seg = signal(idx1:idx2);
    segs_ori{k} = seg;
    

    winLength = 301; 
    if winLength > length(seg)
        winLength = length(seg)-1; % 保证窗口小于段长
    end
    

    lowTrend = movmean(seg, winLength, 'Endpoints', 'shrink');
    

    highRes = seg - lowTrend;
    

    segs_low{k}  = lowTrend;
    segs_high{k} = highRes;
    

    dcVal  = mean(lowTrend);
    dcVec  = dcVal * ones(size(lowTrend));
    

    acPart = highRes;
    
    % 能量
    dcEnergy(k) = sum(dcVec.^2)/(length(dcVec));
    acEnergy(k) = sum(acPart.^2)/(length(acPart));
    
    % 比值
    if dcEnergy(k) > 0
        acdcRatio(k) = acEnergy(k)/dcEnergy(k);
        acdcDB(k)    = 10 * log10(acEnergy(k)/dcEnergy(k));
    else
        acdcRatio(k) = NaN;
        acdcDB(k)    = NaN;
    end
end


disp('Seg  DC_Energy  AC_Energy  AC/DC  AC/DC (dB)');
for k=1:numSeg
    fprintf('%3d  %10.4g   %10.4g   %9.4g   %9.4g\n', ...
        k, dcEnergy(k), acEnergy(k), acdcRatio(k), acdcDB(k));
end


figure;
ax = gca;            
ax.FontName = 'Times New Roman';  
ax.FontSize = 14;     
set(gcf, 'Color', [1 1 1]);
hold on;
subplot(3,1,1);
bar((1:10)*10,dcEnergy,'FaceColor', [4/255 114/255 190/255]); title('DC energe per segment'); ylabel('Energy');

subplot(3,1,2);
bar((1:10)*10,acEnergy,'FaceColor', [4/255 114/255 190/255]); title('AC energe per segment'); ylabel('Energy');

subplot(3,1,3);
bar((1:10)*10,acdcDB,'FaceColor', [4/255 114/255 190/255]); title('AC/DC ratio (dB) per segment'); ylabel('dB');xlabel('LED Power (%)');

figure;
ax = gca;            
ax.FontName = 'Times New Roman';  
ax.FontSize = 14;     
set(gcf, 'Color', [1 1 1]);
hold on;
for k=1:numSeg
    plot(segs_ori{k}); 
end
title('Each segment original'); xlabel('Sample in segment');

figure;
ax = gca;             
ax.FontName = 'Times New Roman';  
ax.FontSize = 14;     
set(gcf, 'Color', [1 1 1]);
hold on;
for k=1:numSeg
    plot(segs_low{k});
end
title('Each segment low frequency (moving average)');

figure;
hold on;
ax = gca;             
ax.FontName = 'Times New Roman';  
ax.FontSize = 14;     
set(gcf, 'Color', [1 1 1]);
for k=1:numSeg
    plot(segs_high{k});
end
title('Each segment high frequency (residual)');

fc = 30;                     
Wn = fc/(fs/2);               
firOrder = 100;               
b_low30 = fir1(firOrder, Wn, 'low', hamming(firOrder+1));

snr_vals = zeros(1,numSeg);  
snr_vals_lin = zeros(1,numSeg);
for k = 1:numSeg
    segHigh = segs_high{k};
    segs_high_ori{k} = segs_high{k}
  
    acSignal = filtfilt(b_low30, 1, segHigh);
    
    segs_high{k} = acSignal;
    

    segLen = length(acSignal);
    acEnergy = sum(acSignal.^2)/segLen;
    

    noiseSignal = segHigh - acSignal;
    noiseEnergy = sum(noiseSignal.^2)/segLen;
    
    if noiseEnergy > 0
        snr_vals_lin(k) = acEnergy / noiseEnergy;
        snr_vals(k)     = 10*log10(snr_vals_lin(k));
    else
        snr_vals_lin(k) = NaN;
        snr_vals(k) = NaN;
    end
    

    subplot(numSeg,1,k);
    plot(acSignal);
    title(['LED Power (%) ', num2str(k), ' AC signal, SNR(dB) = ', num2str(snr_vals(k), '%.2f')]);
    xlabel('Sample'); ylabel('Amplitude');
end

subplot(numSeg,1,k);
plot(acSignal);
title(['LED Power (%) ', num2str(k), ' AC signal, SNR(dB) = ', num2str(snr_vals(k), '%.2f')]);
xlabel('Sample'); ylabel('Amplitude');


figure;
ax = gca;             
ax.FontName = 'Times New Roman';  
ax.FontSize = 14;    
set(gcf, 'Color', [1 1 1]);
bar((1:10)*10,snr_vals,'FaceColor', [4/255 114/255 190/255]);
title('940nm SNR (dB)');
xlabel('LED Power (%)'); ylabel('SNR_{ac} (dB)');


figure;
hold on;
ax = gca;             
ax.FontName = 'Times New Roman';  
ax.FontSize = 14;     
set(gcf, 'Color', [1 1 1]);
plot((1:1:length(inf1))/500,red1,'Color', [217/255 83/255 25/255],'LineWidth', 2);
plot((1:1:length(red1))/500,inf1,'Color', [4/255 114/255 190/255],'LineWidth', 2);

title('PPG Waveform (10% to 100% power)');
xlabel('Time (s)'); ylabel('Amplitude(a.u.)');
legend('940 nm','660nm')
% figure;
% plot(10*log10(abs(fft(segHighFilt))))


for k = 1:numSeg
figure;
plot(segs_high_ori{k},'Color', [217/255 83/255 25/255],'LineWidth', 2);
title(['LED Power (%) ', num2str(k), ' AC signal, SNR(dB) = ', num2str(snr_vals(k), '%.2f')]);
xlabel('Sample'); ylabel('Amplitude');
end