close all

x = ads_filtered(500:500+2944,3);

[y_zfr1,y_zfr2] = ZFR(normalize(x)); 
y_zfr1_norm = normalize(y_zfr1);

figure('Color','w','Position',[100 100 1300 300]);
hold on
plot(normalize(x),'k','LineWidth',1.5);
plot(y_zfr1_norm,'g','LineWidth',1.5);
xlabel('sample point');
ylabel('Normalized amplitude (a.u.)');
title('Original waveform and ZFR output');
legend('PS waveform','ZFR-1');
ax1 = gca; ax1.FontName = 'Times New Roman'; ax1.FontSize = 12;

x_norm = normalize(x);
Peak_index = [];
vally_index = [];
PZ = [];
NZ = [];
pz = 1;
nz = 1;
y_zfr = y_zfr1;

for i = 2:length(y_zfr)
    if y_zfr(i-1) < 0 && y_zfr(i) >= 0
        pz = i;
        PZ(end+1) = pz;
        [~,pindex] = max(x_norm(nz:pz));
        Peak_index(end+1) = pindex + nz - 1;
    end
    if y_zfr(i-1) > 0 && y_zfr(i) <= 0
        nz = i;
        NZ(end+1) = nz;
        [~,nindex] = min(x_norm(pz:nz));
        vally_index(end+1) = nindex + pz - 1;
    end
end
plot(NZ,y_zfr1_norm(NZ),'ro','LineWidth',3);
plot(PZ,y_zfr1_norm(PZ),'gx','LineWidth',3);
 
legend('PS waveform','ZFR-1','NZ','PZ')
figure('Color','w','Position',[100 100 1300 300]);
hold on
plot(x_norm,'Color',[0,0,0],'LineWidth',1.5);
plot(Peak_index,x_norm(Peak_index),'ro','LineWidth',1.5);
plot(vally_index,x_norm(vally_index),'gx','LineWidth',1.5);
xlabel('sample point');
ylabel('Normalized amplitude (a.u.)');
title('ZFR-based Peak and Valley detection');
legend('signal','peak','valley');
ax1 = gca; ax1.FontName = 'Times New Roman'; ax1.FontSize = 12;

if length(Peak_index) > 1
    peak_intervals = diff(Peak_index);
    peak_stability = std(peak_intervals) / mean(peak_intervals);
else
    peak_intervals = [];
    peak_stability = NaN;
end


if length(vally_index) > 1
    valley_intervals = diff(vally_index);
    valley_stability = std(valley_intervals) / mean(valley_intervals);
else
    valley_intervals = [];
    valley_stability = NaN;
end


fprintf('=== Time Stability Metrics ===\n');
fprintf('Peak interval count: %d\n', length(peak_intervals));
fprintf('Peak time stability (std/mean): %.4f\n', peak_stability);
fprintf('Valley interval count: %d\n', length(valley_intervals));
fprintf('Valley time stability (std/mean): %.4f\n', valley_stability);


figure('Color','w');
hold on
plot(peak_intervals,'-ro','LineWidth',1.5);
plot(valley_intervals,'-gx','LineWidth',1.5);
plot(1:length(peak_intervals),mean(peak_intervals),'r--','LineWidth',1.5);
xlabel('index');
ylabel('interval (samples)');
title('Time Intervals of Peaks and Valleys');
legend('Peak intervals','Valley intervals');
ax1 = gca; ax1.FontName = 'Times New Roman'; ax1.FontSize = 12;
ax1.Box = "on";
