close all
%ZFR2
% b = [0, -0.955*0.97, 0];
a1 = [1, -2*0.95, 0.95^2];
a2 = [1, -2*0.97, 0.97^2];
% a  = conv(a1, a2);
%ZFR1
b = [1];
a = conv(a1, a2);


figure('Color','w');

[H, w] = freqz(b, a, 1024);

subplot(2,1,1);
plot(w/pi, abs(H),'LineWidth',1.5);
xlabel('Normalized Frequency × \pi rad/sample');
ylabel('Magnitude');
title('Frequency Response |H(e^{j\omega})|');
grid on;
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;

subplot(2,1,2);
plot(w/pi, 20*log10(abs(H)+eps),'LineWidth',1.5);
xlabel('Normalized Frequency × \pi rad/sample');
ylabel('Magnitude (dB)');
title('Frequency Response (dB)');
grid on;
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;
figure('Color','w');
zplane(b, a);   
title('Pole-Zero Plot of the ZFR-1');
xlabel('Real part');
ylabel('Imaginary part');
grid on;
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;