function [z1,z2] = zfr_filter(x)
    N = length(x);
    y = zeros(size(x));
    z1 = zeros(size(x));
    z2 = zeros(size(x));
    y1 = zeros(3,1);
    y2 = zeros(3,1);
    if N < 3
        error('Signal too short for ZFR filter');
    end

    %method 1
    if 1
    for n = 3:N
        y(n) = x(n) + 2* y(n-1) - y(n-2);
    end
    for n = 3:N
        z1(n) = y(n) + 2* y(n-1) - y(n-2);
    end
    z1 = z1-movmean(z1,500);
    end
   %method 2
   if 1
    for n = 3:N
        y1(3) = 2* y1(2) - y1(1)+ x(n);
        y2(3) = 2* y2(2) - y2(1) + y1(3);
        z2(n) = -y2(3);
        y1(1) = y1(2);
        y1(2) = y1(3);
        y2(1) = y2(2);
        y2(2) = y2(3);
        y2 = y2*0.97;
        y1 = y1*0.95;
    end
   end 
end