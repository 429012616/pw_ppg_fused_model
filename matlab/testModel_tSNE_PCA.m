close all;

data = load('attn_features.mat');  
%X = double(attn_out);
X = double(data.attn_out);                  % N×4×D
BPtrue = double(data.true);                 % N×2
BPpred = double(data.pred);                 % N×2
MaPpred = (BPpred(:,1)+2*BPpred(:,2))/3;
BPuse = BPtrue(:,1);%MaPpred;
N = size(X,1);

X2D = reshape(X, N, []);  % N×(4*D)

cmap = yellow2red(256);


cmin = min(BPuse);
cmax = max(BPuse);

figure('Color',[1,1,1],'Position',[0,0,400,300]);

try
    rng(5);
    initY = randn(N,2);
    Y_tsne = tsne(X2D,'InitialY', initY, 'Perplexity', 30);
    scatter(Y_tsne(:,1), Y_tsne(:,2), 15, BPuse, 'filled');
    colormap(cmap); colorbar;
    caxis([cmin cmax]);
    title('t-SNE by SBP');
    xlabel('Dim 1'); ylabel('Dim 2');
catch
    warning('t-SNE failed on this MATLAB version.');
end
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;

figure('Color',[1,1,1],'Position',[0,0,400,300]);

[coeff, score, ~, ~, explained] = pca(X2D);
Y_pca = score(:,1:2);
scatter(Y_pca(:,1), Y_pca(:,2), 15, BPuse, 'filled');
colormap(cmap); colorbar;
caxis([cmin cmax]);
% b = regress(BPuse, [Y_pca(:,1), Y_pca(:,2), ones(size(Y_pca,1),1)]);
% hold on;
% dir_vec = [b(1), b(2)];  
% dir_vec = dir_vec / norm(dir_vec);  
% origin = mean(Y_pca(:,1:2));
% t = -3:0.1:3;
% plot(origin(1) + t*dir_vec(1), origin(2) + t*dir_vec(2), 'k--', 'LineWidth', 1.5);

title('PCA by SBP');
xlabel('PC 1'); ylabel('PC 2');
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;

%% =========== 3) USBP ==========
if exist('umap','file') == 2
    try
        figure('Color',[1,1,1]);
        rng(2);
        Y_umap = umap(X2D, 'n_components', 2);
        scatter(Y_umap(:,1), Y_umap(:,2), 15, BPuse, 'filled');
        colormap(cmap); colorbar;
        caxis([cmin cmax]);
        title('USBP Colored by True SBP');
        xlabel('USBP 1'); ylabel('USBP 2');
    catch
        warning('USBP call failed. Check toolbox.');
    end
else
    warning('USBP not available.');
end
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;
%% =========== 4) MDS ==========
figure('Color',[1,1,1]);

Dmat = pdist(X2D, 'euclidean');
Dmat2 = squareform(Dmat);
[Y_mds, ~] = cmdscale(Dmat2);
Y_mds2 = Y_mds(:,1:2);
scatter(Y_mds2(:,1), Y_mds2(:,2), 15, BPuse, 'filled');
colormap(cmap); colorbar;
caxis([cmin cmax]);
title('MDS Colored by True SBP');
xlabel('MDS 1'); ylabel('MDS 2');
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;
%% =========== 5) PCA50 + t-SNE ==========
figure('Color',[1,1,1]);

try
    kPCA = min(50, size(X2D,2));
    coeff50 = coeff(:,1:kPCA);
    X_pca50 = X2D * coeff50;
    rng(2);
    Y_tsne50 = tsne(X_pca50, 'Standardize', true, 'Perplexity', 30);
    scatter(Y_tsne50(:,1), Y_tsne50(:,2), 15, BPuse, 'filled');
    colormap(cmap); colorbar;
    caxis([cmin cmax]);
    title('PCA50 + t-SNE Colored by True SBP');
    xlabel('Dim 1'); ylabel('Dim 2');
catch
    warning('PCA50 + t-SNE failed.');
end
ax1 = gca;
ax1.FontName = 'Times New Roman';
ax1.FontSize = 12;
function cmap = yellow2red(n)


if nargin < 1
    n = 256;
end

cYellow = [1 1 0];
cRed    = [1 0 0];
cBlack  = [0 0 0];

halfN = ceil(n/2);


map1 = [ ...
    linspace(cYellow(1), cRed(1), halfN)' ...
    linspace(cYellow(2), cRed(2), halfN)' ...
    linspace(cYellow(3), cRed(3), halfN)' ...
];

remain = n - halfN;
if remain > 0
    map2 = [ ...
        linspace(cRed(1), cBlack(1), remain)' ...
        linspace(cRed(2), cBlack(2), remain)' ...
        linspace(cRed(3), cBlack(3), remain)' ...
    ];
    cmap = [map1; map2];
else
    cmap = map1(1:n,:);
end
end

kSmooth = 10;  
N = size(Y_tsne,1);

dist2D = pdist2(Y_tsne, Y_tsne, 'euclidean');


[~, idx] = sort(dist2D, 2, 'ascend');
idx_k = idx(:, 2:kSmooth+1);


dispersion = zeros(N,1);
for i = 1:N
    neighbors = idx_k(i,:);
    dispersion(i) = mean(abs(BPuse(i) - BPuse(neighbors)));
end

% Smoothness Score
smoothness_score = 1 - mean(dispersion) / (max(BPuse) - min(BPuse));
fprintf('t-SNE Smoothness Score: %.4f\n', smoothness_score);

figure('Color',[1,1,1],'Position',[100,100,500,400]);
scatter(Y_tsne(:,1), Y_tsne(:,2), 15, dispersion, 'filled');
colormap(jet); colorbar;
caxis([min(dispersion) max(dispersion)]);
title(sprintf('t-SNE MAP Dispersion (Smoothness Score = %.3f)', smoothness_score));
xlabel('Dim 1'); ylabel('Dim 2');
ax2 = gca;
ax2.FontName = 'Times New Roman';
ax2.FontSize = 12;

[r1,p1] = corr(Y_tsne(:,1), BPuse);
[r2,p2] = corr(Y_tsne(:,2), BPuse);
disp(sprintf("r1:%f,p1%f",r1,p1));
disp(sprintf("r2:%f,p2%f",r2,p2));
