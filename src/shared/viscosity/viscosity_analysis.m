% viscosity_analysis.m — Enhanced visualization for transformer-based antibody viscosity model

load('viscosity_results.mat');


%% 1. True vs. Predicted Viscosity Parity Plot
% Scatter plot on log-log axes with R² and RMSE annotations, outlier highlighted, and identity line labeled.
figure;
scatter(y_test, y_pred, 60, 'MarkerFaceColor', [0.863, 0.078, 0.235], ...  % crimson markers
        'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.5);
hold on;
% Determine range for axes (include outliers)
minVal = min([y_test(:); y_pred(:)]);
maxVal = max([y_test(:); y_pred(:)]);
% Set log scale and axis limits
set(gca, 'XScale', 'log', 'YScale', 'log');
rangeVal = maxVal - minVal;
minLim = max(0, minVal - 0.05*rangeVal);     % do not go below 0
maxLim = maxVal + 0.05*rangeVal;
plot([minLim, maxLim], [minLim, maxLim], 'k--', 'LineWidth', 1.5);
xlim([minLim, maxLim]); ylim([minLim, maxLim]);
% Plot the diagonal identity line (Perfect Prediction reference)
%plot([minVal, maxVal], [minVal, maxVal], 'k--', 'LineWidth', 1.5);
% Compute R² and RMSE for annotations
R_squared = (corr(y_test, y_pred)).^2;
RMSE_val = sqrt(mean((y_test - y_pred).^2));
annotation_text = sprintf('R^2 = %.2f\nRMSE = %.2f cP', R_squared, RMSE_val);
% Annotate R² and RMSE in top-left area of plot
text(0.7*10^(log10(minVal) + 0.1*(log10(maxVal/minVal))), ...       % 10% from left (log scale)
     10^(log10(minVal) + 0.9*(log10(maxVal/minVal))), ...       % 90% from bottom (log scale)
     annotation_text, 'FontSize', 14, 'Color', 'k');
% Highlight the outlier point with a red star and label it
%[~, outlier_idx] = max(abs(y_true - y_pred));                   % index of largest residual (outlier)
%scatter(y_true(outlier_idx), y_pred(outlier_idx), 100, 'p', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'k');
%text(y_true(outlier_idx) * 0.15, y_pred(outlier_idx) * 1.2, 'Outlier', ... 
%     'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
% Annotate the identity line
%midVal = sqrt(minVal * maxVal);
%text(midVal, midVal * 1.1, 'Perfect Prediction', 'FontSize', 14, 'FontAngle', 'italic', ...
%     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
hold off;
% Formatting: make axes square, add grid and labels
axis equal; axis square;
grid on;
xlabel('True Viscosity (cP, log scale)', 'FontSize', 16);
ylabel('Predicted Viscosity (cP, log scale)', 'FontSize', 16);
%title('True vs. Predicted Viscosity', 'FontSize', 18);
set(gca, 'FontSize', 14);  % tick label font size
% Save figure at high resolution
print(gcf, 'ParityPlot.png', '-dpng', '-r300');

%% ————— Panel 2B: CV fold metrics & Test‐set metrics for R², RMSE, MAE —————
%% Panel 2: Model Performance Metrics (Transformer Highlighted)

data = load('viscosity_results_cv.mat');

% CV metrics explicitly loaded
cv_r2   = [data.cv_r2_transformer(:), data.cv_r2_ridge(:), data.cv_r2_svr(:), data.cv_r2_rf(:)];
cv_rmse = [data.cv_rmse_transformer(:), data.cv_rmse_ridge(:), data.cv_rmse_svr(:), data.cv_rmse_rf(:)];
cv_mae  = [data.cv_mae_transformer(:), data.cv_mae_ridge(:), data.cv_mae_svr(:), data.cv_mae_rf(:)];

% Test-set metrics
test_r2   = data.test_r2_all(:)';
test_rmse = data.test_rmse_all(:)';
test_mae  = data.test_mae_all(:)';

models = {'Transformer','Ridge','SVR','RandomForest'};

% Bar colors: first Transformer highlighted, rest pastel shades
barColors = [
    0.2 0.6 0.8;    % Transformer (stronger blue)
    0.75 0.85 0.95; % Ridge (soft pastel blue)
    0.75 0.85 0.95; % SVR
    0.75 0.85 0.95  % RF
];

diamondColor = [0.3 0.3 0.3]; % CV mean diamonds soft grey
circleEdgeColor = [0.2 0.2 0.2]; % CV folds circles subtle black edge

figure('Units','normalized','Position',[0.05 0.1 0.9 0.7]);

%% Panel B: R² (larger panel)
subplot(2,2,[1,2]); hold on;
for i = 1:4
    bar(i, test_r2(i), 'FaceColor',barColors(i,:),'BarWidth',0.5,'EdgeColor','none');
end
scatter(1:4,mean(cv_r2),100,'d','MarkerEdgeColor',diamondColor,'MarkerFaceColor',diamondColor);
for i = 1:4
    scatter(repmat(i,1,5)+(rand(1,5)-0.5)*0.15, cv_r2(:,i), 40,'o',...
        'MarkerEdgeColor',circleEdgeColor,'MarkerFaceColor','none','LineWidth',1);
end
ylim([0 1]); ylabel('R^2','FontSize',16); xticks(1:4); xticklabels(models);
title('B: Model Performance (R^2)','FontSize',18,'FontWeight','bold');
set(gca,'FontSize',14,'FontName','Helvetica','LineWidth',1.2,'TickDir','out','Box','off');
%legend({'Test-set','CV mean','CV folds'},'Location','best','FontSize',12,'Box','off');
% grab the three graphic handles in plotting order
hBars = findobj(gca,'Type','Bar');
hMean = findobj(gca,'Type','Scatter','Marker','d');
hFold = findobj(gca,'Type','Scatter','Marker','o');

% bars come back in reverse order, so pick the last one for “Test-set”
legend([hBars(end), hMean(1), hFold(end)], ...
       {'Test set','CV mean','CV folds'}, ...
       'Location','best','FontSize',12,'Box','off');
%% Panel C: RMSE (smaller)
subplot(2,2,3); hold on;
for i = 1:4
    bar(i, test_rmse(i), 'FaceColor',barColors(i,:),'BarWidth',0.5,'EdgeColor','none');
end
scatter(1:4,mean(cv_rmse),100,'d','MarkerEdgeColor',diamondColor,'MarkerFaceColor',diamondColor);
for i = 1:4
    scatter(repmat(i,1,5)+(rand(1,5)-0.5)*0.15, cv_rmse(:,i), 40,'o',...
        'MarkerEdgeColor',circleEdgeColor,'MarkerFaceColor','none','LineWidth',1);
end
ylabel('RMSE','FontSize',16); xticks(1:4); xticklabels(models); xtickangle(45);
title('C: RMSE','FontSize',18,'FontWeight','bold');
set(gca,'FontSize',14,'FontName','Helvetica','LineWidth',1.2,'TickDir','out','Box','off');
%legend({'Test-set','CV mean','CV folds'},'Location','best','FontSize',10,'Box','off');

%% Panel D: MAE (smaller)
subplot(2,2,4); hold on;
for i = 1:4
    bar(i, test_mae(i), 'FaceColor',barColors(i,:),'BarWidth',0.5,'EdgeColor','none');
end
scatter(1:4,mean(cv_mae),100,'d','MarkerEdgeColor',diamondColor,'MarkerFaceColor',diamondColor);
for i = 1:4
    scatter(repmat(i,1,5)+(rand(1,5)-0.5)*0.15, cv_mae(:,i), 40,'o',...
        'MarkerEdgeColor',circleEdgeColor,'MarkerFaceColor','none','LineWidth',1);
end
ylabel('MAE','FontSize',16); xticks(1:4); xticklabels(models); xtickangle(45);
title('D: MAE','FontSize',18,'FontWeight','bold');
set(gca,'FontSize',14,'FontName','Helvetica','LineWidth',1.2,'TickDir','out','Box','off');
%legend({'Test-set','CV mean','CV folds'},'Location','best','FontSize',10,'Box','off');

% Overall title
sgtitle('Model Comparison Metrics (Test-set vs CV)', 'FontSize', 20, 'FontWeight', 'bold');

% Save figures clearly and professionally
print(gcf,'Fig2B_ModelMetrics_TransformerHighlighted.png','-dpng','-r300');
print(gcf,'Fig2B_ModelMetrics_TransformerHighlighted.svg','-dsvg');


%% Vertical Feature Importance – Separate Brackets per Comparison

% 1. Data
meanVals    = [0.6979, 0.2891, 0.1220, 0.0501];
stdVals     = [0.1555, 0.1045, 0.0523, 0.0438];
labels      = {'DLS kD','SE-UHPLC Plates','AC-SINS $\lambda_{\max}$','SE-UHPLC FWHM'};
rawBySample = [
    0.9042, 0.3103, 0.1088, 0.0232;
    0.7073, 0.1328, 0.1303, 0.0502;
    0.5423, 0.2819, 0.1482, 0.0072;
    0.5063, 0.2615, 0.0322, 0.0371;
    0.8297, 0.4590, 0.1903, 0.1329
];
pvals       = [0.0055, 0.0011, 0.0008, 0.0232, 0.0049, 0.0383];
pairs       = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];

% 2. Sort by descending |mean|
[~, I] = sort(abs(meanVals),'descend');
y      = meanVals(I);
yerr   = stdVals(I);
cats   = labels(I);
raw    = rawBySample';   % 4×5
raw    = raw(I,:);

% 3. Plot setup
figure('Color','w','Position',[100 100 700 500]);
hold on;

% 4. Bars & colors
cmap = parula(numel(y));
[~,cIdx] = sort(y,'ascend');
hb = bar(1:numel(y), y, ...
    'FaceColor','flat','EdgeColor','none','BarWidth',0.6);
for i = 1:numel(y)
    hb.CData(i,:) = cmap(cIdx(i),:);
end

% 5. Error bars
he = errorbar(1:numel(y), y, yerr, 'k','LineStyle','none','LineWidth',1.5);
he.CapSize = 12;

% 6. Raw sample scatter with jitter
jitter = 0.08;
for i = 1:size(raw,1)
    xj = i + (rand(1,5)-0.5)*jitter;
    scatter(xj, raw(i,:), 36, ...
        'MarkerEdgeColor','k','MarkerFaceColor','none','LineWidth',0.8);
end

% 7. Axes styling
set(gca, ...
    'XTick',1:numel(y), ...
    'XTickLabel',cats, ...
    'TickLabelInterpreter','latex', ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',1, ...
    'FontSize',12);
ylabel('Importance Score','FontSize',14);
title('Feature Importance with Uncertainty & Samples','FontSize',16,'FontWeight','normal');

% 8. Compute separate bracket heights
offset = 0.04 * max(y + yerr);
% group comparisons by first feature index
comps = cell(1,numel(y));
for k = 1:size(pairs,1)
    f = pairs(k,1);
    comps{f}(end+1) = k;
end
bracketY = nan(size(pairs,1),1);
for f = 1:numel(comps)
    for j = 1:numel(comps{f})
        idx = comps{f}(j);
        % baseline = top of error bar of feature f
        base = y(f) + yerr(f);
        % height = baseline + j * offset
        bracketY(idx) = base + j*offset;
    end
end

% 9. Draw brackets & p-text
for k = 1:size(pairs,1)
    i = pairs(k,1);
    j = pairs(k,2);
    yb = bracketY(k);
    % bracket line
    plot([i, i, j, j], ...
         [yb-0.01*max(y+yerr), yb, yb, yb-0.01*max(y+yerr)], ...
         'k','LineWidth',1);
    % text
    txt = sprintf('\\it p\\,=\\,%.3f', pvals(k));
    text((i+j)/2, yb + 0.01*max(y+yerr), txt, ...
         'Interpreter','latex', ...
         'HorizontalAlignment','center', ...
         'FontSize',12);
end

hold off;

% 10. Save
print(gcf,'FeatureImportance_SeparateBrackets.png','-dpng','-r300');


%% 7. Stratified Model Performance by Viscosity Range
% Evaluate model performance in low, mid, high viscosity bins with error bars.
figure;
% Define viscosity strata (low, mid, high ranges)
thr_low = 15;   % threshold for low vs mid (cP)
thr_high = 30;  % threshold for mid vs high (cP)
idx_low = find(y_test <= thr_low);
idx_mid = find(y_test > thr_low & y_test <= thr_high);
idx_high = find(y_test > thr_high);
% Calculate mean true and predicted viscosity for each bin
true_means = [mean(y_test(idx_low)); mean(y_test(idx_mid)); mean(y_test(idx_high))];
pred_means = [mean(y_pred(idx_low)); mean(y_pred(idx_mid)); mean(y_pred(idx_high))];
% Calculate RMSE within each bin for error bars (model error in each range)
rmse_low = sqrt(mean((y_pred(idx_low) - y_test(idx_low)).^2));
rmse_mid = sqrt(mean((y_pred(idx_mid) - y_test(idx_mid)).^2));
rmse_high = sqrt(mean((y_pred(idx_high) - y_test(idx_high)).^2));
rmse_bins = [rmse_low; rmse_mid; rmse_high];
% Plot mean predicted vs mean true with error bars (vertical error = RMSE)
hold on;
% Identity line (perfect predictions)
plot([0, 50], [0, 50], 'k--', 'LineWidth', 1.5);
errorbar(true_means, pred_means, rmse_bins, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
         'MarkerFaceColor', [0, 0.447, 0.741], 'LineWidth', 1.5, 'CapSize', 10, 'LineStyle', 'none');
hold off;
% Annotate each point with category label
labels = {'Low', 'Mid', 'High'};
for i = 1:3
    if ~isempty(labels{i}) && ~isnan(true_means(i))
        if i < 3
            % Low and Mid: label above the point
            text(true_means(i) + 5, pred_means(i) + 5, labels{i}, 'FontSize', 14, ...
                 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
        else
            % High: label below the point (to avoid error bar top)
            text(true_means(i) + 5, pred_means(i) - 5, labels{i}, 'FontSize', 14, ...
                 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
        end
    end
end
grid on;
axis([0 50 0 50]); axis square;
xlabel('True Viscosity (cP)', 'FontSize', 16);
ylabel('Predicted Viscosity (cP)', 'FontSize', 16);
title('Model Performance by Viscosity Range', 'FontSize', 18);
set(gca, 'FontSize', 14);
% Save figure at high resolution
print(gcf, 'StratifiedPerformance.png', '-dpng', '-r300');

%% 7. Stratified Model Performance by Viscosity Range 
%% Improved Stratified Model Performance Visualization 

% Load bootstrap RMSE samples
bootstrap_samples = load('viscosity_bootstrap_metrics.mat');
labels = {'Low', 'Mid', 'High'};
mat_labels = {'rmse_low', 'rmse_mid', 'rmse_high'};

% Define viscosity bins
thr_low = 15; thr_high = 30;
bins = {
    y_test <= thr_low,
    y_test > thr_low & y_test <= thr_high,
    y_test > thr_high
};

% Calculate mean true and predicted viscosities
true_means = cellfun(@(idx) mean(y_test(idx)), bins);
pred_means = cellfun(@(idx) mean(y_pred(idx)), bins);

% RMSE from bootstrap samples
rmse_mean = zeros(3,1);
for i = 1:3
    rmse_mean(i) = mean(bootstrap_samples.all_rmse_samples.(mat_labels{i}));
end

% Initialize figure
figure; hold on; grid on;

% Identity line (perfect prediction)
plot([0, 50], [0, 50], 'k--', 'LineWidth', 1.5);

% Plot mean points with error bars (RMSE)
errorbar(true_means, pred_means, rmse_mean, 'o', 'MarkerSize', 12,...
    'MarkerFaceColor', [0.2 0.7 0.9], 'MarkerEdgeColor', 'k',...
    'LineWidth', 2, 'Color', [0 0.45 0.74], 'CapSize', 15, 'LineStyle', 'none');

% Plot bootstrap RMSE samples vertically around mean predicted points
rng(0); % reproducibility
for i = 1:3
    samples = bootstrap_samples.all_rmse_samples.(mat_labels{i});
    jitter = (rand(size(samples))-0.5)*0.1; % reduced horizontal jitter
    scatter(true_means(i)+jitter, pred_means(i)+samples, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
    scatter(true_means(i)+jitter, pred_means(i)-samples, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
end

% Annotations clearly positioned to the right of each point
offset = 5; % slightly increased from 4
for i = 1:3
    text(true_means(i) + offset, pred_means(i), labels{i}, 'FontSize', 14, 'FontWeight', 'bold',...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end

% Axes formatting for professional aesthetics
axis square;
xlim([0 50]); ylim([0 50]);
set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'TickDir','out', 'Box','off');
xlabel('True Viscosity (cP)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Predicted Viscosity (cP)', 'FontSize', 16, 'FontWeight', 'bold');
title('Stratified Model Performance by Viscosity Range',...
    'FontSize', 18, 'FontWeight', 'bold');

% Save high-quality figure
print(gcf, 'Improved_StratifiedPerformance.png', '-dpng', '-r300');

%% 8A. Robustness analysis 
%% Robustness Analysis – Learning Curve + Horizontal Bars with Simplified Stars

%--- Subplot 1: Learning Curve ---
training_sizes = [21, 28, 35, 42, 49, 56, 63, 70];
r2_scores      = [0.22, 0.48, 0.61, 0.66, 0.65, 0.66, 0.68, 0.73];

figure('Position',[100,100,1200,500]);
subplot(1,2,1);
plot(training_sizes, r2_scores, '-o', ...
     'LineWidth',2, 'MarkerSize',8, 'Color',[0,0.447,0.741]);
xlabel('Training Set Size (% of Total Dataset)', 'FontSize',12, 'FontWeight','bold');
ylabel('Test R^2',                        'FontSize',12, 'FontWeight','bold');
title('Learning Curve',                 'FontSize',14, 'FontWeight','bold');
grid on; axis square;
ylim([min(r2_scores)-0.05, max(r2_scores)+0.05]);

%--- Subplot 2: Feature Removal Impact ---
% raw five-fold R² (5 folds × 4 scenarios)
r2_raw = [
    0.7176, 0.4644, 0.5717, 0.1875;   % Full, Top2, HT assays, No kD
    0.7225, 0.3275, 0.4733, 0.1591;
    0.7283, 0.5313, 0.5245, 0.1269;
    0.7146, 0.5030, 0.4861, 0.1956;
    0.7082, 0.4502, 0.4371, 0.1714
];
labels = {'Full','Top2 (kD,Plates)','HT assays','No kD'};
means  = mean(r2_raw);
stds   = std(r2_raw);
nSc    = numel(means);

% compute all pairwise p-values
pairs = [1 2; 1 3; 1 4];
p_full = nan(1,3);
for k = 1:3
    [~,p_full(k)] = ttest(r2_raw(:,1), r2_raw(:,k+1));
end

subplot(1,2,2);
hold on;

% 1) horizontal bars
ypos = 1:nSc;
colors = lines(nSc);
hBar = barh(ypos, means, 0.6, 'FaceColor','flat','EdgeColor','none');
for i = 1:nSc
    hBar.CData(i,:) = colors(i,:);
end

% 2) errorbars
hErr = errorbar(means, ypos, stds, stds, 'horizontal', ...
    'LineStyle','none','Color','k','LineWidth',1.5);
hErr.CapSize = 12;

% 3) individual‐fold dots
jitter = 0.15;
for i = 1:nSc
    yj = ypos(i) + (rand(size(r2_raw,1),1)-0.5)*jitter;
    scatter(r2_raw(:,i), yj, 36, ...
        'MarkerEdgeColor','k','MarkerFaceColor','none','LineWidth',0.8);
end

% 4) axes formatting
set(gca, ...
    'YTick',ypos, ...
    'YTickLabel',labels, ...
    'YDir','reverse', ...    % keep "Full" at top
    'TickDir','out', ...
    'Box','off', ...
    'FontSize',12);
xlabel('Test R^2','FontSize',12,'FontWeight','bold');
title('Feature Removal Impact','FontSize',14,'FontWeight','bold');
xlim([0, max(means+stds)*1.2]);

% 5) place stars above each ablated bar (positions 2–4)
for k = 1:3
    idx = k+1;
    p   = p_full(k);
    if      p < 0.001, star = '***';
    elseif  p < 0.01,  star = '**';
    elseif  p < 0.05,  star = '*';
    else    star = 'ns';
    end
    x_star = means(idx);
    y_star = ypos(idx) - 0.3;   % small upward shift in data coords
    text(x_star, y_star, star, ...
         'HorizontalAlignment','center', ...
         'FontSize',14, 'FontWeight','bold');
end

hold off;

% 6) Super-title & save
sgtitle('Robustness Analysis of Model Performance','FontSize',16,'FontWeight','bold');
print(gcf, 'combined_robustness_analysis.png', '-dpng', '-r300');

