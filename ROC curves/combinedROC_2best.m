% Load first mat file
data1 = load('SVM_Results_combinedPresetPhysionet_20261012_182637.mat');   % replace with actual filename
tbl1 = data1.SVMresults.all_yTestPred;
labels1 = tbl1.TrueLabel;
scores1 = tbl1.Score;

% Load second mat file
data2 = load('SVM_Results_combined_preset_Yaseen_20261012_201622.mat');   % replace with actual filename
tbl2 = data2.SVMresults.all_yTestPred;
labels2 = tbl2.TrueLabel;
scores2 = tbl2.Score;

% Compute ROC curves
[fpr1, tpr1, ~, auc1] = perfcurve(labels1, scores1, 1); 
[fpr2, tpr2, ~, auc2] = perfcurve(labels2, scores2, 1); 

% Plot both ROC curves on the same graph
figure('Position', [100, 100, 600, 500]);
plot(fpr1, tpr1, 'b-', 'LineWidth', 2); hold on;
plot(fpr2, tpr2, 'r-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Comparison of ROC Curves 10-fold CV');
legend(sprintf('Physionet 2016 (AUC = %.4f)', auc1), ...
       sprintf('Yaseen (AUC = %.4f)', auc2), ...
       'Random','Location','southeast');
grid on; axis square;
