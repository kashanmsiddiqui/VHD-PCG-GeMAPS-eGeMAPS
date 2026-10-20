% Heart Sound Classification Feature Set Comparison
% Refactored script for Physionet SVM results
% Works with filenames like:
%   SVM_Results_shenPhysionet_8020AllFolders20260627_215721.mat
%   SVM_Results_combinedPhysionet_8020AllFolders20260627_215413.mat
% Reports mean difference, Cohen's d and raw p-values
% Also calculates Holm-Bonferroni correction and reports corrected p-values

clc; clear; close all;

% Prompt user to select two MAT files
[file1, path1] = uigetfile('*.mat', 'Select first MAT file');
if isequal(file1,0), disp('No file selected. Exiting.'); return; end

[file2, path2] = uigetfile('*.mat', 'Select second MAT file');
if isequal(file2,0), disp('No file selected. Exiting.'); return; end

% --- Extract identifiers from filenames ---
pattern = 'SVM_Results_(.*?)_2026';
id1 = regexp(file1, pattern, 'tokens', 'once');
id2 = regexp(file2, pattern, 'tokens', 'once');

if isempty(id1) || isempty(id2)
    error('File names do not match expected pattern.');
end

id1 = id1{1};
id2 = id2{1};

% --- Load results struct safely ---
tmp1 = load(fullfile(path1, file1));
tmp2 = load(fullfile(path2, file2));

% Handle different struct names: 'results' or 'SVMresults'
if isfield(tmp1, 'results')
    res1 = tmp1.results;
elseif isfield(tmp1, 'SVMresults')
    res1 = tmp1.SVMresults;
else
    error('No results struct found in %s', file1);
end

if isfield(tmp2, 'results')
    res2 = tmp2.results;
elseif isfield(tmp2, 'SVMresults')
    res2 = tmp2.SVMresults;
else
    error('No results struct found in %s', file2);
end

% --- Build tables from fold metrics ---
metricsList = {'accuracy','balancedAcc','sensitivity','specificity', ...
            'precision','f1','mcc','kappa','auc'};
T1 = struct2table(res1.cv.folds);
T2 = struct2table(res2.cv.folds);

% Keep only selected metrics
T1 = T1(:, metricsList);
T2 = T2(:, metricsList);


T1.Properties.Description = ['Metrics extracted from ', file1];
T2.Properties.Description = ['Metrics extracted from ', file2];

% --- Display tables ---
disp(['--- Metrics from file: ', file1, ' ---']);
disp(T1);
disp(['--- Metrics from file: ', file2, ' ---']);
disp(T2);

% --- Statistical comparison ---
nMetrics = numel(metricsList);
finalResults = cell(nMetrics, 6);
rawP = zeros(nMetrics,1);

for i = 1:nMetrics
    m = metricsList{i};
    x1 = T1.(m);
    x2 = T2.(m);

    % Mean and SD
    mean1 = mean(x1); sd1 = std(x1,1);
    mean2 = mean(x2); sd2 = std(x2,1);

    % Signed mean difference
    meanDiff = mean1 - mean2;

    if numel(x1) ~= numel(x2)
        warning('Metric %s has unequal fold counts (%d vs %d). Truncating to min length.', ...
                 m, numel(x1), numel(x2));
        n = min(numel(x1), numel(x2));
        x1 = x1(1:n);
        x2 = x2(1:n);
    end
    
    diffVals = x1 - x2;
    cohen_d = mean(diffVals) / std(diffVals,1);
    p = signrank(x1, x2);

    % Save raw p-value for Holm correction
    rawP(i) = p;
    
    % Store temporary results
    finalResults{i,1} = m;
    finalResults{i,2} = sprintf('%.4f ± %.4f', mean1, sd1);
    finalResults{i,3} = sprintf('%.4f ± %.4f', mean2, sd2);
    finalResults{i,4} = round(meanDiff,4);
    finalResults{i,5} = round(cohen_d,4);
    finalResults{i,6} = round(p,4);   % raw p-value
    
end

% =====================================================
% Holm-Bonferroni correction
% =====================================================

mTests = length(rawP);
[sortedP, sortIdx] = sort(rawP);
holmP_sorted = zeros(size(sortedP));

for k = 1:mTests
    holmP_sorted(k) = (mTests - k + 1) * sortedP(k);
end

% Cap at 1
holmP_sorted = min(holmP_sorted,1);
% Enforce monotonicity
for k = 2:mTests
    holmP_sorted(k) = max(holmP_sorted(k), ...
                          holmP_sorted(k-1));
end
% Return to original order
holmP = zeros(size(rawP));
holmP(sortIdx) = holmP_sorted;

for i = 1:nMetrics
    finalResults{i,7} = round(holmP(i),4);
end

% --- Convert to summary table ---
SummaryTable = cell2table(finalResults, ...
    'VariableNames', {'Metric', ...
                        [id1 '_Mean±SD'], ...
                        [id2 '_Mean±SD'], ...
                        'MeanDifference', ...
                        'Cohens_d', ...
                        'Raw_pValue', ...
                        'Holm_pValue'});

% --- Display summary ---
disp('--- Statistical Comparison Summary ---');
disp(SummaryTable);

% --- Save outputs ---
outMatFile = sprintf('statistical_comparison_%s_vs_%s.mat', id1, id2);
outExcelFile = sprintf('statistical_comparison_%s_vs_%s.xlsx', id1, id2);

save(outMatFile, 'SummaryTable');
fprintf('Summary saved to %s\n', outMatFile);

writetable(SummaryTable, outExcelFile, 'Sheet', 'Results');
fprintf('Summary exported to %s\n', outExcelFile);
