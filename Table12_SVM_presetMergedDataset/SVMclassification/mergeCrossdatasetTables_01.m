function [trainData_C, testData_C] = mergeCrossdatasetTables_01(trainData_A, testData_A, trainData_B, testData_B)
% mergeAndNormalizeTables
% Merges two datasets (A and B), removes non-numeric and low-variance features,
% normalizes globally, and returns combined tables.
%
% Inputs:
%   trainData_A : Training table A (filename | features | label)
%   testData_A  : Test table A (filename | features | label)
%   trainData_B : Training table B (filename | features | label)
%   testData_B  : Test table B (filename | features | label)
%   varianceThreshold : Minimum variance cutoff for feature removal
%
% Outputs:
%   trainData_C : Combined normalized training table
%   testData_C  : Combined normalized test table

    % --- Step 1: Extract metadata and features ---
    filenames_train = [trainData_A.filename; trainData_B.filename];
    labels_train    = [trainData_A.label;    trainData_B.label];
    features_train  = [trainData_A(:,2:end-1); trainData_B(:,2:end-1)];

    filenames_test = [testData_A.filename; testData_B.filename];
    labels_test    = [testData_A.label;    testData_B.label];
    features_test  = [testData_A(:,2:end-1); testData_B(:,2:end-1)];

    % --- Step 2: Keep only numeric columns ---
    numericMask = varfun(@isnumeric, features_train, 'OutputFormat','uniform');
    nonNumericCount = sum(~numericMask);
    features_train = features_train(:, numericMask);
    features_test  = features_test(:, numericMask);

    fprintf('Removed %d non-numeric feature columns.\n', nonNumericCount);

    % --- Step 3: Convert features to numeric matrices ---
    trainMat = table2array(features_train);
    testMat  = table2array(features_test);

    % --- Step 4: Compute global normalization stats from merged training set ---
    mu = mean(trainMat, 1);
    sigma = std(trainMat, 0, 1);
    sigma(sigma == 0) = 1; % avoid division by zero

    % --- Step 5: Normalize ---
    trainMatNorm = (trainMat - mu) ./ sigma;
    testMatNorm  = (testMat  - mu) ./ sigma;

    % --- Step 6: Adaptive low-variance filtering ---
    featureVar = var(trainMatNorm, 0, 1);
    adaptiveThreshold = 0.01 * mean(featureVar); % α = 0.1 (tunable)
    keepMask   = featureVar >= adaptiveThreshold;
    lowVarCount = sum(~keepMask);

    trainMatNorm   = trainMatNorm(:, keepMask);
    testMatNorm    = testMatNorm(:, keepMask);

    keptFeatureNames = features_train.Properties.VariableNames(keepMask);

    fprintf('Removed %d low-variance feature columns (adaptive threshold = %.4f).\n', ...
            lowVarCount, adaptiveThreshold);

    % --- Step 7: Reconstruct normalized tables ---
    trainData_C = [table(filenames_train,'VariableNames',{'filename'}), ...
                   array2table(trainMatNorm,'VariableNames',keptFeatureNames), ...
                   table(labels_train,'VariableNames',{'label'})];

    testData_C  = [table(filenames_test,'VariableNames',{'filename'}), ...
                   array2table(testMatNorm,'VariableNames',keptFeatureNames), ...
                   table(labels_test,'VariableNames',{'label'})];
end
