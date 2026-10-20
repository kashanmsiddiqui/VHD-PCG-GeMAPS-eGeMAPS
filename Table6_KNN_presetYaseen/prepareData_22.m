function [X_train, y_train, X_test, y_test, classNames] = prepareData_22(trainTable, testTable)

% prepareData function
% Shared helper: extracts feature matrices and numeric labels from tables.
% Expects columns: filename, [features...], label
% 
% Keeps only numeric feature columns
% 
% Normalize feature columns in trainTable with z‑score 
% normalization (subtract mean, divide by standard deviation) and apply the
% same transformation to testTable thus ensuring consistency and preventing
% data leakage.
% 
% Identify and remove low-varaince columns, i.e., whose variance in trainTable 
% is below a minimum variance threshold
% 


    varianceThresholdFactor = 0.01; % 0.01 >>> 1% of avg variance
    
    % Identify feature columns (exclude metadata)
    featureCols = trainTable.Properties.VariableNames;
    featureCols = featureCols(~ismember(featureCols, {'filename','label'}));

    % Keep only numeric features
    isNumeric = varfun(@isnumeric, trainTable(:, featureCols), 'OutputFormat','uniform');
    numericFeatureCols = featureCols(isNumeric);
    nonNumericCount = sum(~isNumeric);

    fprintf('Removed %d non-numeric column(s)\n', nonNumericCount);

    % Convert to arrays
    X_train = table2array(trainTable(:, numericFeatureCols));
    X_test  = table2array(testTable(:,  numericFeatureCols));

    % --- Normalization (fit on training, apply to both) ---
    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1);
    sigma(sigma == 0) = 1; % avoid division by zero

    X_train = (X_train - mu) ./ sigma;
    X_test  = (X_test  - mu) ./ sigma;

    % --- Low-variance filtering after normalization ---
    % Compute variance of normalized training features
    featureVar = var(X_train, 0);
    threshold = varianceThresholdFactor * mean(featureVar);

    % Flag low-variance features
    lowVarCols = featureVar < threshold;

    % Keep only features above threshold
    X_train = X_train(:, ~lowVarCols);
    X_test  = X_test(:,  ~lowVarCols);
    numericFeatureCols = numericFeatureCols(~lowVarCols);
    
    fprintf('Removed %d low-variance columns\n', sum(lowVarCols));
    

    % Labels
    y_train = labelToNumeric(trainTable.label);
    y_test  = labelToNumeric(testTable.label);

    % Class names (human-readable)
    classNames = unique(labelToNumeric(trainTable.label));

end
