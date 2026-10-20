function [trainData, testData] = splitTrainTest_PhysionetClassBalancedFolders_01(matTable)
% Reads feature table and splits into train/test sets
% with 80/20 proportion from each prefix group (a–f),
% while maintaining class balance (ratio of -1 vs 1).
%
% Inputs:
%   matTable   - feature table
%
% Outputs:
%   trainData   - table containing training samples
%   testData    - table containing testing samples

    % Read the Excel file into a table
    T = matTable;

    % Define prefixes
    prefixes = {'a','b','c','d','e','f'};

    % Initialize empty tables
    trainData = T([],:);
    testData  = T([],:);

    rng(42);

    % Loop over each prefix
    for i = 1:numel(prefixes)
        % Select rows for this prefix
        rowsPrefix = startsWith(T.filename, prefixes{i}, 'IgnoreCase', true);
        subset = T(rowsPrefix, :);

        % Convert Label to numeric if string
        labels = subset.label;

        % Separate normal (-1) and abnormal (1)
        idxNormal   = find(labels == -1);
        idxAbnormal = find(labels == 1);

        % Compute 80/20 split counts
        nNormalTrain   = max(1, round(0.8 * numel(idxNormal)));
        nAbnormalTrain = max(1, round(0.8 * numel(idxAbnormal)));

        % Randomly sample indices (row vectors)
        trainIdx = [randsample(idxNormal, nNormalTrain)', ...
                    randsample(idxAbnormal, nAbnormalTrain)'];
        
        % Shuffle order
        trainIdx = trainIdx(randperm(numel(trainIdx)));
        
        % Remaining go to test set
        testIdx = setdiff(1:height(subset), trainIdx);

        % Append to train/test tables
        trainData = [trainData; subset(trainIdx,:)];
        testData  = [testData; subset(testIdx,:)];
    end

    % Print summary
    fprintf('✔ Split complete: %d train samples, %d test samples\n', ...
            height(trainData), height(testData));
end
