function [trainData, testData] = splitTrainTest_Yaseen_01(tableT)
% Splits tableT from Yaseen dataset into 80-20% train/test sets
% generated from 50 random recordings of each pathological VHD group 
% (AS, MR, MS, MVP) and all 200 recordings from normal VHD group (N)
%
% Inputs:
%   Table T - generated in workspace by extractShenFeatures_11 or loaded
%               from shenFeatures.mat
%
% Outputs:
%   trainData   - table containing training samples
%   testData    - table containing testing samples

    % Extract group from filename (before '_')
    % Example: "N_131.wav" → "N"
    groups = cellfun(@(x) regexp(x, '^[A-Z]+', 'match', 'once'), ...
                        tableT.filename, 'UniformOutput', false);

    rng(42); % reproducibility

    % Helper function: sample n recordings from group, excluding already used indices
    function idx = sampleGroup(grp,n,usedIdx)
        available = setdiff(find(strcmp(groups, grp)), usedIdx);
        if numel(available) < n
            error('Group %s has fewer than %d available recordings.', grp, n);
        end
        idx = randsample(available, n);
    end

    % --- TRAINING SET ---
    idxAS_train  = sampleGroup('AS', 40, []);
    idxMR_train  = sampleGroup('MR', 40, []);
    idxMS_train  = sampleGroup('MS', 40, []);
    idxMVP_train = sampleGroup('MVP', 40, []);
    idxN_train   = sampleGroup('N', 160, []);

    trainIdx = [idxAS_train; idxMR_train; idxMS_train; idxMVP_train; idxN_train];

    % --- TEST SET ---
    idxAS_test  = sampleGroup('AS', 10, trainIdx);
    idxMR_test  = sampleGroup('MR', 10, trainIdx);
    idxMS_test  = sampleGroup('MS', 10, trainIdx);
    idxMVP_test = sampleGroup('MVP', 10, trainIdx);
    idxN_test   = sampleGroup('N', 40, trainIdx);

    testIdx = [idxAS_test; idxMR_test; idxMS_test; idxMVP_test; idxN_test];

    % Build tables
    trainData = tableT(trainIdx, :);
    testData  = tableT(testIdx, :);

    % Shuffle rows
    trainData = trainData(randperm(height(trainData)), :);
    testData  = testData(randperm(height(testData)), :);


end
