%% Heart Sound ML Pipeline
% ==========================================================
% Ablation : Grid-search for preset duration
% 
% This script follows this sequence:-
% Step 1: Loads the already saved readHSData incl 'Unsure' recordings
% Step 2: Preprocesses the data with best paramters
% Step 3: No Segmentation 
% Step 4: Extracts various preset durations [2,3,5,10s] signal from procHSData
% Step 5: Extracts Shen features from presetHSData
% Step 6: Runs 5-fold SVM classifier and reports results
% 
% ==========================================================
% 
% Files needed for this pipeline:-
% preprocessHSData_11.m
% springerSegmentationModule_10
% Folder: 'Springer Segmentation Helper', update location in springerSegmentationModule_10
% PresetDurationModule_10
% extractShenFeatures_11
% splitTrainTest_PhysionetClassBalancedFolders_02
% classifySVM_13
% prepareData_21
% computeMetrics_10
% 

clc; clear; clearvars;

%% Step 1: Loads the already saved readHSData
% ============================================

load('D:\PCG_ML_VHD\First Revision - GeMAPS\Physionet Preset\readHSData with label - Physionet\readHSDataPhysionet_inclUnsure.mat');


%% Step 2: Preprocessing with best paramters
% ==========================================
procHSData = preprocessHSData_13(heartSoundData);

%% Step 3: No Springer Segmentation
% =================================
% segmentedHSData = springerSegmentationModule_10(procHSData);

%% Step 4 - 6: Grid search for Preset Duration with Feature extraction and Classification
% =======================================================================================

presetDuration = [2, 3, 5, 10];

for pd = 1:numel(presetDuration)

    % --- Step 4: Extract preset duration ---

    fprintf('\nExtracting preset duration: %d sec\n', presetDuration(pd));
    presetHSData = PresetDurationModule_10(procHSData, presetDuration(pd));

    % --- Step 5: Extract Shen features ---
    shenT = extractShenFeatures_11(presetHSData);

    % --- Step 6: SVM classification ---
    [trainData, testData] = splitTrainTest_PhysionetClassBalancedFolders_02(shenT);
    
    SVMresults = classifySVM_23(trainData, testData);

    % --- Save results with config in filename ---
    timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
    filename = sprintf('SVM_Results_preset%ds_%s.mat', ...
                       presetDuration(pd), char(timestamp));
    save(filename,'SVMresults');

end
