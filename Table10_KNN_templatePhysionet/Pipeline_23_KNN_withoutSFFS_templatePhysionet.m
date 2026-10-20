%% Heart Sound ML Pipeline
% ==========================================================
% KNN Classification of Shen, GeMAPS/eGeMAPS and Combined
% feature set
% 
% This script follows this sequence:-
% Step 1: Loads the already saved readHSData 
% Step 2: Preprocesses the data with best paramters
% Step 3: Springer Segmentation
% Step 4: Extracts template-matching signal from procHSData
% Step 5: Extracts features from templateHSData (Shen, GeMAPS/eGeMAPS,
% Combined)
% Step 6: Runs 10-fold SVM classifier with best settings
% and reports results
% 
% ==========================================================

clc; clear; clearvars;

%% Step 1: Loads the already saved readHSData
% ============================================

% Physionet excl Unsure
load('D:\Documents\MATLAB files\Matlab files - GeMAPS paper\First Revision - GeMAPS\readHSData with label - Physionet\readHSDataPhysionet_exclUnsure.mat');


%% Step 2: Preprocessing with best paramters (Butter, 4th Order, 20-350 Hz)
% =========================================================================
procHSData = preprocessHSData_13(heartSoundData);


%% Step 3: No Springer Segmentation
% =================================
segmentedHSData = springerSegmentationModule_10(procHSData);


%% Step 4: Extract preset duration (5s)
% ======================================
templateHSData = templateMatchingModule_10(segmentedHSData, 'NumberOfCycles', 5);


%% Step 5-6: Extract features, split train-test sets, classify and report results
% ===============================================================================

% Shen Feature Set

shenT = extractShenFeatures_12(templateHSData);
[trainData, testData] = splitTrainTest_PhysionetClassBalancedFolders_02(shenT);
KNNresults = classifyKNN_22(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('KNN_Results_withoutSFFS_shen_preset_Physionet_%s.mat', char(timestamp));
save(filename,'KNNresults');

% GeMAPS / eGeMAPS Feature Set

gemapsT = extractGeMAPSFeatures_12(templateHSData);
[trainData, testData] = splitTrainTest_PhysionetClassBalancedFolders_02(gemapsT);
KNNresults = classifyKNN_22(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('KNN_Results_withoutSFFS_GeMAPS_preset_Physionet_%s.mat', char(timestamp));
save(filename,'KNNresults');

% Combined Feature Set

combinedT = combineFeatures_11(shenT, gemapsT);
[trainData, testData] = splitTrainTest_PhysionetClassBalancedFolders_02(combinedT);
KNNresults = classifyKNN_22(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('KNN_Results_withoutSFFS_combined_preset_Physionet_%s.mat', char(timestamp));
save(filename,'KNNresults');

save('shenT.mat', 'shenT');
save('gemapsT.mat', 'gemapsT');
save('combinedT.mat', 'combinedT');
