%% Heart Sound ML Pipeline
% ==========================================================
% KNN Classification of Shen, GeMAPS/eGeMAPS and Combined
% feature set
% 
% This script follows this sequence:-
% Step 1: Loads the already saved readHSData incl 'Unsure' recordings
% Step 2: Preprocesses the data with best paramters
% Step 3: No Segmentation
% Step 4: Extracts 10s preset duration signal from procHSData
% Step 5: Extracts Shen features from presetHSData (Shen, GeMAPS/eGeMAPS,
% Combined)
% Step 6: Runs 10-fold KNN classifier with best settings
% and reports results
% 
% ==========================================================


clc; clear; clearvars;

%% Step 1: Loads the already saved readHSData
% ============================================
load('D:\Documents\MATLAB files\Matlab files - GeMAPS paper\First Revision - GeMAPS\readHSDataYaseen.mat');
filenamesCell = {heartSoundData.filename};
filenamesCell = regexprep(filenamesCell, '^New_', '');
[heartSoundData.filename] = deal(filenamesCell{:});

%% Step 2: Preprocessing with best paramters (Butter, 4th Order, 20-350 Hz)
% =========================================================================
procHSData = preprocessHSData_13(heartSoundData);


%% Step 3: No Springer Segmentation
% =================================
% segmentedHSData = springerSegmentationModule_10(procHSData);


%% Step 4: Extract preset duration (10s)
% ======================================
presetHSData = PresetDurationModule_10(procHSData);


%% Step 5-6: Extract features, split train-test sets, classify and report results
% ===============================================================================

% Shen Feature Set

shenT = extractShenFeatures_11(presetHSData);
[trainData, testData] = splitTrainTest_Yaseen_01(shenT);
KNNresults = classifyKNN_22(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('KNN_Results_withoutSFFS_shen_preset_Physionet_%s.mat', char(timestamp));
save(filename,'KNNresults');

% GeMAPS / eGeMAPS Feature Set

gemapsT = extractGeMAPSFeatures_11(presetHSData);
[trainData, testData] = splitTrainTest_Yaseen_01(gemapsT);
KNNresults = classifyKNN_22(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('KNN_Results_withoutSFFS_GeMAPS_preset_Physionet_%s.mat', char(timestamp));
save(filename,'KNNresults');

% Combined Feature Set

T_merged = combineFeatures_11(shenT, gemapsT);
[trainData, testData] = splitTrainTest_Yaseen_01(T_merged);
KNNresults = classifyKNN_22(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('KNN_Results_withoutSFFS_combined_preset_Physionet_%s.mat', char(timestamp));
save(filename,'KNNresults');
