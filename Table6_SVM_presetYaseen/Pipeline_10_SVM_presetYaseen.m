%% Heart Sound ML Pipeline
% ==========================================================
% SVM Classification of Shen, GeMAPS/eGeMAPS and Combined
% feature set
% 
% This script follows this sequence:-
% Step 1: Loads the already saved readHSData incl 'Unsure' recordings
% Step 2: Preprocesses the data with best paramters
% Step 3: No Segmentation
% Step 4: Extracts preset duration signal from procHSData
% Step 5: Extracts features from presetHSData (Shen, GeMAPS/eGeMAPS,
% Combined)
% Step 6: Runs 5-fold SVM classifier with best settings
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


%% Step 4: Extract preset duration (5s)
% ======================================
presetHSData = PresetDurationModule_10(procHSData, 5);

%% Step 5-6: Extract features, split train-test sets, classify and report results
% ===============================================================================

% Shen Feature Set

shenT = extractShenFeatures_11(presetHSData);
[trainData, testData] = splitTrainTest_Yaseen_01(shenT);
SVMresults = classifySVM_18(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('SVM_Results_shen_preset_Yaseen_%s.mat', char(timestamp));
save(filename,'SVMresults');

% GeMAPS / eGeMAPS Feature Set

gemapsT = extractGeMAPSFeatures_11(presetHSData);
[trainData, testData] = splitTrainTest_Yaseen_01(gemapsT);
SVMresults = classifySVM_18(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('SVM_Results_GeMAPS_preset_Yaseen_%s.mat', char(timestamp));
save(filename,'SVMresults');

% Combined Feature Set

T_merged = combineFeatures_11(shenT, gemapsT);
[trainData, testData] = splitTrainTest_Yaseen_01(T_merged);
SVMresults = classifySVM_18(trainData, testData);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('SVM_Results_combined_preset_Yaseen_%s.mat', char(timestamp));
save(filename,'SVMresults');

