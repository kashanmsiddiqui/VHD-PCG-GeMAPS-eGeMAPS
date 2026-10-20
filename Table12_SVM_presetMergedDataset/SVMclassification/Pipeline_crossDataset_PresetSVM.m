%% Heart Sound ML Pipeline
% ==========================================================
% SVM Classification of combined respective training and testing 
% sets from Physionet and Yaseen using Shen, GeMAPS/eGeMAPS and Combined
% feature set
% 
% This script follows this sequence:-
% Step 1:   a)  Loads the already saved readHSData excl 'Unsure' recordings 
%               from Physionet
%           b)  Preprocesses and extracts presetHSData
%           c)  Extracts shenP, gemapsP and combinedP
%           d)  Separates out the respective training and testing sets
%               labelled as trainData_shenP, trainData_gemapsP, 
%               trainData_combinedP, testData_shenP, 
%               testData_gemapsP and testData_combinedP 
% Step 2:   a)  Loads the already saved readHSData from Yaseen
%           b)  Preprocesses and extracts presetHSData
%           c)  Extracts shenY, gemapsY and combinedY
%           d)  Separates out the respective training and testing sets 
%               labelled as trainData_shenY, trainData_gemapsY, 
%               trainData_combinedY, testData_shenY, 
%               testData_gemapsY and testData_combinedY
% Step 3:   Combines and standardizes the corresponding training and 
%           testing data labelled as trainData_shenC, trainData_gemapsC, 
%           trainData_combinedC, testData_shenC, 
%           testData_gemapsC and testData_combinedC 
% Step 4:   Runs 10-fold SVM classifier with best settings
%           on respective training and testing sets and reports results
% ==========================================================

clc; clear; clearvars;

%% Step 1: Preparation of training and testing sets - Physionet
% =============================================================
load('D:\Documents\MATLAB files\Matlab files - GeMAPS paper\First Revision - GeMAPS\readHSData with label - Physionet\readHSDataPhysionet_exclUnsure.mat');
procHSData = preprocessHSData_13(heartSoundData);
presetHSData = PresetDurationModule_10(procHSData, 5);

shenP       = extractShenFeatures_11(presetHSData);
gemapsP     = extractGeMAPSFeatures_11(presetHSData);
combinedP   = combineFeatures_11(shenP, gemapsP);

[trainData_shenP, testData_shenP]           = splitTrainTest_PhysionetClassBalancedFolders_01(shenP);
[trainData_gemapsP, testData_gemapsP]       = splitTrainTest_PhysionetClassBalancedFolders_01(gemapsP);
[trainData_combinedP, testData_combinedP]   = splitTrainTest_PhysionetClassBalancedFolders_01(combinedP);

save('trainData_shenP.mat', 'trainData_shenP');
save('testData_shenP.mat', 'testData_shenP');
save('trainData_gemapsP.mat', 'trainData_gemapsP');
save('testData_gemapsP.mat', 'testData_gemapsP');
save('trainData_combinedP.mat', 'trainData_combinedP');
save('testData_combinedP.mat', 'testData_combinedP');

%% Step 2: Preparation of training and testing sets - Yaseen
% ==========================================================
load('D:\Documents\MATLAB files\Matlab files - GeMAPS paper\First Revision - GeMAPS\readHSDataYaseen.mat');
procHSData = preprocessHSData_13(heartSoundData);
presetHSData = PresetDurationModule_10(procHSData, 5);

shenY       = extractShenFeatures_11(presetHSData);
gemapsY     = extractGeMAPSFeatures_11(presetHSData);
combinedY   = combineFeatures_11(shenY, gemapsY);

[trainData_shenY, testData_shenY]           = splitTrainTest_Yaseen_02(shenY);
[trainData_gemapsY, testData_gemapsY]       = splitTrainTest_Yaseen_02(gemapsY);
[trainData_combinedY, testData_combinedY]   = splitTrainTest_Yaseen_02(combinedY);

save('trainData_shenY.mat', 'trainData_shenY');
save('testData_shenY.mat', 'testData_shenY');
save('trainData_gemapsY.mat', 'trainData_gemapsY');
save('testData_gemapsY.mat', 'testData_gemapsY');
save('trainData_combinedY.mat', 'trainData_combinedY');
save('testData_combinedY.mat', 'testData_combinedY');

%% Step 3: Combine and standardize the training and testing sets
% ==============================================================
[trainData_shenC, testData_shenC]           = mergeCrossdatasetTables_01(trainData_shenP, testData_shenP, trainData_shenY, testData_shenY);
[trainData_gemapsC, testData_gemapsC]       = mergeCrossdatasetTables_01(trainData_gemapsP, testData_gemapsP, trainData_gemapsY, testData_gemapsY);
[trainData_combinedC, testData_combinedC]   = mergeCrossdatasetTables_01(trainData_combinedP, testData_combinedP, trainData_combinedY, testData_combinedY);


%% Step 4: Classify and report results
% ====================================

% Cross-dataset Combined Shen Feature Set
SVMresults = classifySVM_18(trainData_shenC, testData_shenC);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('SVM_Results_crossDataset_shen_%s.mat', char(timestamp));
save(filename,'SVMresults');

% Cross-dataset Combined GeMAPS / eGeMAPS Feature Set

SVMresults = classifySVM_18(trainData_gemapsC, testData_gemapsC);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('SVM_Results_crossDataset_GeMAPS_%s.mat', char(timestamp));
save(filename,'SVMresults');

% Cross-dataset Combined Combined Feature Set

SVMresults = classifySVM_18(trainData_combinedC, testData_combinedC);

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
filename = sprintf('SVM_Results_crossDataset_combined_%s.mat', char(timestamp));
save(filename,'SVMresults');

