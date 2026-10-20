% Compile KNN classification summaries from .mat files


folderPath = uigetdir(pwd, 'Select folder containing MAT files');
files = dir(fullfile(folderPath, 'KNN_Results_*.mat'));
nFiles = numel(files);

fprintf('%d "KNN_Results_" files found in the directory\n', nFiles);

% Get folder name only (last part of path)
[~, folderName] = fileparts(folderPath);

SummaryRows = {};
BroadRows   = {};
PubRows     = {};

for k = 1:nFiles
    fname = files(k).name;
    fpath = fullfile(folderPath, fname);
    
    tmp = load(fpath, 'KNNresults');
    if ~isfield(tmp, 'KNNresults') || ~isfield(tmp.KNNresults, 'cv') ...
            || ~isfield(tmp.KNNresults.cv, 'summary')
        warning('File %s does not contain expected structure.', fname);
        continue;
    end
    
    summary = tmp.KNNresults.cv.summary;
    fieldNames = fieldnames(summary);
    values = cellfun(@(f) summary.(f), fieldNames, 'UniformOutput', false);
    
    % Clean filename
    shortName = regexprep(fname, '^KNN_Results_', '');
    shortName = regexprep(shortName, '\.mat$', '');
    
    % --- Full summary row (CV only) ---
    SummaryRows{end+1,1} = shortName;
    SummaryRows(end,2:numel(fieldNames)+1) = values;
    
    if ~exist('metricNames','var')
        metricNames = fieldNames;
    else
        metricNames = union(metricNames, fieldNames, 'stable');
    end
    
    % --- Broad summary (means only) ---
    meanFields = metricNames(contains(metricNames, 'mean'));
    meanValues = cellfun(@(f) summary.(f), meanFields, 'UniformOutput', false);
    
    % CV row
    BroadRows{end+1,1} = [shortName '_CV'];
    BroadRows(end,2:numel(meanFields)+1) = meanValues;
    
    % Test row (align with meanFields by stripping 'mean_')
    if isfield(tmp.KNNresults, 'test')
        testStruct = tmp.KNNresults.test;
        testValues = cellfun(@(mf) ...
            testStruct.(regexprep(mf,'^mean_','')), ...
            meanFields, 'UniformOutput', false);
        
        BroadRows{end+1,1} = [shortName '_Test'];
        BroadRows(end,2:numel(meanFields)+1) = testValues;
    end
    
   % --- Pub results ---
    % CV row: mean + std, plus averaged inference metrics and aggregated confusion matrix
    cvValues = cellfun(@(f) summary.(f), metricNames, 'UniformOutput', false);
    PubRows{end+1,1} = [shortName '_CV'];
    PubRows(end,2:numel(metricNames)+1) = cvValues;
    
    % Average inference times across CV folds
    cv_totalInf = NaN; cv_avgInf = NaN; cv_confMatStr = NaN;
    if isfield(tmp.KNNresults.cv,'folds')
        folds = tmp.KNNresults.cv.folds;
        if isstruct(folds)
            % single struct array
            if isfield(folds,'totalInferenceTime_s')
                cv_totalInf = mean([folds.totalInferenceTime_s]);
            end
            if isfield(folds,'avgInferenceTime_ms')
                cv_avgInf = mean([folds.avgInferenceTime_ms]);
            end
            if isfield(folds,'confMat')
                % sum confusion matrices across folds
                aggConf = sum(cat(3,folds.confMat),3);
                cv_confMatStr = sprintf('TN=%d, FP=%d, FN=%d, TP=%d', ...
                    aggConf(1,1), aggConf(1,2), aggConf(2,1), aggConf(2,2));
            end
        elseif iscell(folds)
            % cell array of fold structs
            totalVals = cellfun(@(f) f.totalInferenceTime_s, folds);
            avgVals   = cellfun(@(f) f.avgInferenceTime_ms, folds);
            cv_totalInf = mean(totalVals);
            cv_avgInf   = mean(avgVals);
            if all(cellfun(@(f) isfield(f,'confMat'), folds))
                aggConf = sum(cat(3,folds{:}.confMat),3);
                cv_confMatStr = sprintf('TN=%d, FP=%d, FN=%d, TP=%d', ...
                    aggConf(1,1), aggConf(1,2), aggConf(2,1), aggConf(2,2));
            end
        end
    end
    
    PubRows(end,numel(metricNames)+2 : numel(metricNames)+3) = {cv_totalInf, cv_avgInf};
    PubRows(end,numel(metricNames)+4) = {cv_confMatStr};

    
    % --- Test row: mean only, NaN for std, plus inference metrics
    if isfield(tmp.KNNresults,'test')
        testStruct = tmp.KNNresults.test;
        testRowVals = cell(1,numel(metricNames));
        for m = 1:numel(metricNames)
            mf = metricNames{m};
            if startsWith(mf,'mean_')
                baseName = regexprep(mf,'^mean_','');
                if isfield(testStruct, baseName)
                    testRowVals{m} = testStruct.(baseName);
                else
                    testRowVals{m} = NaN;
                end
            elseif startsWith(mf,'std_')
                testRowVals{m} = NaN;
            else
                testRowVals{m} = NaN;
            end
        end
        
        PubRows{end+1,1} = [shortName '_Test'];
        PubRows(end,2:numel(metricNames)+1) = testRowVals;
        
        % Inference metrics from test struct
        totalInfVal = NaN;
        if isfield(testStruct,'totalInferenceTime_s')
            totalInfVal = testStruct.totalInferenceTime_s;
        end
        avgInfVal = NaN;
        if isfield(testStruct,'avgInferenceTime_ms')
            avgInfVal = testStruct.avgInferenceTime_ms;
        end
        
        % Flatten confusion matrix
        confMatStr = NaN;
        if isfield(testStruct,'confMat')
            C = testStruct.confMat;
            if numel(C)==4
                confMatStr = sprintf('TN=%d, FP=%d, FN=%d, TP=%d', ...
                    C(1,1), C(1,2), C(2,1), C(2,2));
            else
                confMatStr = mat2str(C);
            end
        end
        
        % Assign in correct order: totalInf, avgInf, confMat
        PubRows(end,numel(metricNames)+2 : numel(metricNames)+3) = {totalInfVal, avgInfVal};
        PubRows(end,numel(metricNames)+4) = {confMatStr};
    end



end

% Convert to tables
FullSummaryTable = cell2table(SummaryRows, ...
    'VariableNames', [{'filename'}, metricNames(:)']);

BroadSummaryTable = cell2table(BroadRows, ...
    'VariableNames', [{'filename'}, meanFields(:)']);

PubTable = cell2table(PubRows, ...
    'VariableNames', [{'filename'}, metricNames(:)', ...
    {'totalInferenceTime_s','avgInferenceTime_ms','confMat'}]);

% Export to Excel
outFile = fullfile(folderPath, ...
    ['KNN_Results_compiled_' folderName '_' datestr(now,'yyyymmdd_HHMM') '.xlsx']);
writetable(FullSummaryTable, outFile, 'Sheet', 'CV_MeanSD');
writetable(BroadSummaryTable, outFile, 'Sheet', 'CV_test_Mean');
writetable(PubTable, outFile, 'Sheet', 'pub_results');

fprintf('Compiled summary exported to %s\n', outFile);
