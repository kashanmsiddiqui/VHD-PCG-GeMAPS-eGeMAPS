function presetHSData = PresetDurationModule_10(segmentedData, presetDurationSec, saveFormat)
% PresetDurationModule
% Creates fixed-duration heart sound signals after segmentation
%
% INPUTS:
%   segmentedData      : segmentedData struct array (from SpringerSegmentationModule)
%   presetDurationSec  : desired duration in seconds (e.g. 5, 10, 20) (default: 5)
%   saveFormat         : 'excel' or 'mat' (default: 'excel')
%
% OUTPUT:
%   presetHSData       : struct array with fixed-duration signals
% 
% HOW TO USE:
%   run PresetDurationModule(segmentedData, 3, 'mat')

    if nargin < 2 || isempty(presetDurationSec)
        presetDurationSec = 5;
    end
    if nargin < 3 || isempty(saveFormat)
        saveFormat = 'mat';
    end

    if isempty(segmentedData)
        error('Input segmentedData struct is empty.');
    end

    presetHSData = struct([]);

    %% Loop through recordings
    for k = 1:length(segmentedData)
        fs            = segmentedData(k).fs;
        targetSamples = presetDurationSec * fs;

        % --- Processed PCG must exist ---
        PCG = segmentedData(k).audioProcessed;
        if isempty(PCG)
            warning('Empty PCG for %s', segmentedData(k).filename);
            continue;
        end

        if length(PCG) >= targetSamples
            presetSignal = PCG(1:targetSamples);
        else
            presetSignal = zeros(targetSamples, 1);
            presetSignal(1:length(PCG)) = PCG;
        end

        % --- Save output ---
        presetHSData(k).filename     = segmentedData(k).filename;
        presetHSData(k).fs           = fs;
        presetHSData(k).duration_sec = presetDurationSec;
        presetHSData(k).signal       = presetSignal;
        presetHSData(k).nSamples     = length(presetSignal);
        % presetHSData(k).states       = segmentedData(k).states;

        if isfield(segmentedData, 'label')
            presetHSData(k).label = segmentedData(k).label;
        end

        fprintf('Preset duration %d/%d: %s\n', k, length(segmentedData), segmentedData(k).filename);
    end

    %% Save output
%     if strcmpi(saveFormat, 'excel')
%         exportPresetDataToExcel(presetHSData);
%         disp('✔ presetHSData.xlsx saved');
%     elseif strcmpi(saveFormat, 'mat')
%         save('presetHSData.mat', 'presetHSData', '-v7.3');
%         disp('✔ presetHSData.mat saved');
%     else
%         error('saveFormat must be ''excel'' or ''mat''');
%     end
end

%%
function exportPresetDataToExcel(presetHSData)
    rows = cell(length(presetHSData), 1);
    for k = 1:length(presetHSData)
        labelVal = '';
        if isfield(presetHSData, 'label')
            lv = presetHSData(k).label;
            if isnumeric(lv)
                labelVal = num2str(lv);
            else
                labelVal = char(lv);
            end
        end
        rows{k} = {
            presetHSData(k).filename, ...
            presetHSData(k).fs, ...
            presetHSData(k).duration_sec, ...
            presetHSData(k).nSamples, ...
            size(presetHSData(k).states, 1), ...   % num cardiac cycles
            labelVal
        };
    end
    % headers = {'filename', 'fs_Hz', 'duration_sec', 'nSamples', 'num_cycles', 'label'};
    % T = cell2table(vertcat(rows{:}), 'VariableNames', headers);
    % writetable(T, 'presetHSData.xlsx', 'Sheet', 'PresetDurationData');
end