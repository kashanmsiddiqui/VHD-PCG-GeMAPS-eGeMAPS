function T = extractGeMAPSFeatures_11(presetHSData, openSMILE_bin, config_gemaps, config_egemaps)
% Extracts GeMAPS + eGeMAPS features via openSMILE from presetHSData struct
%
% INPUTS:
%   presetHSData    : struct array from PresetDurationModule
%   openSMILE_bin   : path to SMILExtract.exe
%   config_gemaps   : path to GeMAPSv01b.conf
%   config_egemaps  : path to eGeMAPSv02.conf
% OUTPUT:
%   T               : merged feature table saved to gemapsFeatures.xlsx

    if nargin < 2
        openSMILE_bin = 'C:\Program Files\MATLAB\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe';   % location of SMILExtract
        config_gemaps = 'C:\Program Files\MATLAB\opensmile-3.0.2-windows-x86_64\config\gemaps\v01b\GeMAPSv01b.conf';   % GeMAPS config
        config_egemaps = 'C:\Program Files\MATLAB\opensmile-3.0.2-windows-x86_64\config\egemaps\v02\eGeMAPSv02.conf'; % eGeMAPS config

    end

    if ~isfile(openSMILE_bin)
        error('SMILExtract not found: %s', openSMILE_bin);
    end

    tmpDir   = fullfile(tempdir, 'presetWAVs');
    if ~isfolder(tmpDir); mkdir(tmpDir); end

    N        = numel(presetHSData);
    merged_all = cell(N, 1);

    for i = 1:N
        fname  = presetHSData(i).filename;
        signal = presetHSData(i).signal;
        fs     = presetHSData(i).fs;

        % Write temp WAV
        tmpWav = fullfile(tmpDir, [fname '.wav']);
        audiowrite(tmpWav, signal, fs);

        out_gemaps  = fullfile(tmpDir, [fname '_gemaps.csv']);
        out_egemaps = fullfile(tmpDir, [fname '_egemaps.csv']);

        % Run GeMAPS
        cmd_gemaps = sprintf('"%s" -C "%s" -I "%s" -csvoutput "%s"', ...
            openSMILE_bin, config_gemaps, tmpWav, out_gemaps);
        system(cmd_gemaps);

        % Run eGeMAPS
        cmd_egemaps = sprintf('"%s" -C "%s" -I "%s" -csvoutput "%s"', ...
            openSMILE_bin, config_egemaps, tmpWav, out_egemaps);
        system(cmd_egemaps);

        % Read and merge
        try
            Tg  = readtable(out_gemaps,  'VariableNamingRule', 'preserve');
            Teg = readtable(out_egemaps, 'VariableNamingRule', 'preserve');
            Tg.Properties.VariableNames  = strcat('GeMAPS_',  Tg.Properties.VariableNames);
            Teg.Properties.VariableNames = strcat('eGeMAPS_', Teg.Properties.VariableNames);

            % Carry label
            labelVal = NaN;
            if isfield(presetHSData, 'label')
                lv = presetHSData(i).label;
                if isnumeric(lv)
                    labelVal = double(lv);
                else
                    labelVal = str2double(lv); % convert string/char to double
                end
            end
            
            merged_all{i} = [table({fname}, labelVal, ...
                             'VariableNames', {'filename','label'}), Tg, Teg];
        catch ME
            warning('Could not read features for %s: %s', fname, ME.message);
        end

        % Cleanup temp files
        if isfile(out_gemaps);  delete(out_gemaps);  end
        if isfile(out_egemaps); delete(out_egemaps); end
        if isfile(tmpWav);      delete(tmpWav);      end

        fprintf('GeMAPS processed %d / %d: %s\n', i, N, fname);
    end

    % Remove empty cells (failed files)
    merged_all = merged_all(~cellfun(@isempty, merged_all));
    T = vertcat(merged_all{:});
    
    % Reorder columns: filename → features → label
    varNames = T.Properties.VariableNames;
    filenameCol = "filename";
    labelCol    = "label";
    featureCols = setdiff(varNames, [filenameCol, labelCol], 'stable');
    
    T = T(:, [filenameCol, featureCols, labelCol]);

end
