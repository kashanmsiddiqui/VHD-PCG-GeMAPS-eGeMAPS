function segmentedData = springerSegmentationModule(procHSData, saveFormat, includePCG)
    % SpringerSegmentationModule
    % Runs Springer HSMM segmentation and saves results with optional PCG inclusion
    %
    % INPUTS:
    %   procHSData  : procHSData struct array (from preprocessHeartSound)
    %   saveFormat  : 'excel' or 'mat' (default: 'excel')
    %   includePCG  : true/false - include processed PCG in output (default: true)
    %
    % OUTPUT:
    %   segmentedData : struct with segmentation results (and PCG if includePCG=true)

    if nargin < 2 || isempty(saveFormat)
        saveFormat = 'mat';
    end

    if nargin < 3 || isempty(includePCG)
        includePCG = true;
    end

    if isempty(procHSData)
        error('Input procHSData struct is empty.');
    end

    %% Load Springer trained parameters
    load Springer_B_matrix
    load Springer_pi_vector
    load Springer_total_obs_distribution

    springer_options = default_Springer_HSMM_options;
    springer_options.use_mex = 1;
    segmentedData = struct([]);

    %% Loop through recordings
    for k = 1:length(procHSData)
        PCG = procHSData(k).audioProcessed;
        Fs  = procHSData(k).fs;

        % --- Run Springer segmentation ---
        assigned_states = runSpringerSegmentationAlgorithm( ...
            PCG, Fs, ...
            Springer_B_matrix, ...
            Springer_pi_vector, ...
            Springer_total_obs_distribution, ...
            false);

        idx_states = get_states(assigned_states);

        % --- Save results ---
        segmentedData(k).filename      = procHSData(k).filename;
        segmentedData(k).fs            = Fs;
        segmentedData(k).states        = idx_states;
        segmentedData(k).S1_mask       = idx_states == 1;
        segmentedData(k).Systole_mask  = idx_states == 2;
        segmentedData(k).S2_mask       = idx_states == 3;
        segmentedData(k).Diastole_mask = idx_states == 4;

        if isfield(procHSData, 'label')
            segmentedData(k).label = procHSData(k).label;
        end

        if includePCG
            segmentedData(k).PCG = PCG;
        end

        fprintf('Segmented %d/%d: %s\n', k, length(procHSData), procHSData(k).filename);
    end

    %% Save output
    if strcmpi(saveFormat, 'excel')
        exportSegmentationToExcel(segmentedData);
        disp('✔ segmentedHSData.xlsx saved');
    elseif strcmpi(saveFormat, 'mat')
        if includePCG
            save('segmentedHSData_withPCG.mat', 'segmentedData', '-v7.3');
            disp('✔ segmentedHSData_withPCG.mat saved');
        else
            save('segmentedHSData.mat', 'segmentedData', '-v7.3');
            disp('✔ segmentedHSData.mat saved');
        end
    else
        error('saveFormat must be ''excel'' or ''mat''');
    end
end

%%
function exportSegmentationToExcel(segmentedData)
    rows = cell(length(segmentedData), 1);
    for k = 1:length(segmentedData)
        labelVal = '';
        if isfield(segmentedData, 'label')
            lv = segmentedData(k).label;
            if isnumeric(lv)
                labelVal = num2str(lv);
            else
                labelVal = char(lv);
            end
        end
        rows{k} = {
            segmentedData(k).filename, ...
            segmentedData(k).fs, ...
            size(segmentedData(k).states, 1), ...   % number of cardiac cycles found
            labelVal
        };
    end
    headers = {'filename', 'fs_Hz', 'num_cycles', 'label'};
    T = cell2table(vertcat(rows{:}), 'VariableNames', headers);
    writetable(T, 'segmentedHSData.xlsx', 'Sheet', 'SegmentationResults');
end

%%
function idx_states = get_states(assigned_states)
    indx = find(abs(diff(assigned_states))>0);
    if assigned_states(1)>0
        switch assigned_states(1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=4;
        end
    else
        switch assigned_states(indx(1)+1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=0;
        end
        K=K+1;
    end
    indx2                = indx(K:end);
    rem                  = mod(length(indx2),4);
    indx2(end-rem+1:end) = [];
    idx_states           = reshape(indx2,4,length(indx2)/4)';
end