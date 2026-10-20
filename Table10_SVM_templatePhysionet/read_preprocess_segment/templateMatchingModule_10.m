function templateHSData = templateMatchingModule_10(segmentedData, varargin)

% This function received segmentedHSdata, and saves the variables such as
% 'states' and 'PCG' into a local struct called 'segmentedData'.
% The 'states' are stored in cols = [S1, Systole, S2, Diastole].
% It then saves S1-to-S1 intervals as a new field 'detectedCycles', in the
% same struct.
% In next module, the best HS cycle amongst detectedCycles, is selected and
% saved against every HS file. The metrics used are RMS energy, Entropy and
% Kurtosis. The tables 'cycleScores' are also saved separately.
% The last module takes 'bestCycle' as template, compares it with all the
% 'detectedCycles', and finds best cross-correlated cycles. Then, the best 
% cross-correlated cycles are concatenated and saved in the field 'topMatchedConcat'
% of the struct 'segmentedHSData', which is the output of this function.
% This struct does not have initially saved fields of original signal 'PCG'
% and 'detectedCycles', for leaner data saving.
% 
% INPUTS:
%   segmentedHSData :   segmentedData (from SpringerSegmentationModule)
%   numCycles       :   number of template-matching cycles to be extracted
% 
% OPTIONAL NAME-VALUE INPUTS:
%   'NumberOfCycles': number of template-matching cycles  (default:   5)
%   'wRMS'          : weight for RMS energy [0.1 - 0.9]   (default: 0.7)
%   'wEntropy'      : weight for Entropy    [0.1 - 0.9]   (default: 0.3)
%   'wKurtosis'     : weight for Kurtosis   [0.1 - 0.9]   (default: 0.4)
% 
% OUTPUT:
%   templateHSData  :   struct array with template-matching signals
% 
% HOW TO USE:
%   run templateHSData = templateMatchingModule_10(segmentedHSData, 5)

    if isempty(segmentedData)
        error('Input segmentedData struct is empty.');
    end

    %% ---- Parse inputs ----
    p = inputParser;
    addParameter(p, 'NumberOfCycles',   5);
    addParameter(p, 'wRMS',           0.7);
    addParameter(p, 'wEntropy',       0.3);
    addParameter(p, 'wKurtosis',      0.4);
    parse(p, varargin{:});
    opt = p.Results;
    
    templateHSData = segmentedData;

    %% Find and save detectedCycles
    for k = 1:numel(templateHSData)
        states = templateHSData(k).states; % rows = cycles, cols = [S1, Systole, S2, Diastole]
        pcg    = templateHSData(k).PCG;
        
        % Extract S1 markers (column 1)
        S1_positions = states(:,1);
        
        % Build cycles: each cycle = PCG segment from S1(i) to S1(i+1)
        detectedCycles = cell(length(S1_positions)-1,1);
        for i = 1:(length(S1_positions)-1)
            cycle_start = S1_positions(i);
            cycle_end   = S1_positions(i+1);
            
            % Slice the PCG waveform for this cycle
            detectedCycles{i} = pcg(cycle_start:cycle_end);
        end
        
        % Add new field to struct
        templateHSData(k).detectedCycles = detectedCycles;
        templateHSData(k).numCyclesDetected = numel(detectedCycles);
    
        fprintf('\n Detecting Cycles in PCG File: %d\n', k);
    end

    %% Evaluate the best HS cycle among 'detectedCycles' 
    for k = 1:numel(templateHSData)
        cycles = templateHSData(k).detectedCycles;
        fs     = templateHSData(k).fs;
        
        % Check number of cycles
        if numel(cycles) < opt.NumberOfCycles
            warning('Warning: only %d cycles detected in %s', ...
                     numel(cycles), templateHSData(k).filename);
        end
        
        % Preallocate score table: [cycleNumber, RMS, entropy, kurtosis, totalScore]
        scoreTable = zeros(numel(cycles), opt.NumberOfCycles);
        
        for i = 1:numel(cycles)
            x = cycles{i};
            
            % --- Signal quality metrics ---
            % RMS energy
            rmsVal = sqrt(mean(x.^2));
            
            % Entropy (Shannon entropy of normalized histogram)
            p = histcounts(x,100,'Normalization','probability');
            p(p==0) = []; % avoid log(0)
            entropyVal = -sum(p .* log2(p));
            
            % Kurtosis
            kurtVal = kurtosis(x);
            
            % Combine metrics into weighted score
            totalScore = opt.wRMS*rmsVal + opt.wEntropy*entropyVal - opt.wKurtosis*abs(kurtVal-3);
            
            % Store results
            scoreTable(i,:) = [i, rmsVal, entropyVal, kurtVal, totalScore];
        end
        
         % Find best cycle index
        [~, bestIdx] = max(scoreTable(:,opt.NumberOfCycles));
    
    
        % Save best cycle and full score table
        templateHSData(k).bestCycle   = cycles{bestIdx};
        templateHSData(k).cycleScores = scoreTable;
    
        fprintf('\n Selecting best cycle in PCG File: %d\n', k);
    
    end

   

    for k = 1:numel(templateHSData)
        cycles   = templateHSData(k).detectedCycles;
        template = templateHSData(k).bestCycle(:);  % ensure column vector
        
        % Preallocate results
        corrScores = zeros(numel(cycles),1);
        
        for i = 1:numel(cycles)
            x = cycles{i}(:); % current cycle
            
            % --- Normalized correlation coefficient ---
            L = min(length(template), length(x));
            tSeg = template(1:L);
            xSeg = x(1:L);
            
            numerator   = sum(tSeg .* xSeg);
            denominator = sqrt(sum(tSeg.^2) * sum(xSeg.^2));
            
            if denominator > 0
                corrScores(i) = numerator / denominator; % [-1,1]
            else
                corrScores(i) = NaN;
            end
        end
        
        % Save cycle-wise correlation scores into struct
        templateHSData(k).cycleCorrScores = corrScores;
        
        % Create labeled table
        cycleNumbers = (1:numel(cycles))';
        scoreTable   = table(cycleNumbers, corrScores, ...
                             'VariableNames', {'CycleNumber','CorrelationWithBest'});
        templateHSData(k).cycleCorrTable = scoreTable;
        
        % --- Select top opt.NumberOfCycles cycles ---
        [~, sortIdx] = sort(corrScores, 'descend', 'MissingPlacement','last');
        topIdx       = sortIdx(1:min(opt.NumberOfCycles,numel(sortIdx)));
        
        templateHSData(k).topMatchedCycles = cycles(topIdx);
        templateHSData(k).topMatchedScores = corrScores(topIdx);
        
        % --- Concatenate top opt.NumberOfCycles cycles into one long vector ---
        concatenatedSignal = vertcat(templateHSData(k).topMatchedCycles{:});
        templateHSData(k).topMatchedConcat = concatenatedSignal;
        % 
        % % Export concatenated signal as WAV
        % [~, baseName, ~] = fileparts(templateHSData(k).filename);
        % outFile = fullfile(outFolder, [baseName '.wav']);
        % audiowrite(outFile, concatenatedSignal, fs);
        % 
        % fprintf('\n Exporting File: %d\n', k);
    
    end

    % Update struct by removing unwanted fields
    templateHSData = rmfield(templateHSData, {'PCG','detectedCycles'});
    
    % save the output struct as mat file for step-wise debugging
    save('templateMatchingHSData_Physionet.mat','templateHSData');

end
