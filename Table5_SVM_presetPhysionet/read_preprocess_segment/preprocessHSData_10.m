function procHSData = preprocessHSData_10(heartSoundData)
% preprocessHeartSound
% Preprocesses each recording from a heartSoundData struct array
% and saves procHSData.mat 
%
% INPUT:
%   rawHSData  - heartSoundData struct (from readHSData functions)
%
% OUTPUT:
%   procHSData - processed heart sound struct and procHSData.mat file

    if isempty(heartSoundData)
        error('Input struct array is empty.');
    end

    %% ---- Initialize output struct ----
    procHSData = struct([]);

    %% ---- Process each recording ----
    for k = 1:numel(heartSoundData)

        % --- Step 1: Convert to mono ---
        if heartSoundData(k).numChannels > 1
            signal = mean(heartSoundData(k).audioData, 2);
        else
            signal = heartSoundData(k).audioData;
        end

        % --- Step 2: Resample to 2000 Hz ---
        targetFs = 2000;
        [P, Q] = rat(targetFs / heartSoundData(k).fs);
        signal = resample(signal, P, Q);

        % --- Step 3: Bandpass filter (25–400 Hz) ---
        bpFilt = designfilt('bandpassiir', ...
                            'FilterOrder', 4, ...
                            'HalfPowerFrequency1', 20, ...
                            'HalfPowerFrequency2', 350, ...
                            'SampleRate', targetFs, ...
                            'DesignMethod', 'butter' ...
                            );
        signal = filtfilt(bpFilt, signal);

        % --- Step 4: Normalize HS Signal ---
        maxVal = max(abs(signal));
        if maxVal > 0
            signal = signal ./ maxVal;
        end

%         % --- Step 5: Envelope filter smoothing (50 ms) ---
%         winLength = round(0.05 * targetFs);
%         envSignal = movmean(abs(signal), winLength);
% 
%         % --- Step 6: Normalize Envelope Signal ---
%         maxVal = max(abs(envSignal));
%         if maxVal > 0
%             envSignal = envSignal ./ maxVal;
%         end

        %% ---- Store processed data ----
        procHSData(k).filename       = heartSoundData(k).filename;
        % 'record' field exists in PhysioNet struct but not Yaseen — fallback to filename
        if isfield(heartSoundData, 'record')
            procHSData(k).record     = heartSoundData(k).record;
        else
            procHSData(k).record     = heartSoundData(k).filename;
        end
        procHSData(k).fs             = targetFs;
        procHSData(k).duration       = length(signal) / targetFs;
        procHSData(k).audioProcessed = signal;
%         procHSData(k).envelope       = envSignal;
        procHSData(k).nSamples       = length(signal);
        procHSData(k).duration       = length(signal) / targetFs;
        procHSData(k).label          = heartSoundData(k).label;

        fprintf('Preprocessed %d/%d: %s\n', k, numel(heartSoundData), heartSoundData(k).filename);
    end
% save('procHSData.mat', 'procHSData');
end
