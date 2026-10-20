function heartSoundData = readHSData_10()
% Reads heart sound WAV files from ALL subfolders,
% attaches labels from each folder's REFERENCE-SQI.csv,
% and saves a combined labeled struct array, for only those files whose
% 'sure' tag is '1'

    %% ---- Root folder ----
    rootPath = 'D:\PCG_ML_VHD\First Revision - GeMAPS\Codes run by Kashan - 3jun2026\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0';

    if ~isfolder(rootPath)
        error('Root training folder does not exist.');
    end

    %% ---- Get all subfolders ----
    folders = dir(rootPath);
    folders = folders([folders.isdir]);
    folders = folders(~ismember({folders.name}, {'.', '..'}));

    %% ---- Initialize struct ----
    heartSoundData = struct([]);
    idxGlobal = 1;
    globalSure = 0;

    %% ---- Loop over folders ----
    for f = 1:length(folders)

        folderPath = fullfile(rootPath, folders(f).name);
        fprintf('\n📁 Processing folder: %s\n', folders(f).name);

        %% ---- Collect WAV files ----
        wavFiles = dir(fullfile(folderPath, '*.wav'));

        if isempty(wavFiles)
            warning('No WAV files in %s. Skipping...', folders(f).name);
            continue;
        end

        %% ---- Read REFERENCE-SQI.csv ----
        refFile = fullfile(folderPath, 'REFERENCE-SQI.csv');

        if ~isfile(refFile)
            warning('REFERENCE-SQI.csv missing in %s. Skipping folder...', folders(f).name);
            continue;
        end

        refTable = readtable(refFile, 'ReadVariableNames', false);
        refTable.Properties.VariableNames = {'record', 'label', 'sure'};

        %% ---- Loop through WAV files ----
        for i = 1:length(wavFiles)

            filename = fullfile(folderPath, wavFiles(i).name);
            [audioData, fs] = audioread(filename);
            info = audioinfo(filename);

            % Remove extension for label matching
            [~, recordName, ~] = fileparts(wavFiles(i).name);

            % Find label
            idx = strcmp(refTable.record, recordName);

            if any(idx)
                label = refTable.label(idx);
                sure = refTable.sure(idx);
            else
                label = NaN;   % safety
            end

            % Store in struct
            heartSoundData(idxGlobal).filename      = wavFiles(i).name;
            heartSoundData(idxGlobal).folder        = folders(f).name;
            heartSoundData(idxGlobal).record        = recordName;
            heartSoundData(idxGlobal).fs            = fs;
            heartSoundData(idxGlobal).audioData     = audioData;
            heartSoundData(idxGlobal).nSamples      = length(audioData);
            heartSoundData(idxGlobal).duration      = info.Duration;
            heartSoundData(idxGlobal).bitsPerSample = info.BitsPerSample;
            heartSoundData(idxGlobal).numChannels   = info.NumChannels;
            heartSoundData(idxGlobal).label         = label;
            heartSoundData(idxGlobal).sure          = sure;

            fprintf('  Loaded %s/%s (label=%d)\n', ...
                    folders(f).name, wavFiles(i).name, label);

            if sure == 1
                idxGlobal = idxGlobal + 1;
            else
                globalSure = globalSure + 1;
            end
        end
    end

    %% ---- Save combined dataset ----
    save('readHSDataPhysionet.mat', 'heartSoundData', '-v7.3');

    fprintf('\n✔ Physionet dataset saved as readHSDataPhysionet.mat\n');
    fprintf('✔ Total recordings: %d\n', length(heartSoundData));
    fprintf('Unsure recordings skipped: %d\n', globalSure);

end
