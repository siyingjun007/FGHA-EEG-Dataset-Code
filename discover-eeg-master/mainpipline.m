%%
%# Citation for DISCOVER-EEG-2.0.0
%# If this code contributes to a project that leads to a scientific publication, please acknowledge this work by citing the following paper:
%# Gil Ávila C, Bott FS, Tiemann L, Hohn VD, May ES, Nickel MM, Zebhauser PT, Gross J, Ploner P. DISCOVER-EEG: an open, fully automated EEG pipeline for biomarker discovery in clinical neuroscience. Sci Data 10, 613 (2023). doi:10.1038/s41597-023-02525-0
%# Additionally, cite the specific version of DISCOVER-EEG used in your analyses: 10.5281/zenodo.8207523

% Imports EEG dataset, preprocesses it
% 
% Cristina Gil, TUM, cristina.gil@tum.de, 25.07.2022
clear all; close all;
rng('default'); 
% Define the parameters
params = define_params('params.json');
ft_defaults;
%% ======= IMPORT RAW DATA =========
% Try to load the already created study, otherwise import raw data with pop_importbids
if exist(fullfile(params.PreprocessedDataPath,[params.StudyName '.study']),'file')
    [STUDY, ALLEEG] = pop_loadstudy('filename', [params.StudyName '.study'], 'filepath', params.PreprocessedDataPath);
else
    % Import raw data in BIDS format
    [STUDY, ALLEEG] = pop_importbids(params.RawDataPath,'outputdir',params.PreprocessedDataPath,...
        'studyName',params.StudyName,'sessions',params.Session,'runs',params.Run,'bidstask',params.Task,...
        'bidschanloc',params.BidsChanloc,'bidsevent','off');  
    for iRec=1:length(ALLEEG)
        % Retrieve data
        EEGtemp = eeg_checkset(ALLEEG(iRec),'loaddata');      
        % For event-related data, check if the recording has the specified EventMarker
        targetMarkers = strsplit(params.EventMarker, ',');        
        for i = 1:numel(targetMarkers)
            detectedInstances = 0;
            if any(strcmp({EEGtemp.event.type}, targetMarkers{i}))
                detectedInstances = detectedInstances + sum(strcmp({EEGtemp.event.type}, targetMarkers{i}));              
                if detectedInstances > 0
                    fprintf('We detected %d instances of marker(s) %s.\n', detectedInstances, targetMarkers{i});
                else
                    error('We could not find any of the EventMarkers in the data.');
                end
            end
        end
        
        % Add reference electrode
        EEGtemp = pop_chanedit(EEGtemp, 'append',EEGtemp.nbchan, ...
            'changefield', {EEGtemp.nbchan+1,'labels',ALLEEG(iRec).BIDS.tInfo.EEGReference},...
            'changefield', {EEGtemp.nbchan+1, 'X', params.RefCoord.X}, ...
            'changefield', {EEGtemp.nbchan+1, 'Y', params.RefCoord.Y}, ...
            'changefield', {EEGtemp.nbchan+1, 'Z', params.RefCoord.Z},...
            'setref',{['1:' num2str(EEGtemp.nbchan)],ALLEEG(iRec).BIDS.tInfo.EEGReference});
        
        % Use electrode positions from the electrodes.tsv file or from a standard template in the MNI coordinate system
        if strcmp(params.BidsChanloc, 'on')
            % If electrode positions are chosen from the .tsv, the coordinate
            % system might need to be adjusted (user has to define it in params.json)
            EEGtemp = pop_chanedit(EEGtemp, 'nosedir',params.NoseDir);
            eegchans = find(contains(lower({ALLEEG(iRec).chanlocs.type}),'eeg'));
        else
            % Look for electrode positions in a standard template
            EEGtemp=pop_chanedit(EEGtemp, 'lookup','standard_1020.elc');
            non_standard_chans = cellfun(@isempty,{EEGtemp.chanlocs.X});
            eegchans = find(~non_standard_chans);
            if any(non_standard_chans)
                clabels = {EEGtemp.chanlocs(non_standard_chans).labels};
                c = sprintf('%s ', clabels{:});
                warning(['The position of the channel(s) ' c 'was not found in a standard template and they will be removed. If you want to include them please specify their position in a electrodes.tsv and change define_params accordingly.']);
            end
        end
        
        % Select only EEG channels for preprocessing
        EEGtemp = pop_select(EEGtemp, 'channel', eegchans);
        EEGtemp.chaninfo.removedchans = [];        
        % Save datafile, clear it from memory and store it in the ALLEEG structure
        EEGtemp = pop_saveset(EEGtemp, 'savemode', 'resave');
        EEGtemp.data = 'in set file';
        ALLEEG = eeg_store(ALLEEG, EEGtemp, iRec);
        STUDY = pop_savestudy(STUDY, ALLEEG, 'filename', params.StudyName, 'filepath', params.PreprocessedDataPath);
        
    end
end
% OPTIONAL - Check that the electrode positions are ok
% figure; topoplot([],ALLEEG(1).chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo',ALLEEG(1).chaninfo);
% figure; topoplot([],ALLEEG(1).chaninfo.nodatchans, 'style', 'blank',  'electrodes', 'labelpoint','chaninfo',ALLEEG(1).chaninfo);

%% ======== PREPROCESSING =========
% Find the latest preprocessed recording and start with the next one
not_preprocessed = find(~cellfun(@(x) strcmp(x,'preprocessed'), {ALLEEG.comments}));
to_delete = {};
% Log file of all the recordings
fid0 = fopen(fullfile(params.ReportsPath,'preprocessing.log'),'a');
% Loop over the recordings that have not been preprocessed
params.EpochLength=6;
params.DownsamplingRate=500;
params.RejectBadTimeSegments = 'on';
params.FlatLineCriterion= 5;
params.ChannelCriterion= 0.8;
params.LineNoiseCriterion= 4;
params.HighPass=[0.25,0.75];
column_counts = [];
total_recording_length = [];
for iRec =not_preprocessed  
    t1 = tic;
    % Log file of single recordings
    id = regexp(ALLEEG(iRec).filename,'.*(?=(_eeg.set))','match');
    id = id{:};
    fid = fopen(fullfile(params.ReportsPath,[id '.log']),'a');
    % Retrieve data
    EEGtemp = eeg_checkset(ALLEEG(iRec),'loaddata');   
    % 0. OPTIONAL. DOWNSAMPLE DATA
    EEGtemp = pop_resample(EEGtemp, params.DownsamplingRate);
    fprintf(fid,'0. Downsampling performed. \n');   
    
    % 1. CLEAN LINE NOISE
    try
        EEGtemp = pop_cleanline(EEGtemp,'linefreqs',EEGtemp.BIDS.tInfo.PowerLineFrequency,'newversion',1);
        fprintf(fid,'1. CleanLine performed. \n');
    catch ME
        fprintf(fid,['--- CleanLine not performed: ' ME.message ' Make sure you specify the Line Noise Frequency in the *_eeg.json file. \n']);
        fprintf(fid0,'--- %s not preprocessed successfully.\n', id);
    end
     % 2. REMOVE M1M2 CHANNELS
    try
        M1_label = 'M1'; 
        M2_label = 'M2'; 
        M1_idx = find(strcmp({EEGtemp.chanlocs.labels}, M1_label));           
        if isempty(M1_idx)
            error('The channel corresponding to M1 was not found.');
        end
        channels_to_remove = [M1_idx]; 
        EEGtemp = pop_select(EEGtemp, 'nochannel', channels_to_remove);
        M2_idx = find(strcmp({EEGtemp.chanlocs.labels}, M2_label)); 
        if isempty(M2_idx)
            error('The channel corresponding to M2 was not found.');
        end
        channels_to_remove = [M2_idx]; 
        EEGtemp = pop_select(EEGtemp, 'nochannel', channels_to_remove);  
        EEGtemp = eeg_checkset(EEGtemp);
    %     EEG = eeg_chanedit(EEG, 'lookup','standard_1005.elc');
        fprintf(fid,'2. Remove M1 M2. \n');
    end
    % 3.Bandpass filtering
    %Set the frequency range for bandpass filtering.
    low_freq = 1;  % lower frequency limit
    high_freq = 47;   % upper limit frequency
    EEGtemp = pop_eegfiltnew(EEGtemp, low_freq, high_freq);
    fprintf(fid,'3. Bandpass filtering. \n');
    % 4. REMOVE BAD CHANNELS
    try
        [EEGtemp.urchanlocs] = deal(EEGtemp.chanlocs); % Keep original channels
        EEGtemp = pop_clean_rawdata(EEGtemp,'FlatlineCriterion', params.FlatLineCriterion,...
            'ChannelCriterion',params.ChannelCriterion,...
            'LineNoiseCriterion',params.LineNoiseCriterion,...
            'Highpass',params.HighPass,...
            'BurstCriterion','off',...
            'WindowCriterion','off',...
            'BurstRejection','off',...
            'Distance','Euclidian',...
            'WindowCriterionTolerances','off');
        if(isfield(EEGtemp.etc,'clean_channel_mask'))
            clean_channel_mask = EEGtemp.etc.clean_channel_mask;
        end
        fprintf(fid,'4. Bad channel removal performed. \n');
    catch ME
        fprintf(fid, ['--- Bad channel removal not performed: ' ME.message '. \n']);
        fprintf(fid0,'--- %s not preprocessed successfully.\n', id);
    end    
    % 5. REREFERENCE TO AVERAGE REFERENCE
    try
        if strcmp(params.AddRefChannel,'on')
            EEGtemp = pop_reref(EEGtemp,[],'interpchan',[],'refloc', EEGtemp.chaninfo.nodatchans);
        else
            EEGtemp = pop_reref(EEGtemp,[],'interpchan',[]);
        end
        fprintf(fid,'5. Re-referencing performed. \n');
    catch ME
        fprintf(fid,['--- Re-referencing not performed: ' ME.message '. \n']);
        fprintf(fid0,'--- %s not preprocessed successfully.\n', id);
    end
    % 6. REMOVE ARTIFACTS WITH ICA
    % 7. INTERPOLATE MISSING CHANNELS
    % 8. REMOVE BAD TIME SEGMENTS
    error_parallelization = false;
    try
        nRep = params.NICARepetitions;
        EEGtemp_clean = cell(1,nRep);
        parfor iRep =1:nRep
            % Steps 6. ICA, 7. Channel interpolation, and 8. bad time segments   
            EEGtemp_clean{iRep} = preprocessing_ICA(EEGtemp,params);
        end
               
        % Log warnings if some repetitions were not performed
        error_flags = cellfun(@isnumeric, EEGtemp_clean);
        error_flags = cell2mat(EEGtemp_clean(error_flags));
        if ~isempty(error_flags)
            fprintf(fid,'--- Warning: %d repetitions failed at ICA. \n', sum(error_flags ==6));
            fprintf(fid,'--- Warning: %d repetitions failed at channel interpolation. \n', sum(error_flags ==7));
            fprintf(fid,'--- Warning: %d repetitions failed at bad channel rejection. \n', sum(error_flags ==8));
        end
        
        % Custom function that selects the repetition closest to the
        EEGtemp = preprocessing_select_ICA_rep(EEGtemp_clean);
        
        % Log success
        fprintf(fid,'6. ICA performed. \n7. Channel interpolation performed. \n8. Bad interval removal performed. \n');             
    catch
        fprintf(fid,'--- No ICA repetition was successful with parallelization. Trying without parallelization... \n');
        error_parallelization = true;
    end
    
    % Try steps 6, 7,8 without parallelization
    if (error_parallelization)
        try
            EEGtemp_clean = cell(1,nRep);
            for iRep =1:nRep
                % Steps 6, 7 and 8
                EEGtemp_clean{iRep} = preprocessing_ICA(EEGtemp,params);
            end
            
            % Log warnings if some repetitions were not performed
            error_flags = cellfun(@isnumeric, EEGtemp_clean);
            error_flags = cell2mat(EEGtemp_clean(error_flags));
            if ~isempty(error_flags)
                fprintf(fid,'--- Warning: %d repetitions failed at ICA. \n', sum(error_flags == 6));
                fprintf(fid,'--- Warning: %d repetitions failed at channel interpolation. \n', sum(error_flags == 7));
                fprintf(fid,'--- Warning: %d repetitions failed at bad channel rejection. \n', sum(error_flags == 8));
                fprintf(fid0,'--- %s Warning. Not all ICA repetitions were successful.\n', id);
            end
            
            % Custom function that selects the repetition closest to the
            % 'average' bad time segments mask
            EEGtemp = preprocessing_select_ICA_rep(EEGtemp_clean);
            
            % Log success
            fprintf(fid,'6. ICA performed. \n7. Channel interpolation performed. \n8. Bad interval removal performed. \n');        

        catch
            fprintf(fid,['--- No ICA repetition was successful without parallelization. \n Exclude recording.\n']);
            to_delete{end +1} = EEGtemp.filename;
            fprintf(fid0,'--- %s not preprocessed successfully.\n', id);
            continue; % to next recording
        end
    end    
    if params.PreprocEventData
        if(~isempty(params.EpochLength))
            try
                EEGtemp = pop_epoch(EEGtemp, {'1001','1002','1004','1007'},params.EventBounds,'epochinfo','yes'); % Assuming event.type contains the marker you want to use
            catch ME
                warning('During epoching in dataset %s:this dataset is empty.',  EEGtemp.filename);
                disp(ME.message);
                continue; 
            end

%             EEGtemp = pop_epoch(EEGtemp, {'1001','1002','1004','1007'},params.EventBounds,'epochinfo','yes'); % Assuming event.type contains the marker you want to use
            EEGtemp.original_events = EEGtemp.event; 
            if EEGtemp.trials == 0
                to_delete{end + 1} = EEGtemp.filename;
            end
            fprintf(fid,'9. Segmentation into epochs performed. \n');
        end
    end  
     i = 1; 
     while i <= length(EEGtemp.epoch)   
         p= length(EEGtemp.epoch);
         if i<=p           
             event_types = {EEGtemp.epoch(i).eventtype};
             if iscellstr(event_types)
                 fprintf('event_types is a cell array of character type, with a size of %d.\n', numel(event_types));
                 cell_size = numel(event_types );
             else               
                 fprintf('The event_types is not a character cell array, with a size of %d.\n', numel(event_types{1}));
                 cell_size = numel(event_types{1} );
             end
             %         cell_size = numel(epochs(i).eventtype );
             if  cell_size>1
                 event_type_str1 =EEGtemp.epoch(i).eventtype{1};  
                 event_type_str2 =EEGtemp.epoch(i).eventtype{2};
                 if str2double(event_type_str1)~=str2double(event_type_str2)
                     fprintf('Removing epoch %d with multiple different events.\n', i);    
                     fprintf(fid,'---Removing epoch %d with multiple different events.\n', i);
                     EEGtemp.epoch(i) = [];
                     EEGtemp.data(:,:,i)=[];
                     EEGtemp.trials=EEGtemp.trials-1;
                     EEGtemp.event(i)=[];                      
                     if i<=length(EEGtemp.event)
                         if EEGtemp.event(i).epoch~=i
                             EEGtemp.event(i).epoch=i;
                         end
                         EEGtemp.event(i)=[];                         
                     end
                     if cell_size==3
                        EEGtemp.event(i)=[];
                     end
                     i = i - 1;
                 end
                 if str2double(event_type_str1)==str2double(event_type_str2)                  
                     fprintf('Removing second event from epoch %d with multiple identical events.\n', i);  
                     fprintf(fid,'---Removing second event from epoch %d with multiple identical events.\n', i);
                     field_names = fieldnames( EEGtemp.epoch(i));
                     for j = 1:length(field_names)
                         field_val =  EEGtemp.epoch(i).(field_names{j});
                         if numel(field_val) > 1
                             if iscell(field_val)
                                 EEGtemp.epoch(i).(field_names{j}) = field_val{1};
                             else
%                                  EEGtemp.epoch(i).(field_names{j}) = field_val(1);
                                 EEGtemp.epoch(i).(field_names{j}) = field_val(1);
                             end
                         end
                     end
                     if  i<length(EEGtemp.event)
                       EEGtemp.event(i+1)=[];                       
                     end
                     if EEGtemp.event(i).epoch~=i
                         EEGtemp.event(i).epoch=i;
                     end
                     i = i - 1;
                 end
             end
             if i~=0
                 if EEGtemp.event(i).epoch~=i
                     EEGtemp.event(i).epoch=i;
                 end
             end
             i = i + 1;  
         end
     end
     disp('After removing epochs with multiple different events');
 
    % Save datafile, clear it from memory, and store it in the ALLEEG structure
    EEGtemp.comments = 'preprocessed';
    if exist('clean_channel_mask','var'), EEGtemp.etc.clean_channel_mask = clean_channel_mask; end; clear 'clean_channel_mask';
    EEGtemp = pop_saveset(EEGtemp, 'savemode', 'resave');
    EEGtemp.data = 'in set file';
    ALLEEG = eeg_store(ALLEEG, EEGtemp, iRec);
    STUDY = pop_savestudy(STUDY, ALLEEG, 'filename', params.StudyName, 'filepath', params.PreprocessedDataPath);
    
    fprintf(fid,'Preprocessed data saved to disk');
    fclose(fid);
    t2 = toc(t1);
    fprintf(fid0,'%s preprocessed successfully. It took %.2f seconds.\n', id, t2);
end
fclose(fid0);

fNames = {ALLEEG.filename};
javaStrings = javaArray('java.lang.String', numel(fNames));
for i = 1:numel(fNames)
    javaStrings(i) = java.lang.String(fNames{i});
end
toDeleteJavaStrings = javaArray('java.lang.String', numel(to_delete));
for i = 1:numel(to_delete)
    toDeleteJavaStrings(i) = java.lang.String(to_delete{i});
end
toDeleteLogical = false(size(fNames));
for i = 1:numel(fNames)
    for j = 1:numel(toDeleteJavaStrings)
        if javaStrings(i).equals(toDeleteJavaStrings(j))
            toDeleteLogical(i) = true;
            break;
        end
    end
end
mask=toDeleteLogical;

ALLEEG(mask) = [];
STUDY.datasetinfo(mask) = [];
s = split(to_delete,{'_'});
s = unique(s(:,:,1));

javaSubjects = javaArray('java.lang.String', numel(STUDY.subject));
for i = 1:numel(STUDY.subject)
    javaSubjects(i) = java.lang.String(STUDY.subject{i});
end
javaS = javaArray('java.lang.String', numel(s));
for i = 1:numel(s)
    javaS(i) = java.lang.String(s{i});
end
maskl = false(size(STUDY.subject));
for i = 1:numel(STUDY.subject)
    for j = 1:numel(javaS)
        if javaSubjects(i).equals(javaS(j))
            maskl(i) = true;
            break; 
        end
    end
end
mask=maskl;

STUDY.subject(mask) = [];
% Save study
STUDY = pop_savestudy(STUDY, ALLEEG, 'filename', [params.StudyName '-clean'], 'filepath', params.PreprocessedDataPath);
% PLOTTING PREPROCESSING
% Visualization of detected bad channels. If you set 'FuseChanRej' on, the union of bad channels in all tasks is rejected!
single_path=2;
plot_badchannels(params,ALLEEG,single_path);
% Visualization of rejected ICs
plot_ICs(params,ALLEEG);
% Visualization of rej
plot_badtimesegments(params,ALLEEG,single_path);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          lot_badtimesegments(params,ALLEEG);
% Visualize number of clean epochs per recording
if params.EpochLength~= 0
    plot_epochs(params,ALLEEG);
end
%%
%Sub09
single_path=1;
plot_badchannels(params, ALLEEG(48:53),single_path);
plot_badtimesegments(params, ALLEEG(48:53),single_path);
%set2mat
main_folder_path =params.PreprocessedDataPath;
PreprocessedMatDataPath = fullfile(main_folder_path, 'feature');
if ~exist(PreprocessedMatDataPath, 'dir')
    mkdir(PreprocessedMatDataPath);
end
PreprocessedMatDataPath = fullfile(PreprocessedMatDataPath, 'mat');
if ~exist(PreprocessedMatDataPath, 'dir')
    mkdir(PreprocessedMatDataPath);
end
process_mat_data( ALLEEG,main_folder_path,PreprocessedMatDataPath);
