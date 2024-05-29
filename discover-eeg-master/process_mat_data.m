function process_mat_data(ALLEEG,main_folder_path,PreprocessedDataPath)
%%   ���ù˼�params��תepoch ��label mat��ʽ
% clear all; close all;
% ����һ���յ� ALLEEG �ṹ
% ALLEEG = [];
% % �ݹ�����ļ��������� .set �ļ�
% ALLEEG = load_all_sets(main_folder_path, ALLEEG);
% ���� ALLEEG �ṹ�а��������м��ص� .set �ļ��� EEG ����
for iRec=1:length(ALLEEG)
  EEGtemp = eeg_checkset(ALLEEG(iRec),'loaddata');
  for i =1:length(EEGtemp.epoch)
    event_types = {EEGtemp.epoch(i).eventtype};
    if EEGtemp.event(i).epoch~=i
        EEGtemp.event(i).epoch=i;
    end           
  end 
    label= cellfun(@str2num,{EEGtemp.event.type});
    % ����Ҫ������ļ���
    id = regexp(ALLEEG(iRec).filename,'.*(?=(_eeg.set))','match');
    id = id{:};
    short_id = id(1:6);  % ��ȡ ID ��ǰ 6 ���ַ�
    % ������ short_id Ϊ�����ļ��У���������ڵĻ���
    folder_path = fullfile(PreprocessedDataPath, short_id);
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end
    newname = [id, '_', 'label'];
    filename1 = fullfile(folder_path,  [newname,'.mat']);
    save(filename1, 'label');    
    filename = [id,'.mat'];
    full_path = fullfile(folder_path, filename);
    save(full_path, 'EEGtemp');
end
end