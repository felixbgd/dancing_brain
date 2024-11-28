%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Imaging the dancing brain" paper - Bigand et al. (2024) %
%           Preprocessing automatic pipeline               %
%%%%%%%%% Félix Bigand, NPA Lab, 2022-2024 - Rome %%%%%%%%%%

%% Import libraries, initialise fieldtrip

clc
clear all
close all

addpath(genpath('.\functions\'));       % personal functions provided in github repo
addpath(genpath('.\mat\'));   % folder with EEG data (.mat), to be adapted

% toolboxes to download separately: fieldtrip, EEGlab clean_rawdata
addpath('.\fieldtrip-20221014\');
addpath(genpath(['.\fieldtrip-20221014\external\eeglab\']));
addpath(genpath(['.\clean_rawdata2.7\']));
ft_defaults;

%% Load data (already stored as .mat files)

dir_inputEEG = '.\mat\raw\';     % ADAPT WITH YOUR PATH, being where the mat files you want to clean are
output_dir_EEG = '.\mat\cleaned_8Hz_100fps\';  % ALSO ADAPT based on your cleaning/preprocessing

disp('LOADING DATA...')
fileNames = {dir([dir_inputEEG '*.mat']).name}; 

iParticipant = 1;

for p = 1:80       
    disp(['Participant ' num2str(p)])
    
    if p < 10  numParticipant = ['0' num2str(p)];
    else numParticipant = num2str(p);
    end

    load([dir_inputEEG fileNames{p}]);

    cfg = data.cfg;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%           PREPROCESSING          %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%-------     (0) SOME SIMPLE STUFF      -------%%
    
    % All the triggers (then also including info about the songs)
    statusevent = data.statusevent;
    
    % Remove unused channels
    data = EEG_removeChannels(data, {'EXG7','EXG8','GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp'});
    
    %%-------    (1) BP FILTER [1-8Hz]       -------%%
    maxfreq = 8;
    data_bp = EEG_bandPass(data,1,maxfreq,3,'yesNotch');

    %%-------          (2) DOWNSAMPLE         -------%%
    cfg = [];
    fps=100;
    cfg.resamplefs = fps;
    [data_bp_rs] = ft_resampledata(cfg, data_bp);

    % Create distinct EEG data struct and external electrodes data struct
    cfg = [];
    cfg.channel = {'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6'};
    data_bp_rs_aux = ft_selectdata(cfg,data_bp_rs); % saves the aux for later
    data_bp_rs_eeg = EEG_removeChannels(data_bp_rs,{'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6'});
    
    %%-------      (3) REMOVE BAD CHANNELS      ------%%
    % Flat -- high-freq noise -- low-correlation with neighbours (eeglab toolbox)
    data_bp_rs_eeg_all = data_bp_rs_eeg;        % saves data with all EEG channels for later interpolation
    steps_to_do = {'flat','corr-noise'};
    criteria = {[5] [0.8,4]};
    
    bad_channels_flatNoisy = RFT_detect_FlatCorrNoise_electrodes(data_bp_rs_eeg,'eeglab_template_biosemi64.mat',steps_to_do,criteria);
    bad_channels_flat = bad_channels_flatNoisy.flat;     % you could also concatenate bad channels of both flat and noise check
    bad_channels_corrNoise = bad_channels_flatNoisy.corrNoise;
    data_bp_rs_eeg = EEG_removeChannels(data_bp_rs_eeg , bad_channels_flat);     % remove them
    data_bp_rs_eeg = EEG_removeChannels(data_bp_rs_eeg , bad_channels_corrNoise);     % remove them

    % Noisy electrodes (our function)
    [ bad_channels_catchNoise ] = EEG_CatchNoisyElectrodes( data_bp_rs_eeg, 'all', 3, 'recursive' );
    data_bp_rs_eeg = EEG_removeChannels(data_bp_rs_eeg,bad_channels_catchNoise);
    
    % Concatenate bad channels from every specific issue
    bad_channels = [bad_channels_flat bad_channels_corrNoise bad_channels_catchNoise];
    bad_channels_num(iParticipant) = length(bad_channels);
    bad_channels_store{p} = bad_channels;

    %%-------      (4) RE-REFERENCE (CAR)       -------%%
    % Re-ref external electrodes with same average than scalp
    cfg                 = []; cfg.layout          = 'lay_biosemi64';
    data_bp_rs_all = ft_appenddata(cfg, data_bp_rs_eeg, data_bp_rs_aux);
    ch_for_reref = data_bp_rs_eeg.label(~ismember(data_bp_rs_eeg.label, bad_channels)); %% Get only the remaining 'good' channels
    if p > 10    % to remove for saras
        cfg                      = [];
        cfg.channel              = 'all';
        cfg.reref                = 'yes';
        cfg.refchannel           = ch_for_reref; %% Reref only using EXT channels
        data_bp_rs_all         = ft_preprocessing(cfg,data_bp_rs_all);
    end
    cfg = [];
    cfg.channel = {'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6'};
    data_bp_rs_aux = ft_selectdata(cfg,data_bp_rs_all); % saves the aux for later

    % Take only eye, cheek, neck electrodes 
    cfg = [];
    cfg.channel = {'EXG1','EXG2'};
    data_bp_rs_eye = ft_selectdata(cfg,data_bp_rs_aux);

    cfg = [];
    cfg.channel = {'EXG3','EXG4'};
    data_bp_rs_cheek = ft_selectdata(cfg,data_bp_rs_aux);

    cfg = [];
    cfg.channel = {'EXG5','EXG6'};
    data_bp_rs_neck = ft_selectdata(cfg,data_bp_rs_aux);

    % Re-ref EEG electrodes (with average of EEG electrodes)
    ch_for_reref = data_bp_rs_eeg.label(~ismember(data_bp_rs_eeg.label, bad_channels)); %% Get only the remaining 'good' channels
    cfg                      = [];
    cfg.channel              = {'EEG'};
    cfg.reref                = 'yes';
    cfg.refchannel           = ch_for_reref;        %% Reref only using good channels
    data_bp_rs_eeg           = ft_preprocessing(cfg,data_bp_rs_eeg);


    %%-------           (5) RUN ASR            -------%%
    ref_maxbadchannels=0.075;
    THRESH_ASR = 5;
    % On EEG
    [data_bp_rs_asr_eeg , asr_ref_section, asr_eigen_ref_time, asr_eigen_ref_topo]  = RFT_clean_asr_combined_trials_ftstruct(data_bp_rs_eeg,'eeglab_template_biosemi64.mat',THRESH_ASR,[],ref_maxbadchannels);
    
    % On EMG/EOG
    [data_bp_rs_asr_eye , asr_ref_section, asr_eigen_ref_time, asr_eigen_ref_topo]  = RFT_clean_asr_combined_trials_ftstruct(data_bp_rs_eye,'eeglab_template_biosemi64.mat',THRESH_ASR,[],ref_maxbadchannels);
    [data_bp_rs_asr_cheek , asr_ref_section, asr_eigen_ref_time, asr_eigen_ref_topo]  = RFT_clean_asr_combined_trials_ftstruct(data_bp_rs_cheek,'eeglab_template_biosemi64.mat',THRESH_ASR,[],ref_maxbadchannels);
    [data_bp_rs_asr_neck , asr_ref_section, asr_eigen_ref_time, asr_eigen_ref_topo]  = RFT_clean_asr_combined_trials_ftstruct(data_bp_rs_neck,'eeglab_template_biosemi64.mat',THRESH_ASR,[],ref_maxbadchannels);
    data_bp_rs_asr_aux = ft_appenddata([], data_bp_rs_asr_eye, data_bp_rs_asr_cheek, data_bp_rs_asr_neck);


    %%-------         (6) RUN ICA AUTO         -------%%
    % Auto IClabel
    numComp = 30;
    [ data_bp_rs_asr_ica_eeg, rejected_comps ] = RFT_IClabel(data_bp_rs_asr_eeg , 'eeglab_template_biosemi64.mat', numComp, [0 0;0 0; 0 1; 0 0; 0 0; 0 0; 0 0],data_bp_rs_aux);
    close;

    %%-------  (7) INTERPOLATE BAD CHANNELS  -------%%
    if ~isempty(bad_channels)
        disp('Interpolating channels...');
        cfg                 = [];
        cfg.layout          = 'lay_biosemi64';
        cfg.method          = 'distance'; % for prepare_neigh
        cfg.neighbourdist   = 0.2;         % results in avg 5 channels
        cfg.neighbours      = ft_prepare_neighbours(cfg, data_bp_rs_eeg_all);
        cfg.badchannel      = bad_channels; %data.label(ChanInterpol);
        cfg.method          = 'nearest';     
        cfg.elec            = ft_read_sens('standard_1020.elc'); % Trinh trick

        data_bp_rs_asr_ica_interp_eeg  = ft_channelrepair(cfg, data_bp_rs_asr_ica_eeg);           
    else
        data_bp_rs_asr_ica_interp_eeg = data_bp_rs_asr_ica_eeg;
        disp('No channels interpolated');
    end

    %%% Recombine EEG+EMG/EOG for Raw, +reref, +ASR, +ICA, +interp %%%
    data_bp_rs_PLOT = data_bp_rs;
    data_bp_rs_eeg_PLOT = ft_appenddata(cfg, data_bp_rs_eeg, data_bp_rs_asr_aux);
    data_bp_rs_asr_PLOT = ft_appenddata(cfg, data_bp_rs_asr_eeg, data_bp_rs_asr_aux);
    data_bp_rs_asr_ICA_PLOT = ft_appenddata(cfg, data_bp_rs_asr_ica_eeg, data_bp_rs_asr_aux);
    data_bp_rs_asr_ica_interp_all = ft_appenddata(cfg, data_bp_rs_asr_ica_interp_eeg, data_bp_rs_asr_aux); % <- that's the one we will use in all further analyses
    
    %%% (Optional) plot these different EEG versions, to visualize the different prepro steps %%%
    viz_raw = 0;

    % Visualisation parameters
    cfgint                 = [];
    cfgint.layout          = 'lay_biosemi64';
    cfgint.ylim            = [-30 30];
    cfgint.blocksize       = 20;
    
    % Raw
    if viz_raw==1
        ft_databrowser(cfgint,data_bp_rs_PLOT);
        saveas(gcf,['.\tests\participant_' num2str(p) '_1-RAW.jpg']);
        close;
    end

    % Raw reref
    if viz_raw==1
        ft_databrowser(cfgint,data_bp_rs_eeg_PLOT);
        saveas(gcf,['.\tests\participant_' num2str(p) '_2-RAW-reref.jpg']);
        close;
    end

    % ASR
    if viz_raw==1
        ft_databrowser(cfgint,data_bp_rs_asr_PLOT);
        saveas(gcf,['.\tests\participant_' num2str(p) '_3-ASR.jpg']);
        close;
    end
    
    % ICA
    if viz_raw==1
        ft_databrowser(cfgint,data_bp_rs_asr_ICA_PLOT);
        saveas(gcf,['.\tests\participant_' num2str(p) '_4-ICA.jpg']);
        close;
    end
    
    % ICA interp
    if viz_raw==1
        ft_databrowser(cfgint,data_bp_rs_asr_ica_interp_all);
        saveas(gcf,['.\tests\participant_' num2str(p) '_5-ICA-interp.jpg']);
        close;
    end

    %%%% SAVE CLEANED DATA OF THE PARTICIPANT %%%%
    data_bp_rs_asr_ica_interp_all.statusevent = statusevent;
    save([output_dir_EEG 'subj_' numParticipant],'data_bp_rs_asr_ica_interp_all');

    iParticipant = iParticipant +1;

end