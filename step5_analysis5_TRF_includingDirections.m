%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Imaging the dancing brain" paper - Bigand et al. (2024) %
%    TRF analysis including self/other mov. directions     %
%%%%%%%%% Félix Bigand, NPA Lab, 2022-2024 - Rome %%%%%%%%%%

%% Import libraries, initialise fieldtrip

clc
clear all
close all


addpath(genpath('.\functions\'));       % personal functions provided in github repo
addpath(genpath('.\mat\'));   % folder with EEG data (.mat), to be adapted

% toolboxes to download separately: fieldtrip, mTRF
addpath('.\fieldtrip-20221014\');
addpath(genpath(['.\fieldtrip-20221014\external\eeglab\']));
addpath('.\fieldtrip-20221014\external\brewermap\')     % for colormaps
addpath(['.\mtrf\']); 
ft_defaults;


%% Set parameters, folders, paths etc

output_dir ='.\results_step5_TRF_includingDirections\';
if ~isfolder(output_dir)  mkdir(output_dir);   end

NB_TRIALS=32;
NB_SONGS=8;
fps=100;

stim_names = {'song1','song2','song3','song4','song5','song6','song7','song8'};

% Compute length of songs/temporal structure
song_bpms = [111.03,116.07,118.23,118.95,120.46,125.93,128.27,129.06];
periodbeat = (60./song_bpms);
musParts_beats = [1,17,33,49,65,81];       % start drums , bass, harmony, voice, end
beats_tFrames = [];         musParts_tFrames_s = [];
for i=1:NB_SONGS
    beats_tFrames(i,:) = linspace(0,80*periodbeat(i),81); % Because 80 beats for each song
    musParts_tFrames_s(i,:) = beats_tFrames(i,musParts_beats);
end
musParts_tFrames = round(musParts_tFrames_s * fps);

%% Load data (already stored as .mat files)

disp('LOADING DATA...')
dir_inputEEG = '.\mat\cleaned_8Hz_100fps\';     % ADAPT WITH YOUR PATH
fileNames = {dir([dir_inputEEG '*.mat']).name};

load(['.\\songNums_allTrials_allSubj.mat']);
load(['.\\conds_allTrials_allSubj.mat']);

% Directories for predictor/stimuli to predict the EEG
% Audio waveforms --> music model
stim_style = 'all';
predictor_audio = 'StimAcoustics';
allpred_audio = load(['.\\' predictor_audio]);

% Principal Movements (PMs)) --> for self, other models
predictor_motion_pm = 'StimMotion_pms';
allpred_motion_pm = load(['.\\' predictor_motion_pm]);

% Audio Lag  (s)
audioLag = 0.0535;

% TRF model parameters
minlag = -250;
maxlag = 300  ;
lambdas = [0 10.^(-4:8)]; 

% Loop for training subject-specific TRF models
Stim_trialsCond_audio=[]; Stim_trialsCond_motion=[];  EEG_trialsCond=[];
iParticipant = 1;
for p = [1:40 43:44 47:50 55:56 59:80]          % only participants with OK mocap data
    disp(['Participant ' num2str(p)])
    
    if p < 10  numParticipant = ['0' num2str(p)];
    else numParticipant = num2str(p);
    end
    
    % Load EEG data of the participant
    load([dir_inputEEG fileNames{p}]);
    tr = data_bp_rs_asr_ica_interp_all.trialinfo;  statusevent = data_bp_rs_asr_ica_interp_all.statusevent;
    data_bp_rs_asr_ica_interp_all = ft_preprocessing([],data_bp_rs_asr_ica_interp_all);

    % Retrieve EEG scalp data
    data_bp_rs_asr_ICA_interp           = Giac_removeChannels(data_bp_rs_asr_ica_interp_all,{'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6'});
    [labels_sorted , idx_labels_sorted] = sort(data_bp_rs_asr_ICA_interp.label);

    %%% Define pred/stim and EEG/EMG/EOG for each trial
    % Find trials corresponding to each condition
    for cond=1:4  tr_cond{cond} = find(tr==cond)';      end
    
    % Exception message for one trial (the last one) of one dyad absent
    if ismember(p,[73,74])
        tr_missing=32;
        tr_nb = NB_TRIALS - 1;
    else
        tr_nb = NB_TRIALS;
    end
    
    % Loop over
    iTrial = 1;
    for ii = 1:tr_nb       % For each trial that you wanna analyse
        disp(['TRIAL ' num2str(ii)]);

        % Compute first onset frame
        prestim     = round((1+audioLag) * fps);   % Because there was 1s of prestim in our EEG first definition (prepro)
        idxStartTr  = (ii-1)*4 + 1;       % Because 4 triggers per trial (startTrial, startSong, stopSong, stopTrial)
        startSong_s = (statusevent(idxStartTr+1).sample-statusevent(idxStartTr).sample)/1024;         % first music onset (startSong) in s (there was silent/still period between startTrial and startSong)
        startSong   = round(startSong_s * fps);
        
        % Check what songs were played for both subjects
        if mod(p,2)==1 iOtherParticipant = p+1; end
        if mod(p,2)==0 iOtherParticipant = p-1; end
        songNum_self  = songNums_allTrials_allSubj(p,ii);
        songNum_other = songNums_allTrials_allSubj(iOtherParticipant,ii);
        
        lensong       = min(musParts_tFrames([songNum_self,songNum_other] , end));  % Trim stop to the common time they listened to music in the trial
        cutOnset      = 2*round(periodbeat(songNum_self) * fps);  % Trim start after 2 beats of music (to avoid confounding effect of first "startle" responses at beginning, see TRF litterature)
        
        % Define trials of, and trim, EEG data
        cfg                    = [];
        cfg.trials             = ii; 
        EEG_of_one_trial       = ft_selectdata(cfg,data_bp_rs_asr_ICA_interp);

        startSong    = startSong + prestim + EEG_of_one_trial.sampleinfo(1);
        cfg          = [];
        cfg.trl(:,1) = startSong + cutOnset;        % start 
        cfg.trl(:,2) = startSong + lensong;         % stop 
        cfg.trl(:,3) = 0;
        
        % Trim/cut EEG
        ft_EEG_of_one_trial_cut      = ft_redefinetrial(cfg,EEG_of_one_trial);
        EEG_of_one_trial_cut         = ft_EEG_of_one_trial_cut.trial{1,1};

        % Reorder the channels labels for EEG data (this is necessary
        % because of interpolation step in prepro)
        EEG_of_one_trial_cut = EEG_of_one_trial_cut(idx_labels_sorted,:); 
        eeg_label_reorder = {ft_EEG_of_one_trial_cut.label{idx_labels_sorted}};

        %%% Store EEG per trial
        EEG_trialsCond{iTrial} = EEG_of_one_trial_cut;
        
        %%% Store predictors/stimuli per trial %%%
        %%% Audio/music %%%
        pred_idx             = find(strcmp(allpred_audio.stim_names,['song' num2str(songNum_self) '_' stim_style '.wav']));
        Stim_one_trial_audio = [allpred_audio.specflux_avg{pred_idx}']; sz_feat=[1]; feat_names={'T specflux'};
        Stim_one_trial_audio = Stim_one_trial_audio(:,cutOnset:lensong);  % trim accordingly

        %%% Self %%%
        iSelfParticipant = p; pm = 10;
        Stim_one_trial_motion_self_signed = cell2mat(allpred_motion_pm.motion_pms{iSelfParticipant,ii});
        Stim_one_trial_motion_self_signed = Stim_one_trial_motion_self_signed(:,cutOnset:lensong);  % trim accordingly
        Stim_one_trial_motion_self_signed = Stim_one_trial_motion_self_signed(pm,:);
        Stim_one_trial_motion_self        = abs(Stim_one_trial_motion_self_signed);
        Stim_one_trial_motion_self_sign = sign(Stim_one_trial_motion_self_signed);

        %%% Other %%%
        Stim_one_trial_motion_other_signed = cell2mat(allpred_motion_pm.motion_pms{iOtherParticipant,ii}); 
        Stim_one_trial_motion_other_signed = Stim_one_trial_motion_other_signed(:,cutOnset:lensong);  % trim accordingly
        Stim_one_trial_motion_other_signed = Stim_one_trial_motion_other_signed(pm,:);
        Stim_one_trial_motion_other        = abs(Stim_one_trial_motion_other_signed);
        Stim_one_trial_motion_other_sign   = sign(Stim_one_trial_motion_other_signed);

        %%% Social coordination (IMS) %%%
        Stim_one_trial_IMS = sign(Stim_one_trial_motion_self_signed) .* sign(Stim_one_trial_motion_other_signed);

        Stim_trialsCond_audio{iTrial} = Stim_one_trial_audio;
        Stim_trialsCond_motion_self{iTrial} = Stim_one_trial_motion_self;
        Stim_trialsCond_motion_other{iTrial} = Stim_one_trial_motion_other;
        Stim_trialsCond_motion_self_sign{iTrial} = Stim_one_trial_motion_self_sign;
        Stim_trialsCond_motion_other_sign{iTrial} = Stim_one_trial_motion_other_sign;
        Stim_trialsCond_IMS{iTrial} = Stim_one_trial_IMS;
        
        iTrial = iTrial+1;
    end
    
    % Exception message for one trial (the last one) of one dyad absent
    if ismember(p,[73,74])
        Stim_trialsCond_audio{tr_missing} = [];
        Stim_trialsCond_motion_self{tr_missing} = [];
        Stim_trialsCond_motion_other{tr_missing} = [];
        Stim_trialsCond_motion_self_sign{tr_missing} = [];
        Stim_trialsCond_motion_other_sign{tr_missing} = [];
        Stim_trialsCond_IMS{tr_missing} = [];
        EEG_trialsCond{tr_missing} = [];
    end


    %%% Normalisation of EEG and stimuli/predictors (see Crosse et al., 2021) %%%%
    %%% Normalize them based on "global std" of the participant               %%%%
    % Compute "global std" of predictors/stimuli
    Stim_allTrials_concat_audio = [Stim_trialsCond_audio{:}];
    Stim_audio_global_std = sqrt(mean(var(Stim_allTrials_concat_audio,[],2)));

    Stim_allTrials_concat_motion_self = [Stim_trialsCond_motion_self{:}];
    Stim_motion_self_global_std = std(Stim_allTrials_concat_motion_self,[],2);

    Stim_allTrials_concat_motion_other = [Stim_trialsCond_motion_other{:}];
    Stim_motion_other_global_std = std(Stim_allTrials_concat_motion_other,[],2);

    Stim_allTrials_concat_motion_self_sign = [Stim_trialsCond_motion_self_sign{:}];
    Stim_motion_self_sign_global_std = std(Stim_allTrials_concat_motion_self_sign,[],2);

    Stim_allTrials_concat_motion_other_sign = [Stim_trialsCond_motion_other_sign{:}];
    Stim_motion_other_sign_global_std = std(Stim_allTrials_concat_motion_other_sign,[],2);

    Stim_allTrials_concat_IMS = [Stim_trialsCond_IMS{:}];
    Stim_motion_IMS_global_std = std(Stim_allTrials_concat_IMS,[],2);


    % Compute "global std" of EEG
    EEG_allTrials_concat = [EEG_trialsCond{:}];
    EEG_global_std = sqrt(mean(var(EEG_allTrials_concat,[],2)));

    % Normalize, and create stim/pred cell arrays for full and reduced models
    for iTrial=1:tr_nb
        Stim_trialsCond_audio{iTrial} = (Stim_trialsCond_audio{iTrial} ./ Stim_audio_global_std)';
        Stim_trialsCond_motion_self{iTrial} = (Stim_trialsCond_motion_self{iTrial} ./ Stim_motion_self_global_std)';  
        Stim_trialsCond_motion_other{iTrial} = (Stim_trialsCond_motion_other{iTrial} ./ Stim_motion_other_global_std)';  
        Stim_trialsCond_motion_self_sign{iTrial} = (Stim_trialsCond_motion_self_sign{iTrial} ./ Stim_motion_self_sign_global_std)';   
        Stim_trialsCond_motion_other_sign{iTrial} = (Stim_trialsCond_motion_other_sign{iTrial} ./ Stim_motion_other_sign_global_std)'; 
        Stim_trialsCond_IMS{iTrial} = (Stim_trialsCond_IMS{iTrial} ./ Stim_motion_IMS_global_std)'; 
        
        EEG_trialsCond{iTrial} = (EEG_trialsCond{iTrial} ./ EEG_global_std)';

        % Create full and reduced models
        Stim_trialsCond_full{iTrial} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial},Stim_trialsCond_motion_self_sign{iTrial},Stim_trialsCond_motion_other{iTrial},Stim_trialsCond_motion_other_sign{iTrial}, Stim_trialsCond_IMS{iTrial}];
        Stim_trialsCond_minus_audio{iTrial} = [Stim_trialsCond_motion_self{iTrial},Stim_trialsCond_motion_self_sign{iTrial},Stim_trialsCond_motion_other{iTrial},Stim_trialsCond_motion_other_sign{iTrial}, Stim_trialsCond_IMS{iTrial}];
        Stim_trialsCond_minus_self{iTrial} = [Stim_trialsCond_audio{iTrial} ,Stim_trialsCond_motion_self_sign{iTrial},Stim_trialsCond_motion_other{iTrial},Stim_trialsCond_motion_other_sign{iTrial}, Stim_trialsCond_IMS{iTrial}];
        Stim_trialsCond_minus_other{iTrial} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial},Stim_trialsCond_motion_self_sign{iTrial},Stim_trialsCond_motion_other_sign{iTrial}, Stim_trialsCond_IMS{iTrial}];
        Stim_trialsCond_minus_self_sign{iTrial} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial},Stim_trialsCond_motion_other{iTrial},Stim_trialsCond_motion_other_sign{iTrial}, Stim_trialsCond_IMS{iTrial}];
        Stim_trialsCond_minus_other_sign{iTrial} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial},Stim_trialsCond_motion_self_sign{iTrial},Stim_trialsCond_motion_other{iTrial}, Stim_trialsCond_IMS{iTrial}];
        Stim_trialsCond_minus_IMS{iTrial} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial},Stim_trialsCond_motion_self_sign{iTrial}, Stim_trialsCond_motion_other{iTrial},Stim_trialsCond_motion_other_sign{iTrial}];
        
    end
    
    % Store the pred/stim and EEG per participant and condition --> for the
    % generic approach when we test generic TRF on each participant (later)
    for cond=1:4
        trials_of_cond = tr_cond{cond};
        % Exception message for one trial (the last one) of one dyad absent
        if ismember(p,[73,74]) && length(find(trials_of_cond==32))>0    
            cond_missing = cond;  tr_missing_in_cond=find(tr_cond{cond}==32);
            trials_of_cond(tr_missing_in_cond)=[]; 
        end

        Stim_part_full{cond,iParticipant}             = {Stim_trialsCond_full{trials_of_cond}};
        Stim_part_minus_audio{cond,iParticipant}      = {Stim_trialsCond_minus_audio{trials_of_cond}};
        Stim_part_minus_self{cond,iParticipant}       = {Stim_trialsCond_minus_self{trials_of_cond}};
        Stim_part_minus_other{cond,iParticipant}      = {Stim_trialsCond_minus_other{trials_of_cond}};
        Stim_part_minus_self_sign{cond,iParticipant}  = {Stim_trialsCond_minus_self_sign{trials_of_cond}};
        Stim_part_minus_other_sign{cond,iParticipant} = {Stim_trialsCond_minus_other_sign{trials_of_cond}};
        Stim_part_minus_IMS{cond,iParticipant}        = {Stim_trialsCond_minus_IMS{trials_of_cond}};
        EEG_part{cond,iParticipant}                   = {EEG_trialsCond{trials_of_cond}};
    end
   
    %%%%%%%%%%%%%%%%%
    %%%% RUN TRF %%%%
    %%%%%%%%%%%%%%%%%
    nb_models = 7;
    nb_feat = [6,5,5,5,5,5,5];
    for feat=1:nb_models
        if feat==1 Stim = Stim_trialsCond_full; end
        if feat==2 Stim = Stim_trialsCond_minus_audio; end
        if feat==3 Stim = Stim_trialsCond_minus_self; end
        if feat==4 Stim = Stim_trialsCond_minus_other; end
        if feat==5 Stim = Stim_trialsCond_minus_IMS; end
        if feat==6 Stim = Stim_trialsCond_minus_self_sign; end
        if feat==7 Stim = Stim_trialsCond_minus_other_sign; end
       
        
        %%% Train one TRF per condition and for full+reduced models
        cv_avgtr=[]; cv_avgtrch=[]; opt_idx=[];  opt_lmb=[]; mdl_test=[]; st_test=[];
        for cond=1:4
            trials_of_cond = tr_cond{cond};
            
            % Exception message for one trial (the last one) of one dyad absent
            if ismember(p,[73,74]) && length(find(trials_of_cond==32))>0    
                cond_missing = cond;  tr_missing_in_cond=find(tr_cond{cond}==32);
                trials_of_cond(tr_missing_in_cond)=[]; 
            end
            
            for test_tr=trials_of_cond
                disp(test_tr)
                % Leave-one-trial-out cross-validation (CV) training/testing
                train_tr = setxor(trials_of_cond,test_tr);
                Stim_cond = {Stim{:}}';
                EEG_cond = {EEG_trialsCond{:}}';
                stats = mTRFcrossval(Stim_cond(train_tr),EEG_cond(train_tr),fps,1,minlag,maxlag,lambdas,'verbose',0);
                
                % Find optimal lambda out of the CV folds
                % average over CV folds
                cv_avgtr = squeeze(mean(stats.r,1));
                % average over channels
                cv_avgtrch(:,test_tr) = mean(cv_avgtr,2)';
                % identify the optimal lambda
                opt_idx = find(cv_avgtrch(:,test_tr)==max(cv_avgtrch(:,test_tr)),1,'first'); % if multiple lambdas have identical performance, pick the smallest lambda
                opt_lmb(test_tr) = lambdas(opt_idx);

                % Fit the model on all training trials
                mdl_test = mTRFtrain(Stim_cond(train_tr),EEG_cond(train_tr),fps,1,minlag,maxlag,opt_lmb(test_tr));
                % Test on the left-out testing trial
                [~,st_test] = mTRFpredict(Stim_cond(test_tr),EEG_cond(test_tr),mdl_test);

                % Store TRF model per trial
                r_test(feat,test_tr,:)                   = st_test.r;  % TRF prediction (r) 
                w_test(feat,test_tr,1:nb_feat(feat),:,:) = mdl_test.w; % TRF model weights  
                b_test(feat,test_tr,:)                   = mdl_test.b; % TRF model intercept
            end
        end
    end

    

   %%% Store TRF model per condition (instead of trial)
    r_cond = [];  w_cond = [];  b_cond=[];
    for feat=1:nb_models
        for cond=1:4  
            trials_of_cond = tr_cond{cond};
            
            % Exception message for one trial (the last one) of one dyad absent
            if ismember(p,[73,74]) && length(find(trials_of_cond==32))>0    
                cond_missing = cond;  tr_missing_in_cond=find(tr_cond{cond}==32);
                trials_of_cond(tr_missing_in_cond)=[]; 
            end
               
            % Store
            for tr=1:length(trials_of_cond)
                r_cond(feat,cond,tr,:)     = r_test(feat,trials_of_cond(tr),:);
                w_cond(feat,cond,tr,:,:,:) = w_test(feat,trials_of_cond(tr),:,:,:);
                b_cond(feat,cond,tr,:)     = b_test(feat,trials_of_cond(tr),:);
            end
        end
    end
    % Average per cond
    r_cond_avg = squeeze(mean(r_cond,3));
    r_part(iParticipant,:,:,:) = r_cond_avg;
    w_cond_avg = squeeze(mean(w_cond,3));
    w_part(iParticipant,:,:,:,:,:) = w_cond_avg;
    b_cond_avg = squeeze(mean(b_cond,3));
    b_part(iParticipant,:,:,:) = b_cond_avg;

    % Interim plot of the subject-specific TRF weights
    chans_to_plot={'Fz','Cz','Oz'};
    feats_to_plot_label={'Audio','Motion-self','Motion-self-sign','Motion-other','Motion-other-sign','Synchrony'};
    colors = [ [0 0.447 0.741];[0.929 0.694 0.125];[0.529 0.894 0.1];[0.466 0.674 0.188];[0.166 0.274 0.588];[0.2 0.54 0.6]];

    for chan=chans_to_plot
        chan=cell2mat(chan);
        ch=find(strcmp(eeg_label_reorder,chan));
        idxChan = ch;
        cond_label = {'YV-SM','YV-DM','NV-SM','NV-DM'};
        figure;
        set(gcf, 'Position',  [0, 0, 2800, 1000])
        
        absMaxZ = max(abs(w_cond_avg(1,:,:,:,idxChan)),[],'all');        
        for cond=1:4
            nexttile
            
            for feat=1:nb_feat(1)
                plot(squeeze(w_cond_avg(1,cond,feat,:,idxChan)),'LineWidth',1.3,'Color',colors(feat,:)); hold on
            end
            ylim([-absMaxZ*1.3 absMaxZ*1.3])
            xlim([1 length(mdl_test.t)])
            xticks(1:5:length(mdl_test.t)); xticklabels(mdl_test.t(1:5:end));
            if cond==4 legend(feats_to_plot_label,'FontSize',12); end
            title(cond_label{cond},'FontSize',14); 
        end
        exportgraphics(gcf,[output_dir '\INDIV_p' numParticipant '_weights_' chan '.jpg']);
        close;
    end

    iParticipant = iParticipant +1;
end


% Save weights and intercepts to store the subject-specific TRF models
save([output_dir 'TRFmodels'],'mdl_test','w_part','b_part');

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         PLOT GRAND AVERAGE TRF WEIGHTS         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot normalised TRF weights
for p = 1:iParticipant-1
    for model=1:nb_models
        for pred=1:nb_feat(1)
            for ch=1:64
                w_part_plot(p,model,:,pred,:,ch) = w_part(p,model,:,pred,:,ch) ./ max(abs(w_part(p,model,:,pred,:,ch)),[],"all");
            end
        end
    end
end
w_grandavg_plot        = squeeze(mean(w_part_plot,1)); 
w_grandavg_plot_stderr = squeeze(std(w_part_plot,1)) ./ sqrt(size(w_part_plot,1)); 


% Plot
chans_to_plot={'Fz','Cz','Oz'};

for chan=1:length(chans_to_plot)
    ch=find(strcmp(eeg_label_reorder,chans_to_plot{chan}));
    idxChan = ch;
    cond_label = {'YV-SM','YV-DM','NV-SM','NV-DM'};
    figure;
    set(gcf, 'Position',  [0, 0, 1800, 1000])
    tiledlayout(2,4, 'TileSpacing', 'compact', 'Padding', 'compact');    
    
    for i=1:nb_feat(1)
        pred=i;
        absMaxZ = max(abs(w_grandavg_plot(1,:,pred,10:end-10,idxChan)),[],'all');
        nexttile

        xval = 1:length(mdl_test.t);
        plot(squeeze(w_grandavg_plot(1,1,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(1,:));  fefe_plotShadedError(xval, squeeze(w_grandavg_plot(1,1,pred,:,idxChan))', squeeze(w_grandavg_plot_stderr(1,1,pred,:,idxChan))'); 
        plot(squeeze(w_grandavg_plot(1,2,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(2,:));  fefe_plotShadedError(xval, squeeze(w_grandavg_plot(1,2,pred,:,idxChan))', squeeze(w_grandavg_plot_stderr(1,2,pred,:,idxChan))'); 
        plot(squeeze(w_grandavg_plot(1,3,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(3,:));  fefe_plotShadedError(xval, squeeze(w_grandavg_plot(1,3,pred,:,idxChan))', squeeze(w_grandavg_plot_stderr(1,3,pred,:,idxChan))');
        plot(squeeze(w_grandavg_plot(1,4,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(4,:));  fefe_plotShadedError(xval, squeeze(w_grandavg_plot(1,4,pred,:,idxChan))', squeeze(w_grandavg_plot_stderr(1,4,pred,:,idxChan))');
        
        ylim([-absMaxZ*1.5 absMaxZ*1.5])
        xlim([1 length(mdl_test.t)])
        xticks(1:5:length(mdl_test.t)); xticklabels(mdl_test.t(1:5:end));
        if cond==4 legend(cond_label,'FontSize',12); end
        title(feats_to_plot_label{i},'FontSize',14); 
    end
    exportgraphics(gcf,[output_dir 'GDAVG_weights_' cell2mat(eeg_label_reorder(ch)) '.jpg']);
    exportgraphics(gcf,[output_dir 'GDAVG_weights_' cell2mat(eeg_label_reorder(ch)) '.pdf']);
    close;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     TEST TRF ON AVERAGE WEIGHTS OF N-1 SUBJ    %%%
%%%            TESTED ON 1 LEFTOUT SUBJ            %%%
%%%               (GENERIC APPROACH)               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TRF (only test, not train)
idxStart = 1;
idxStop = length(mdl_test.t);
for feat=1:nb_models
    if feat==1 Stim = Stim_part_full; end
    if feat==2 Stim = Stim_part_minus_audio; end
    if feat==3 Stim = Stim_part_minus_self; end
    if feat==4 Stim = Stim_part_minus_other; end
    if feat==5 Stim = Stim_part_minus_IMS; end
    if feat==6 Stim = Stim_part_minus_self_sign; end
    if feat==7 Stim = Stim_part_minus_other_sign; end
    for cond=1:4
        r_opt=[]; r_opt_avg=[];
        disp(['TRF RUNNING - FEAT ' num2str(feat) ' - COND ' num2str(cond)])
        for p_test=1:iParticipant-1
            disp(['PARTICIPANT ' num2str(p_test)])

            % Create a TRF model based on average of N-1 subjects' TRF
            p_train = setxor(1:iParticipant-1,p_test);
            w_avgTrain = mean(w_part(p_train,feat,cond,1:nb_feat(feat),idxStart:idxStop,:),1);
            b_avgTrain = mean(b_part(p_train,feat,cond,:),1);

            mdl_subjTrain = mdl_test;     % Just create the structure of TRF model, then update it
            mdl_subjTrain.t = mdl_test.t(idxStart:idxStop);
            mdl_subjTrain.w = reshape(w_avgTrain,size(w_avgTrain,4:6)); 
            mdl_subjTrain.b = reshape(b_avgTrain,size(b_avgTrain,3:4));
   
            % Select Stim and EEG data of the 1 left-out subject
            Stim_cond = Stim{cond,p_test};
            EEG_cond = EEG_part{cond,p_test};
            
            % Predict EEG of this left-out subject from the averaged TRF of
            % N-1 subjects
            [~,st_test] = mTRFpredict(Stim_cond,EEG_cond,mdl_subjTrain);
            gen_r_part(feat,cond,p_test,:) = squeeze(mean(st_test.r));
        end
    end
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  PLOT TOPOGRAPHIES OF DELTA R FOR EACH MODEL   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute delta r per participant
gen_diff_r_part_topo = squeeze(gen_r_part(1,:,:,:) - gen_r_part(2:end,:,:,:));
% Compute grand average
GA_gen_diff_r_part = squeeze(mean(gen_diff_r_part_topo,3));

% Plot it in two files providing 4 columns: 
% YesVision-sameMusic | YesVision-diffMusic | NoVision-sameMusic | NoVision-diffMusic
% and 6 rows:
% 1. audio/music
% 2. motion self
% 3. motion other
% 4. motion self sign
% 5. motion other sign
% 6. social coordination
iPlot=1;
figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
for d=1:3
    absMaxZ_orig{d} = max(abs(GA_gen_diff_r_part(d,:,:)),[],'all');
    absMaxZ = absMaxZ_orig{d};
    for cond=1:4
        subplot(3,4,iPlot);
        ft_corrNeck_grandavg{cond} = struct;
        ft_corrNeck_grandavg{cond}.freq = 1;
        ft_corrNeck_grandavg{cond}.label = eeg_label_reorder;
        ft_corrNeck_grandavg{cond}.powspctrm = squeeze(GA_gen_diff_r_part(d,cond,:));
    
        cfg = [];
        cfg.marker       = 'on';
        cfg.colorbar     = 'yes';
        cfg.colormap     = '*RdBu';
        cfg.layout       = 'biosemi64';
        cfg.figure = 'gca';
        cfg.comment='no';
        cfg.zlim = [-absMaxZ absMaxZ];
        ft_topoplotTFR(cfg, ft_corrNeck_grandavg{cond});
        title(cond_label(cond),'FontSize',14); 
        iPlot=iPlot+1;
    end
end
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG1.jpg']);
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG1.pdf']);
close;


iPlot=1;
figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
for d=4:nb_models-1
    absMaxZ_orig{d} = max(abs(GA_gen_diff_r_part(d,:,:)),[],'all');
    absMaxZ = absMaxZ_orig{d};
    for cond=1:4
        subplot(nb_models-4,4,iPlot);
        ft_corrNeck_grandavg{cond} = struct;
        ft_corrNeck_grandavg{cond}.freq = 1;
        ft_corrNeck_grandavg{cond}.label = eeg_label_reorder;
        ft_corrNeck_grandavg{cond}.powspctrm = squeeze(GA_gen_diff_r_part(d,cond,:));
    
        cfg = [];
        cfg.marker       = 'on';
        cfg.colorbar     = 'yes';
        cfg.colormap     = '*RdBu';
        cfg.layout       = 'biosemi64';
        cfg.figure = 'gca';
        cfg.comment='no';
        cfg.zlim = [-absMaxZ absMaxZ];
        ft_topoplotTFR(cfg, ft_corrNeck_grandavg{cond});
        title(cond_label(cond),'FontSize',14); 
        iPlot=iPlot+1;
    end
end
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG2.jpg']);
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG2.pdf']);
close;


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   STATISTICAL COMPARISONS ACROSS CONDITIONS    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Statistical comparisons of delta r across conditions
% ROI is found for each subject, as electrodes showing a gain (delta r>0)
% ROI is defined across conditions where a response is expected to be
% i.e. all for "audio"/"self" models
% visual conditions for "other"/"coordination" models

% Compute delta r per participant
gen_diff_r_part_topo = squeeze(gen_r_part(1,:,:,:) - gen_r_part(2:end,:,:,:));

% Compute the participant-specific delta r on the ROI, for each condition
nb_other_store=[];
for part=1:iParticipant-1
    % Compute average delta r for YV and NV conditions
    gen_diff_r_YV = squeeze(mean(gen_diff_r_part_topo(:,1:2,part,:),2));
    gen_diff_r_all = squeeze(mean(gen_diff_r_part_topo(:,1:4,part,:),2));
        
    % Define ROI (on NV for audio/self and YV for other/IMS)
    ROI_audio = find(gen_diff_r_all(1,:)>0);
    ROI_self = find(gen_diff_r_all(2,:)>0);
    ROI_other = find(gen_diff_r_YV(3,:)>0);
    ROI_IMS = find(gen_diff_r_YV(4,:)>0);

    % (optional) check the number of electrodes the ROI includes
    nb_audio_store(part) = round(length(ROI_audio));
    nb_self_store(part) = round(length(ROI_self));
    nb_other_store(part) = round(length(ROI_other));
    nb_IMS_store(part) = round(length(ROI_IMS));
    
    % Compute the delta r on the ROI, for each condition
    for cond=1:4
        gen_diff_r_part_av(1,cond,part)=squeeze(mean(gen_diff_r_part_topo(1,cond,part,ROI_audio)));
        gen_diff_r_part_av(2,cond,part)=squeeze(mean(gen_diff_r_part_topo(2,cond,part,ROI_self)));
        gen_diff_r_part_av(3,cond,part)=squeeze(mean(gen_diff_r_part_topo(3,cond,part,ROI_other)));
        gen_diff_r_part_av(4,cond,part)=squeeze(mean(gen_diff_r_part_topo(4,cond,part,ROI_IMS)));
    end
end

% STATISTICS
% ANOVA on these delta r
pValues_feat = zeros(4,4);
F_feat = zeros(4,4);
for feat=1:4
    feature_stat = squeeze(gen_diff_r_part_av(feat,:,:));
    data = table([1:iParticipant-1].', feature_stat(1,:)',  feature_stat(2,:)', feature_stat(3,:)', feature_stat(4,:)', 'VariableNames', {'id', 'v1_m0', 'v1_m1', 'v0_m0', 'v0_m1'});
    w = table(categorical([1 1 2 2].'), categorical([1 2 1 2].'), 'VariableNames', {'vis', 'music'}); % within-desing
    rm = fitrm(data, 'v0_m1-v1_m0 ~ 1', 'WithinDesign', w);
    anova_res = ranova(rm, 'withinmodel', 'vis*music');
    pValues_feat(:,feat) = anova_res.pValue([1,3,5,7]);
    F_feat(:,feat) = anova_res.F([1,3,5,7]);
    posthoc{feat} = multcompare(rm,'music','By','vis');
end
save([output_dir 'pValues_notCorrected.mat'], 'pValues_feat')
save([output_dir 'fValues.mat'], 'F_feat')
save([output_dir 'posthoc_tests.mat'], 'posthoc')

% PLOTS
% Compute grand average (GA) and standard error (SE) of delta r 
% (now 1bar = 1 feat/1 condition)
GA_gen_diff_r_av = squeeze(mean(gen_diff_r_part_av,3));
SE_gen_diff_r_av = squeeze(std(gen_diff_r_part_av,0,3))./sqrt(70);

x=1:4;
figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
for feat=1:4
    subplot(1,4,feat);
    bar(x,GA_gen_diff_r_av(feat,:));
    hold on
    er = errorbar(x,GA_gen_diff_r_av(feat,:),SE_gen_diff_r_av(feat,:),SE_gen_diff_r_av(feat,:));    
    er.Color = [0 0 0];                            
    er.LineStyle = 'none'; 
end
exportgraphics(gcf,[output_dir 'ROI_UNIQUE_contrib.jpg']);
exportgraphics(gcf,[output_dir 'ROI_UNIQUE_contrib.pdf']);
close;






