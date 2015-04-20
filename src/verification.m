%verification

%verify the prediction results
%The results will be verified against:
%  1. Image analysis, training, test, verification and predictione error
%  2. AMSR-E


%The process of verification is essentially a process of comaprison between
%two difference data sources, our output are image based, but the
%verification data could be image or vector, so I am writing both
function verification()
    window=40;
    fid = fopen('table_errors.tex','w');
    
    s = ['\\begin{table}[h] \r\n'...
        '\\centering\r\n'...
        '\\begin{tabular}{l||lll}\r\n'...
        '\\hline\r\n'...
                '& $E_{L1}$ & $E_{sig}$ & Std. \\\\\r\n'...
         '\\hline\r\n'];
    fprintf(fid,s);

    [el1_train,esig_train,std_train] = run_verification(window,'train');
    fprintf(fid,'Train \t& %.2f\t& %.2f\t& %.2f\\\\\r\n',el1_train,esig_train,std_train);
    [el1_train,esig_train,std_train] = run_verification(window,'test');
    fprintf(fid,'Test \t& %.2f\t& %.2f\t& %.2f\\\\\r\n',el1_train,esig_train,std_train);
    [el1_train,esig_train,std_train] = run_verification(window,'valid');
    fprintf(fid,'Valid \t& %.2f\t& %.2f\t& %.2f\r\n',el1_train,esig_train,std_train);
    
    s = ['\\end{tabular}\r\n'...
        '\\caption{The error statistics for train, test and validataion datasets.}\r\n'...
        '\\label{table:table_errors}\r\n'...
    '\\end{table}'];
    fprintf(fid,s);
    fclose(fid);
    %copyfile('table_errors.tex','~/Dropbox/WorkSVN/Paper-CNN-SAR-IC/table_errors.tex')
end

function [el1_train,esig_train,std_train] = run_verification(window,set)
    pred_dir = '0/';
    pred_dir = 'SFCRF/data/';
    mask_dir = '~/Work/Sea_ice/gsl2014_hhv_ima/mask/';
    ima_dir = '../dataset/gsl2014_40/ima_used/';
    fid = fopen([set '.txt']);
    days = [];
    str = fgets(fid);
    while ischar(str)
        [~,name,~] = fileparts(str);
        days = [days;name(1:15)];
        str = fgets(fid);
    end

    %fns = strcat(pred_dir,days,'-ic.tif'); 
    fns = strcat(pred_dir,days,'-x1.mat');
    stats = zeros(size(fns,1),4);
    ad = [];
    for i = 1:size(fns,1)   
        %im = imread(fns(i,:));%read prediction reuslts
        load(fns(i,:))
        im = 2-x;
        im(im>1)=1;
        im(im<0)=0;
        disp( fns(i,:))
        ima = load([ima_dir days(i,:) '_ima_used.txt']);
        mask = imread([mask_dir days(i,:) '-mask.tif']);
        index = zeros(size(ima,1),1);
        for j = 1:size(ima,1)
            y = uint32(ceil(ima(j,2)));
            x = uint32(ceil(ima(j,1)));
            index(j) = sum(sum(mask(y-19:y+19,x-19:x+19)));
        end
        index = index == 0;
        ima = ima(index,:);
        [h,w] = size(im);
        d = verify_vector(im,ima);
        stats(i,:) = [mean(abs(d(:,1)-d(:,2))), mean(d(:,1)-d(:,2)), std(d(:,1)-d(:,2)), size(d,1)];
        ad = [ad;d];
    end
    
    t = stats;
    sum(t(:,4))
    %dlmwrite([set '_error.txt'],t,'delimiter','\t','precision',3);
    %dlmwrite([set '_date.txt'],fns,'delimiter','');
    el1_train = sum(t(:,1).*t(:,4))/sum(t(:,4));
    esig_train = sum(t(:,2).*t(:,4))/sum(t(:,4));
    std_train = sqrt(sum(t(:,4).*(t(:,3).^2))/sum(t(:,4)));
    disp([set ':' 'el1:' num2str(el1_train) ' ' ...
        'esig:' num2str(esig_train) ' ' ...
        'std:' num2str(std_train) ' ' ...
         'squar error:' num2str(sqrt(std_train^2+esig_train^2))])

    %plot the error box image
    m=zeros(1,11);
    s=zeros(1,11);
    j=1;
    for i = 0:0.1:1
        index = find(ad(:,2)>i-0.01 & ad(:,2)<i+0.01);
        m(j)=mean(ad(index,1));
        s(j)=std(ad(index,1));
        j=j+1;
    end
    
    %plot_errorbar(m,s,set);
    %plot_hist(ad(:,2),set);
    %plot the histgram of the samples
    
end

function [C, order] = run_verification_quantilize(window,which_set)
    pred_dir = ['~/Work/deeplearning/sar_dnn/output/train_with_2010_' num2str(window) '/'];
    pred_dir = ['~/Work/deeplearning/sar_dnn/output/train_with_2010_2l_32_12/'];
    pred_dir = ['~/Work/deeplearning/sar_dnn/output/train_with_2010_2l_40/'];
    pred_dir = ['~/Work/deeplearning/sar_dnn/output/train_with_2010_2l_40_64/original_500/0/'];
    
    %pred_dir = ['~/Work/deeplearning/sar_dnn/output/train_with_2010_2l_40_64/augment/0/'];
    days=[20100730,
        20100806,
        20100822,
        20100829,
        20100909,
        20100929,
        20101003,
        20101006,
        20101008,
        20110709,
        20110710,
        20110720,
        20110817,
        20110725,
        20110811,
        ]; 
    days = num2str(days);
    traindays = days(1:13,:);
    testdays=days(14,:);
    validdays=days(15,:);  
    
    %traindays = days(1:9,:);
    %testdays=days(10:12,:);
    %validdays=days(13:15,:);
    if strcmp(which_set,'train')
        days = traindays;
    elseif strcmp(which_set,'valid')
        days = validdays;
    elseif strcmp(which_set,'test')
        days = testdays;
    else
        days = which_set
    end
    
    %fns = strcat(pred_dir,days,'.tif');
    fns = strcat(pred_dir,days,'0.tif');
   
    stats = zeros(numel(fns),4);
    ad = [];
    for i = 1:size(fns,1)  
        
        im = imread(fns(i,:));%read prediction reuslts
        
        im(im>1)=1;
        im(im<0)=0;
        disp( fns(i,:))
        %read image analysis, it is a matrix, each row is [x,y,ic]
        date = days(i,1:8);
        data = loadtxt(date);
        [h,w] = size(im);
        index = find(data(:,1) >0 & data(:,1) < w & data(:,2) >0 & data(:,2) < h );
        data = data(index,:);
        
        d = verify_vector(im,data);
        d = round(d*10);
        stats(i,:) = [mean(abs(d(:,1)-d(:,2))), mean(d(:,1)-d(:,2)), std(d(:,1)-d(:,2)), size(d,1)];
        %data(:,3) = d(:,1)-d(:,2);
        %visualize_image_analysis(str2num(date),data);
        ad = [ad;d];
    end
    t = stats;
    %generate confusion matrix: horizontal: xt, vertical: predict
    [C,order]  = confusionmat(ad(:,2),ad(:,1));
    a = sum(C')';
    C = C./repmat(a,1,11);

    %plot the error box image
    
    
end

function plot_errorarea(m,s,which_set)
    figure
    index = find(isnan(m));
    if index== 11
        m(11) = m(10);
        s(11) = s(10);
    else 
        m(index) = (m(index-1)+m(index+1))/2;
        s(index) = (s(index-1)+s(index+1))/2;
    end
    hold on
    patch([0:0.1:1 fliplr(0:0.1:1)],[m+s fliplr(m-s)],[0.7 0.7 0.7],'EdgeColor','none');
    plot(0:0.1:1,0:0.1:1,'--b','LineWidth',1)
    plot(0:0.1:1,m,'r','LineWidth',2)
    %title(['window:' num2str(pred_dir(end-2:end-1)) ' ' which_set])
    set(gcf,'Color','w')
    xlabel('Image analysis');
    ylabel('Estimation from CNN');
    ylim([0,1])
    xtick = 0:0.1:1;
    set(gca,'XTick',xtick);
    set(gca,'FontName','Helvetica');
    set(gca,'FontSize',12);
    set(gca,...
    'Box'         ,'off',...
    'TickDir'     ,'out',...
    'TickLength'  , [.02 .02] , ...
    'XColor'      , [.3 .3 .3], ...
    'YColor'      , [.3 .3 .3], ...
    'YTick'       , 0:0.1:1, ...
    'LineWidth'   , 1         );
    %export_fig tmp.pdf
    %copyfile('tmp.pdf',['~/Dropbox/WorkSVN/Paper-CNN-SAR-IC/figures/','error_',which_set,'.pdf'])
end

function plot_errorbar(m,s,which_set)
    %plot the error bar figure
    figure
    hold on
    barwidth=0.51;
    x = 0:0.1:1;
    %plot up bound 
    bar(x,s+m,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none','BarWidth',barwidth)
    
    %plot average bar
    thick=0.005;
    bar(x,m+thick,'FaceColor','r','EdgeColor','none','BarWidth',barwidth-0.01) % 0.01 narrower to make sure no red board appears in bar plot
    bar(x,m-thick,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none','BarWidth',barwidth)

    %olot lower bound,
    bar(x,m-s,'FaceColor','w','EdgeColor','none','BarWidth',barwidth+0.05)
    %plot lower bound when it is <0
    index = find(m-s>=0);
    m(index)=0;
    s(index)=0;
    h = bar(x,m-s,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none','BarWidth',barwidth);
    baseline_handle = get(h,'BaseLine');
    set(baseline_handle,'Color','w');
    %plot(0.05:0.1:1.05,0,'Color','w','LineWidth',1)
    plot(0:0.1:1,0:0.1:1,'--b','LineWidth',1)
    set(gcf,'Color','w')
    
    ylim([-0.1,1.1])
    xtick = 0:0.1:1;
    set(gca,'XTick',xtick);
    set(gca,'FontName','Helvetica');
    set(gca,'FontSize',12);
    xlabel('Image analysis', 'FontSize',16);
    ylabel('Estimation from CNN', 'FontSize',16);
    set(gca,...
    'Box'         ,'off',...
    'TickDir'     ,'out',...
    'TickLength'  , [.02 .02] , ...
    'XColor'      , [.3 .3 .3], ...
    'YColor'      , [.3 .3 .3], ...
    'YTick'       , 0:0.1:1, ...
    'LineWidth'   , 1         );

    export_fig tmp.pdf
    copyfile('tmp.pdf',['~/Dropbox/WorkSVN/2015-03-CNN-SFCRF-ice-RSE/figures/','error_',which_set,'.pdf'])
end

function plot_hist(ad,which_set)
    figure
    n = hist(ad,0:0.1:1);
    n = n/sum(n);
    bar(0:0.1:1,n,'FaceColor',[0.7,0.7,0.7],'BarWidth',1,'EdgeColor','none');
    set(gcf,'Color','w')
    xlabel('Ice concentration');
    ylabel('Percentage');
    xtick = 0:0.1:1;
    set(gca,'XTick',xtick);
    set(gca,'FontName','Helvetica');
    set(gca,'FontSize',12);
    set(gca,...
    'Box'         ,'off',...
    'TickDir'     ,'out',...
    'TickLength'  , [.02 .02] , ...
    'XColor'      , [.3 .3 .3], ...
    'YColor'      , [.3 .3 .3], ...
    'YTick'       , 0:0.1:1, ...
    'LineWidth'   , 1         );
    export_fig tmp.pdf
    copyfile('tmp.pdf',['~/Dropbox/WorkSVN/2015-03-CNN-SFCRF-ice-RSE/figures/','hist_',which_set,'.pdf'])
end

function delay(seconds)
% function pause the program
% seconds = delay time in seconds
tic;
while toc < seconds
end
end