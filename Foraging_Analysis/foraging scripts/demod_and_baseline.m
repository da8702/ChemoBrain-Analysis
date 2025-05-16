function [Photo, Time] = demod_and_baseline(SessionData)
    lowCutoff = 15 ;
    trials_to_skip =0;
    sampleRate = SessionData.TrialSettings(1).GUI.NidaqSamplingRate;
    modAmpgreen = SessionData.TrialSettings(1).GUI.LED1_Amp;
    modFreqgreen = SessionData.TrialSettings(1).GUI.LED1_Freq;
        
    DecimateFactor=100;

    signal = [];
    decSR =sampleRate./DecimateFactor;
    for i=1:length(SessionData.NidaqData)
        if i > trials_to_skip
            rawDatagreen = SessionData.NidaqData{1,i}(:,1);
            refDatagreen= SessionData.NidaqData{1,i}(:,2);

            demodData{i}= decimate(AP_Demodulation(rawDatagreen,refDatagreen,sampleRate,modAmpgreen,modFreqgreen,lowCutoff), DecimateFactor);
          
            %demodData{i}(1:5) = nanmean(demodData{i}(4:7));
            demodData{i}(1:5) = mean(demodData{i}(4:7), 'omitnan');
  
            decSR =sampleRate./DecimateFactor;
            duration=length( demodData{i})./ decSR;
            ExpectedSize = length( demodData{i});

            %fit double exponential for bleaching
            xdata = linspace(0,duration,ExpectedSize);
            ydata=  demodData{i}';
  
            ydata = ydata - mean(rmoutliers(ydata), 'omitnan');
            t = xdata;
            y = ydata;
            F = @(x,xdata)x(1)*exp(-x(2)*xdata);% + x(3)*exp(-x(4)*xdata);
            intercept = max(rmoutliers(ydata)) - 1;
            x0 = [intercept 0.005];% 1 0.001] ;
            xunc = lsqcurvefit(F, x0, t, y); 
            ydata = (ydata - F(xunc, xdata))';
    
            
            signal= [signal; squeeze(ydata)];
            
        else
            rawDatagreen = SessionData.NidaqData{1,i}(:,1);
            refDatagreen= SessionData.NidaqData{1,i}(:,2);


            demodData{i}= zeros(size(decimate(AP_Demodulation(rawDatagreen,refDatagreen,sampleRate,modAmpgreen,modFreqgreen,lowCutoff), DecimateFactor)));

            %signal= [signal; squeeze(demodData{i})];
        end
    end
    
    %take care of demodulation artifacts 
    
    ydata = signal;
    ydata =  sgolayfilt(ydata, 2, 7);

    duration=length(ydata)./ decSR;
    ExpectedSize = length(ydata);

    %fit double exponential for bleaching
    xdata = linspace(0,duration,ExpectedSize)';

    ydata = ydata - mean(rmoutliers(ydata), 'omitnan');
    t = xdata;
    y = ydata;
    F = @(x,xdata)x(1)*exp(-x(2)*xdata);% + x(3)*exp(-x(4)*xdata);
    intercept = max(rmoutliers(ydata)) - 1;
    x0 = [intercept 0.005];% 1 0.001] ;
    xunc = lsqcurvefit(F, x0, t, y); 
    ydata = (ydata - F(xunc, xdata));
    
    order = 1;
    %30 second moving filter
    %add this back in??
    framelen = .5*decSR*60 + 1;
    
    %for looking at baseline, much longer (200 seconds :o) filter
    %framelen = decSR*200 + 1;
    
    sgf = sgolayfilt(ydata,order,framelen);
    ydata_filt = ydata - sgf;
    ydata_filt=  filloutliers(ydata_filt,"linear", "percentiles", [.1, 99.99]);
   

    dist = fitdist(ydata_filt - min(ydata_filt), 'Gamma');
    
    if dist.a > 25
      dist.a% print(dist.a)
    end
    
    %gamma normalization
   
    quants = 0:.0001:1;
    dig_bins = discretize(ydata_filt, quantile(ydata_filt, quants));
    pd = makedist('Gamma', 1, 1);
    ydata_filt = pd.icdf(quants(dig_bins));

    
    %shove data back into trial structure
    begin = 1;
    for i=1:length(SessionData.NidaqData)
        if i > trials_to_skip
            ExpectedSize = length(demodData{i});
            duration= ExpectedSize/ decSR;
            Photo{i}= ydata_filt(begin: begin+ExpectedSize -1);
            Time{i} = linspace(0,duration,ExpectedSize);
            begin = begin + ExpectedSize;
        else
            Photo{i} = [NaN];
            Time{i} = [0];
            
    end
    
end