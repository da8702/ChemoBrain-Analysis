function demodData=AP_Demodulation(rawData,refData,sampleRate,modAmp,modFreq,lowCutoff)
% Demodulate an AM-modulated input ('rawData') in quadrature given a
% reference ('refData'). 'LowCutOff' is a corner frequency for 5-pole
% butterworth lowpass filter.

if nargin<5
    lowCutoff=[];
end

%% Prepare reference data and generates 90deg shifted ref data
refData             = refData(1:length(rawData),1);   % adjust length of refData to rawData
refData             = refData-mean(refData);          % suppress DC offset
samplesPerPeriod    = (1/modFreq)/(1/sampleRate);
quarterPeriod       = round(samplesPerPeriod/4);
%refData90           = circshift(refData,[1 quarterPeriod]); %original form
refData90           = circshift(refData,[-quarterPeriod]); % update Michi (7/10/19)

%% Decoding
processedData_0     = rawData .* refData;
processedData_90    = rawData .* refData90;

%% Filter
    lowCutoff = lowCutoff/(sampleRate/2); % normalized CutOff by half SampRate (see doc)
    [b, a] = butter(5, lowCutoff, 'low'); 
    % pad the data to suppress windows effect upon filtering
    pad = 1;
    if pad
        paddedData_0        = processedData_0(1:sampleRate, 1);
        paddedData_90       = processedData_0(1:sampleRate, 1);
        demodDataFilt_0     = filtfilt(b,a,[paddedData_0; processedData_0]);
        demodDataFilt_90    = filtfilt(b,a,[paddedData_90; processedData_90]);        
        processedData_0     = demodDataFilt_0(sampleRate + 1: end, 1);
        processedData_90    = demodDataFilt_90(sampleRate + 1: end, 1);
    else
        processedData_0     = filtfilt(b,a,processedData_0);
        processedData_90    = filtfilt(b,a,processedData_90); 
    end
    
demodData = (processedData_0 .^2 + processedData_90 .^2) .^(1/2);

%% Correct for amplitude of reference
demodData=demodData*2/modAmp;

end