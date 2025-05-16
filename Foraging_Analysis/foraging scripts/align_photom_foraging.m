function [all] = align_photom_foraging(SessionData, Photo, Time)
    sampleRate = SessionData.TrialSettings(1).GUI.NidaqSamplingRate;      
    DecimateFactor=100;
    decSR =sampleRate./DecimateFactor;
    %% Align photometry data
    points=2*decSR;
    all = NaN(length(Photo),(2.5*points)+1);

    for i =1:length(Photo)
        try
            LeftReward=find(SessionData.RawData.OriginalStateData{1, i }==5);
            RightReward=find(SessionData.RawData.OriginalStateData{1, i  }==6);

            if ~isempty(LeftReward)
                try
                    OnsetTime(i)=SessionData.RawData.OriginalStateTimestamps{1,i }(LeftReward);
                    PhotoOnset(i)=find(Time{i}> OnsetTime(i),1);

                    all(i,:)=Photo{i}((PhotoOnset(i)- points):(PhotoOnset(i) + 1.5*points)); %./ mean(signal{i}((PhotoOnset(i)- points):(PhotoOnset(i) - points))); %Photo.DFF{i};%(PhotoOnset(i):PhotoOnset(i)+ points);
                catch
                    all(i,:)=NaN;
                end
            elseif ~isempty(RightReward)
                try
                    OnsetTime(i)=SessionData.RawData.OriginalStateTimestamps{1,i}(RightReward);
                    PhotoOnset(i)=find(Time{i}> OnsetTime(i),1);

                    all(i,:)=(Photo{i}((PhotoOnset(i)- points):(PhotoOnset(i) + 1.5*points))); %./ mean(signal{i}((PhotoOnset(i)- points):(PhotoOnset(i) - points)));%Photo.DFF{i};
                catch
                    all(i,:)=NaN;
                end
            end
        catch
            all(i,:)=NaN;
        end
    end

end