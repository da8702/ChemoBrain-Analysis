function datetimeStr = extractDatetime(filename)
    % Regular expression to find the date in the format Mar27_2023
    datePattern = '\w{3}\d{2}_\d{4}';
    
    % Search for the date pattern in the filename
    [startIndex,endIndex] = regexp(filename, datePattern);
    
    if ~isempty(startIndex)
        % Extract the date string from the filename
        dateString = filename(startIndex:endIndex);
        
        % Convert the date string to the 'DD-MMM-YYYY' format
        datetimeObj = datetime(dateString, 'InputFormat', 'MMMdd_yyyy');
        
        % Format the datetime object to the desired output format with default time
        datetimeStr = [datestr(datetimeObj, 'dd-mmm-yyyy'), ' 00:00:00'];
    else
        % If the date pattern is not found, return an empty string or a message
        datetimeStr = 'Date not found in filename';
    end
end