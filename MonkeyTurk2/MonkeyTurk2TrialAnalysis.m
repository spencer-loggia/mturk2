function MonkeyTurk2TrialAnalysis(nms)

numFiles = length(nms);

allFileData = cell(1, numFiles);

for n = 1:length(nms)
    nm = nms{n};
    [d, ~] = parse_json_file(nm);
    
    if n == 1
        % first file - get field names
        fldnms = fieldnames(d);
        nfld = length(fldnms);
    end
    
    for f = 1:nfld
        fldnm = fldnms{f};
        
        switch class(d.(fldnm))
            case 'char'
                data.(fldnm) = d.(fldnm);
            case 'double'
                data.(fldnm) = d.(fldnm);
            case 'cell'
                s = d.(fldnm);
                sz = size(s);
                
                if sz(1) > 0
                    fldlen = sz(2);
                    fldwid = numel(d.(fldnm){1});
                    newFld = [];
                    
                    for r = 1:fldlen
                        nextRow = (d.(fldnm){r});
                        if isa(nextRow, 'cell')
                            try
                                nextRow = cell2mat(nextRow);
                            catch
                                % broken most likely from empty element
                                nextRow = cell2mat(cellfun(@makeDouble, nextRow, 'UniformOutput', false));
                            end
                        end
                        % we don't preallocate newFld because we don;t know if it's a number or string
                        newFld = [newFld; nextRow];
                    end
                    data.(fldnm) = newFld;
                end
        end  % switch
    end  % fields
    allFileData{n} = data;
end  % files

for d = 1:length(allFileData)
    data = allFileData{d};
    
    disp(nms{d});
    disp(data.BatteryPercent(end, :));
    
    
end  % dara


function d = makeDouble(x)
if isempty(x)
    d = NaN;
else
    d = double(x);
end



