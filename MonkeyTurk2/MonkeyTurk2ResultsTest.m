function MonkeyTurk2ResultsTest()


nm2 = '2021-04-19_12-07-41_Human_2_Tablet_1.txt';
nm3 = '2021-04-20_16-24-16_Human_2_Tablet_1.txt';

nm4 = '2021-04-21_10-43-32_Human_1_Tablet_4.txt';
nm5 = '2021-04-22_09-37-21_Human_1_Tablet_4.txt';
nm6 = '2021-04-23_15-37-55_Human_1_Tablet_4.txt';

nm7 = '2021-04-22_21-01-07_Human_4_Tablet_1.txt';
nm8 = '2021-04-22_23-35-40_Human_4_Tablet_1.txt';
nm9 = '2021-04-23_21-47-56_Human_4_Tablet_1.txt';

nm10 = '2021-04-28_14-38-50_Human_5_Tablet_3.txt';

nm11 = '2021-05-20_12-53-45_Tina_183_Training_Tablet_2.txt';
nm12 = '2021-05-20_12-59-07_Sally_344_Training_Tablet_4.txt';
nm13 = '2021-05-20_13-08-10_Yuri_321_Training_Tablet_1.txt';
nm14 = '2021-05-20_13-13-40_Buzz_319_Training_Tablet_3.txt';
nm15 = '2021-05-21_11-26-18_Buzz_319_Training_Tablet_1.txt';
nm16 = '2021-05-21_11-30-17_Tina_183_Training_Tablet_4.txt';
nm17 = '2021-05-21_11-37-09_Yuri_321_Training_Tablet_2.txt';
nm18 = '2021-05-21_11-41-19_Sally_344_Training_Tablet_3.txt';
nm19 = '2021-05-21_12-39-43_Sally_344_Training_Tablet_3.txt';

nm20 = '2021-05-24_13-36-27_Sally_344_Training_Tablet_2.txt';
nm21 = '2021-05-24_13-40-55_Buzz_319_Training_Tablet_4.txt';
nm22 = '2021-05-24_13-43-53_Tina_183_Training_Tablet_1.txt';
nm23 = '2021-05-24_13-47-12_Yuri_321_Training_Tablet_3.txt';
nm24 = '2021-05-24_14-34-07_Yuri_321_Training_Tablet_3.txt';

nm25 = '2021-06-02_13-59-07_Tina_183_Training_Tablet_2.txt';
nm26 = '2021-06-02_14-04-18_Buzz_319_Training_Tablet_4.txt';
nm27 = '2021-06-02_14-10-35_Sally_344_Training_Tablet_1.txt';
nm28 = '2021-06-02_14-13-55_Yuri_321_Training_Tablet_3.txt';
nm29 = '2021-06-03_14-09-06_Sally_344_Training_Tablet_1.txt';
nm30 = '2021-06-03_14-11-36_Buzz_319_Training_Tablet_2.txt';
nm31 = '2021-06-03_14-29-47_Tina_183_Training_Tablet_4.txt';
nm32 = '2021-06-03_14-37-12_Yuri_321_Training_Tablet_3.txt';

nms = {nm4, nm5, nm6};
nms = {nm7, nm8, nm9};
nms = {nm2, nm3};
nms = {nm10};
nms = {nm2, nm3, nm4, nm5, nm6, nm7, nm8, nm9};

% week 1
snms = {nm12, nm18, nm19, nm20};  % Sally
bnms = {nm14, nm15, nm21};  % Buzz
tnms = {nm11, nm16, nm22};  % Tina
ynms = {nm13, nm17, nm23, nm24};  % Yuri

% week 2
snms = {nm27, nm29};  % Sally
bnms = {nm26, nm30};  % Buzz
tnms = {nm25, nm31};  % Tina
ynms = {nm28, nm32};  % Yuri


% [f, p, idx] = uigetfile('*.txt', 'Data file');
allnms = {snms, bnms, tnms, ynms};



for a = 1:length(allnms)
    
    nms = allnms{a};
    
    allShape = [];
    allColor = [];
    allLum = [];
    allRsp = [];
    allRew = [];
    rspLum = [];
    maxRew = [];
    
    
    [~, fnm, ~] = fileparts(nms{1});
    
    
    for n = 1:length(nms)
        nm = nms{n};
        
        
        [d, ~] = parse_json_file(nm);
        
        
        % 	load('testData');
        
        nt = length(d.Test);
        
        for t = 1:nt
            % what was presented this trial
            shapes = d.Test{1,t};
            colors = d.TestC{1,t};
            lums = d.TestLum{1,t};
            rewSz = d.RewardStage{1,t};
            
            % convert from gnarly JSON to simple vectors
            rs = [];
            rc = [];
            rl = [];
            rw = [];
            for s = 1:length(shapes)
                rs = [rs, shapes{s}];
                rc = [rc, colors{s}];
                rl = [rl, lums{s}];
                rw = [rw, rewSz{s}];
            end
            
            % check for duplicate stimuli, just for info sake.  Should be rare
            udat = unique([rs;rc;rl]', 'rows');
            if size(udat, 1) ~= 4
                disp(['trial ', int2str(t), ' with duplicate stims']);
                disp(udat);
            end
            
            rspIdx = 1+d.Response{t};
            
            % accumulate this trials values into the experiment totals
            allShape = [allShape; rs];
            allColor = [allColor; rc];
            allLum = [allLum; rl];
            allRsp = [allRsp; 1+rs(rspIdx), 1+rc(rspIdx)];
            allRew = [allRew; rw(rspIdx)];
            maxRew = [maxRew; max(rw(:))];
            rspLum = [rspLum; rl(rspIdx)];
            
        end  % looping through trials
        
        
    end  % looping through files
    
    
    % how many of what were presented?
    S = 1+allShape(:);
    C = 1+allColor(:);
    L = 1+allLum(:);
    ssz = length(unique(S));
    csz = length(unique(C));
    lsz = length(unique(L));
    
    
    % make results matrices
    z = zeros(36, 36);
    zc = z;
    zr = z;
    
    % z is times stim was chosen
    for x = 1:length(S)
        try
        z(S(x), C(x)) = z(S(x), C(x)) + 1;
        catch
            disp(' ');
        end;
    end
    
    % zr is rewards received
    for x = 1:size(allRew,1)
        zr(allRsp(x,1), allRsp(x,2)) = zr(allRsp(x,1), allRsp(x,2)) + allRew(x)+1;
    end
    
    % zz is stims chosen
    for x = 1:size(allRsp,1)
        zc(allRsp(x,1), allRsp(x,2)) = zc(allRsp(x,1), allRsp(x,2)) + 1;
    end
    
    clr = [0.4, 0.4, 0.4];
    
    z(z==0) = NaN;
    zc(zc==0) = NaN;
    zr(zr==0) = NaN;
    
    [ss, cc] = meshgrid(unique(S), unique(C));
    
    
    figure;
    subplot(1,2,1);
    BubblePlot(cc(:), ss(:), z(:), clr);
    xlabel('shape');
    ylabel('color');
    set(gca, 'YDir', 'reverse');
    axis square;
    title('Stimuli shown');
    
    subplot(1,2,2);
    BubblePlot(cc(:), ss(:), zc(:), clr);
    ylabel('color');
    set(gca, 'YDir', 'reverse');
    axis square;
    title('Stimuli chosen');
    
    maxr = sum(maxRew-1);
    allr = allRew-1;
    allr(allRew < 1) = 0;
    
    r = sum(allr);
    
    rp = r/maxr;
    nt = size(allRew, 1);
    nf = length(nms);
    
    st1 = [int2str(nf), ' files,   ', int2str(nt), ' trials'];
    st2 = ['Received ', int2str(r), ' rewards out of ', int2str(maxr), ' = ', num2str(100*rp), '%'];
    
    
    suptitle({fnm, st1, st2});
    
    
    
    figure;
    histogram(rspLum);
    xlabel('Luminance Index');
    ylabel('# trials');
    title(fnm);
    
    
end;

done;






