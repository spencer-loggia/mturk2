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



nms = {nm4, nm5, nm6};
nms = {nm7, nm8, nm9};
nms = {nm2, nm3};
nms = {nm10};
nms = {nm2, nm3, nm4, nm5, nm6, nm7, nm8, nm9};

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
        shapes = d.Test{1,t};
        colors = d.TestC{1,t};
        lums = d.TestLum{1,t};
        rewSz = d.RewardStage{1,t};
        
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
        
        
        udat = unique([rs;rc;rl]', 'rows');
        if size(udat, 1) ~= 4
            disp(udat);
        end
        
        rspIdx = 1+d.Response{t};
        
        allShape = [allShape; rs];
        allColor = [allColor; rc];
        allLum = [allLum; rl];
        allRsp = [allRsp; 1+rs(rspIdx), 1+rc(rspIdx)];
        allRew = [allRew; rw(rspIdx)];
        maxRew = [maxRew; max(rw(:))];
        rspLum = [rspLum; rl(rspIdx)];
        
    end  % looping through trials
    
    
end  % looping through files


S = 1+allShape(:);
C = 1+allColor(:);
L = 1+allLum(:);


ssz = length(unique(S));
csz = length(unique(C));
lsz = length(unique(L));

z = zeros(ssz, csz);
zc = z;
zr = z;

% z is times stim was chosen
for x = 1:length(S)
    z(S(x), C(x)) = z(S(x), C(x)) + 1;
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
title([fnm, ' - Stimuli shown']);

subplot(1,2,2);
BubblePlot(cc(:), ss(:), zc(:), clr);
ylabel('color');
set(gca, 'YDir', 'reverse');
axis square;
title([fnm, ' - Stimuli chosen']);


figure;
histogram(rspLum);
xlabel('Luminance Index');
ylabel('# trials');

done;






