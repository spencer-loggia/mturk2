function scriptTool(cmd)

% 
% foo = matlab.codetools.requiredFilesAndProducts('scriptTool', 'toponly');
% 
% for f = 1:length(foo)
%     disp(foo{f});
% end
% 
% errorok

fontheight = 9; % font height in popup menus
gry = [1 1 1];  % vector for shades of grey in ui controls

% procopts = {...
% 	'Spike density plot',...
% 	'Response latency plot',...
% 	'Ecode values',...
% 	'Trial browser',...
% 	'Saccade latency plot',...
% 	userdefstr,...
% 	};
arrw = ~[...   % % arrow for Add Selected button
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0;...
    0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0;...
    0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0;...
    0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0;...
    0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0;...
    0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0;...
    0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0;...
    0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;...
    ];
r_arrow = zeros(size(arrw,1), size(arrw, 2), 3);
for i = 1:3  % % make RGB array
    r_arrow(:,:,i) = arrw;
end

if nargin  % % argument passed - assumes the figure exists
    % find all the scriptTool figures
    toolfig = findobj(get(0,'children'),'flat',...
        'tag','rextool_fig');
    if length(toolfig) > 1
        % if more than one, pick the TOP one
        toolfig = toolfig(1);
    elseif isempty(toolfig)
        % none found - must be an error
        error('Cannot find the REXtool. There are no figures with Tag = rextool_fig.');
    end
end


if ~nargin
    fight = 600;	% height of figure
    figwid = 600;	% width of figure
    colwid = 250;	% width of text lists and big buttons, etc.
    edwid = 48;		% width of edit buttons
    itemht = 20;	% height of most items
    btnht = 25;		% height of the bigger buttons
    hspc = 3;		% horizontal space between adjacent items
    vspc = 2;		% vertical space between adjacent items
    
    set(0, 'DefaultTextInterpreter', 'none');
    
    unitDesignation = 'pixels';
    
    % make the figure window
    rextool_fig = figure('Color',0.8.*gry, ...
        'MenuBar','none',...
        'Position',[200 50 figwid fight], ...
        'NumberTitle','off', ...
        'Name','REX Tool', ...
        'Tag','rextool_fig');
    
    % place the working directory menu
    pwdpos.x = 10;
    pwdpos.y = fight - 25;
    wdwid = 300;  % width of working dir menu (wider than other items)
    
    % text tag above menu
    % 	ui.workdir_menu_tag = uicontrol('Parent',rextool_fig, ...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.73.*gry, ...
    % 		'Position',[pwdpos.x pwdpos.y wdwid-(edwid+hspc) itemht], ...
    % 		'String','Current working directory', ...
    % 		'Style','text', ...
    % 		'Tag','workdir_menu_tag',...
    %         'Units', unitDesignation);
    %
    % 	% button to edit list contents
    % 	ui.workdir_menu_edit_btn = uicontrol('Parent',rextool_fig, ...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.73.*gry, ...
    % 		'Callback','scriptTool(''workdir_menu_edit_btn'');', ...
    % 		'Position',[pwdpos.x+wdwid-edwid, pwdpos.y, edwid, itemht], ...
    % 		'String','Edit', ...
    % 		'Tag','workdir_menu_edit_btn',...
    %         'Units', unitDesignation);
    %
    % 	% list of valid working directories
    % 	ui.workdir_menu = uicontrol('Parent',rextool_fig, ...
    % 		'CallBack','scriptTool(''workdir_menu'');',...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.9.*gry, ...
    % 		'FontSize',fontheight, ...
    % 		'HorizontalAlignment','left', ...
    % 		'Position',[pwdpos.x, pwdpos.y-(itemht+vspc), wdwid, itemht], ...
    % 		'String','empty list', ...
    % 		'Value',1, ...
    % 		'Style','popupmenu', ...
    % 		'Tag','workdir_menu',...
    %         'Units', unitDesignation);
    %
    %
    %
    %
    %
    % 	% place the ecode file menu relative to working dir menu
    % 	wmpos = get(ui.workdir_menu_edit_btn, 'Position');
    % 	ecdpos.x = wmpos(1) + wmpos(3) + 15;
    % 	ecdpos.y = wmpos(2);
    %
    
    
    
    % place the data directory menu
    ddirpos.x = 10;
    ddirpos.y = fight-40;
    
    % text tag above the menu
    ui.datadir_menu_tag = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry, ...
        'Position',[ddirpos.x ddirpos.y colwid-(edwid+hspc) itemht], ...
        'String','Data directory', ...
        'Style','text', ...
        'Tag','datadir_menu_tag',...
        'Units', unitDesignation);
    
    % button to edit list contents
    ui.datadir_menu_edit_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry, ...
        'Callback','scriptTool(''datadir_menu_edit_btn'');', ...
        'Position',[ddirpos.x+colwid-edwid, ddirpos.y, edwid, itemht], ...
        'String','Edit', ...
        'Tag','datadir_menu_edit_btn',...
        'Units', unitDesignation);
    
    % list of raw data directories
    ui.datadir_menu = uicontrol('Parent',rextool_fig, ...
        'CallBack','scriptTool(''datadir_menu'');',...
        'Units','pixels', ...
        'BackgroundColor',0.9.*gry, ...
        'FontSize',fontheight, ...
        'HorizontalAlignment','left', ...
        'Position',[ddirpos.x, ddirpos.y-(itemht+vspc), colwid, itemht], ...
        'String','empty list', ...
        'Value',1, ...
        'Style','popupmenu', ...
        'Tag','datadir_menu',...
        'Units', unitDesignation);
    
    
    
    
    
    % 	% place the status bar at the bottom of the figure
    statpos.x = 10;
    statpos.y = 5;
    tagwid = 55;
    stat_ht = 30;
    %
    % 	% tag to left of status field
    % 	ui.status_tag = uicontrol('Parent',rextool_fig, ...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.73.*gry, ...
    % 		'CallBack','scriptTool(''start_btn'');',...
    % 		'Position',[statpos.x statpos.y tagwid stat_ht], ...
    % 		'String','Paused', ...
    % 		'Tag','status_tag',...
    %         'Units', unitDesignation);
    %
    %
    % 	% Status text field for holding information to the user
    % 	swidth = figwid - (tagwid+statpos.x+tagwid+hspc);
    % 	ui.status_field = uicontrol('Parent',rextool_fig, ...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.85.*gry, ...
    % 		'FontSize', 9,...
    % 		'HorizontalAlignment', 'left',...
    % 		'Position',[statpos.x+tagwid+hspc statpos.y swidth stat_ht], ...
    % 		'String',' ', ...
    % 		'Style','text', ...
    % 		'Tag','status_field',...
    %         'Units', unitDesignation);
    %
    %
    %
    % 	% Button to retrieve data from lab vis FTP
    % 	% Holding SHIFT while clicking lets you change the server parameters
    dbtnx = 10;
    dbtny = 43;
    % 	ui.getdata_btn = uicontrol('Parent',rextool_fig, ...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.85.*gry, ...
    % 		'BusyAction', 'cancel',...
    % 		'CallBack','scriptTool(''getdata_btn'');',...
    % 		'FontSize', 9,...
    % 		'HorizontalAlignment', 'left',...
    % 		'Interruptible', 'off',...
    % 		'KeyPressFcn', @fig_keypress,...
    % 		'Position',[dbtnx, dbtny, colwid, btnht], ...
    % 		'String','Retrieve new data from lab', ...
    % 		'Tag','getdata_btn',...
    %         'Units', unitDesignation);
    %
    % 	% button to backup data to a server vis FTP
    % 	% Holding SHIFT while clicking lets you change the server parameters
    % 	ui.backupdata_btn = uicontrol('Parent',rextool_fig, ...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.85.*gry, ...
    % 		'BusyAction', 'cancel',...
    % 		'CallBack','scriptTool(''backupdata_btn'');',...
    % 		'FontSize', 9,...
    % 		'HorizontalAlignment', 'left',...
    % 		'Interruptible', 'off',...
    % 		'KeyPressFcn', @fig_keypress,...
    % 		'Position',[dbtnx, dbtny+btnht+vspc, colwid, btnht], ...
    % 		'String','Backup data to server', ...
    % 		'Tag','backupdata_btn',...
    %         'Units', unitDesignation);
    
    % button for filtering the raw data directory
    dbtnwid = 70;
    dbtnypos = ddirpos.y - 2*(itemht+2*vspc);
    ui.filter_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.85.*gry, ...
        'Callback','scriptTool(''filter_btn'')', ...
        'Position',[dbtnx dbtnypos dbtnwid itemht], ...
        'String','Rescan dir', ...
        'Tag','filter_btn',...
        'Units', unitDesignation);
    % radio button to sort raw data directory by date
    ui.date_check = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.85.*gry, ...
        'Callback','scriptTool(''date_check'')', ...
        'Position',[dbtnx+dbtnwid+hspc dbtnypos dbtnwid itemht], ...
        'Style', 'CheckBox',...
        'String','by Date', ...
        'Value',1, ...
        'Tag','date_check',...
        'Units', unitDesignation);
    
    % Field for entering filter string for raw data directory
    ftagwid = 80;
    fypos = dbtnypos - (itemht+vspc);
    
    % tag to right of text entry area
    ui.filter_tag = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry, ...
        'Position',[dbtnx fypos ftagwid itemht], ...
        'String','File filter', ...
        'Style','text', ...
        'Tag','filter_tag',...
        'Units', unitDesignation);
    
    % File filter string
    ui.filter_string = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',1.0.*gry, ...
        'Callback','scriptTool(''filter_btn'')', ...
        'HorizontalAlignment', 'left',...
        'Position',[dbtnx+ftagwid+hspc fypos colwid-(ftagwid+hspc) itemht], ...
        'String','*tablet*.txt', ...
        'Style','edit', ...
        'Tag','filter_string',...
        'Units', unitDesignation);
    
    
    % place the list holding the raw data file names
    % 	bpos = get(ui.backupdata_btn, 'Position');
    % 	lbot = bpos(2)+btnht+vspc;
    lbot = 2*(btnht+vspc);
    ltop = fypos-4*vspc;
    lht = ltop-lbot;
    ui.rawdata_list = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[1 1 1], ...
        'Callback','scriptTool(''rawdata_list'');', ...
        'FontName','Courier', ...
        'Min',1,...
        'Max',5000,...
        'Position',[ddirpos.x lbot colwid lht], ...
        'String',[], ...
        'Style','listbox', ...
        'Tag','rawdata_list', ...
        'Value',1,...
        'Units', unitDesignation);
    
    % the list holding the chosen files (files to be processed)
    fwid = get(rextool_fig, 'Position');
    l_left = fwid(4) - (colwid+hspc);
    ui.chosendata_list = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[1 1 1], ...
        'Callback','scriptTool(''chosendata_list'');', ...
        'FontName','Courier', ...
        'Min',1,...
        'Max',5000,...
        'Position',[l_left lbot colwid lht], ...
        'String',[], ...
        'Style','listbox', ...
        'Tag','chosendata_list', ...
        'Value',1,...
        'Units', unitDesignation);
    
    
    
    % the ADD button sits between the raw data and the chosen data lists
    pos_a = get(ui.rawdata_list, 'Position');
    pos_b = get(ui.chosendata_list, 'Position');
    add_left = pos_a(1)+pos_a(3)+hspc;
    add_right = pos_b(1)-hspc;
    add_wid = fwid(4)-add_left-(fwid(4)-add_right);
    add_bot = (pos_a(2)+pos_a(4)/2)-(4*itemht+vspc)/2;
    % the text tag sits above the button
    ui.add_tag = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry, ...
        'Position',[add_left add_bot+3*itemht+vspc add_wid itemht], ...
        'String','Add selected', ...
        'Style','text', ...
        'Tag','add_tag',...
        'Units', unitDesignation);
    % the button has an arrow (CData) showing the direction that the
    % choices go
    ui.add_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.85.*gry, ...
        'CData', 0.85.*r_arrow,...
        'Callback','scriptTool(''add_btn'')', ...
        'Position',[add_left add_bot add_wid 3*itemht], ...
        'String','', ...
        'Tag','add_btn',...
        'Units', unitDesignation);
    
    % button to remove selected files from the chosen list
    removewid = colwid*.75;
    removebot = pos_b(2)-(itemht+vspc);
    ui.remove_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.85.*gry, ...
        'Callback','scriptTool(''remove_btn'')', ...
        'Position',[pos_b(1) removebot removewid itemht], ...
        'String','Remove selected', ...
        'Tag','remove_btn',...
        'Units', unitDesignation);
    
    % button to clear the entire chosen list
    clearwid = colwid-(removewid+hspc);
    ui.clear_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.85.*gry, ...
        'Callback','scriptTool(''clear_btn'')', ...
        'Position',[pos_b(1)+removewid+vspc removebot clearwid itemht], ...
        'String','Clear list', ...
        'Tag','clear_btn',...
        'Units', unitDesignation);
    
    % button for saving a list of filenames
    clistpos = get(ui.remove_btn, 'Position');
    savewid = 80;
    ui.save_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[0.733333 0.733333 0.733333], ...
        'Callback','scriptTool(''save_btn'')', ...
        'Position',[clistpos(1), clistpos(2)-(itemht+vspc), savewid, itemht], ...
        'String','Save list', ...
        'Tag','save_btn',...
        'Units', unitDesignation);
    
    % button for loading a list of filenames
    ui.load_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry, ...
        'Callback','scriptTool(''load_btn'')', ...
        'KeyPressFcn', @fig_keypress,...
        'Position',[clistpos(1)+savewid+hspc, clistpos(2)-(itemht+vspc), 80, itemht], ...
        'String','Load List', ...
        'Tag','load_btn',...
        'Units', unitDesignation);
    
    
    % the buttons and menus that determine processing of the chosen files
    procpos = ddirpos;
    
    procpos.x = l_left;
    bwid = 0.75;
    % text tag above the menus
    ui.proc_tag = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry, ...
        'FontSize', 10,...
        'Position',[procpos.x+(colwid/2-(colwid*bwid)/2) procpos.y colwid*bwid 1.25*itemht], ...
        'String','Processing options', ...
        'Style','text',...
        'Tag','proc_tag',...
        'Units', unitDesignation);
    % menu holding pre-defined processing options
    % 	ui.proc_menu = uicontrol('Parent',rextool_fig, ...
    % 		'CallBack','scriptTool(''proc_menu'');',...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.9.*gry, ...
    % 		'FontSize',fontheight, ...
    % 		'HorizontalAlignment','left', ...
    % 		'Interruptible','off',...
    % 		'Position',[procpos.x, procpos.y-(itemht+vspc), colwid, itemht], ...
    % 		'String',procopts, ...
    % 		'Value',1, ...
    % 		'Style','popupmenu', ...
    % 		'Tag','proc_menu',...
    %         'Units', unitDesignation);
    % menu holding user-defined processing options
    % only visible when "User Defined..." is chosen in above menu
    ui.userproc_menu = uicontrol('Parent',rextool_fig, ...
        'CallBack','scriptTool(''userproc_menu'');',...
        'Units','pixels', ...
        'BackgroundColor',0.9.*gry, ...
        'FontSize',fontheight, ...
        'HorizontalAlignment','left', ...
        'Interruptible','off',...
        'Position',[procpos.x, procpos.y-2*(itemht+2*vspc), colwid, itemht], ...
        'String','empty list', ...
        'Value',1, ...
        'Visible', 'on',...
        'Style','popupmenu', ...
        'Tag','userproc_menu',...
        'Units', unitDesignation);
    % the button that initiates processing of chosen files
    ui.proc_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry+[0 .25 0], ...
        'CallBack','scriptTool(''proc_btn'');',...
        'FontWeight', 'bold',...
        'Position',[procpos.x pos_b(2)+pos_b(4)+vspc colwid 1.25*itemht], ...
        'String','PROCESS FILES', ...
        'Tag','proc_btn',...
        'Units', unitDesignation);
    
    % button for editing user-defined process list
    % only visible when the menu is
    listpos = get(ui.userproc_menu, 'Position');
    ui.procedit_btn = uicontrol('Parent',rextool_fig, ...
        'Units','pixels', ...
        'BackgroundColor',0.73.*gry+[0 .25 0], ...
        'CallBack','scriptTool(''procedit_btn'');',...
        'KeyPressFcn', @fig_keypress,...
        'Position',[listpos(1)-(edwid+hspc) listpos(2) edwid itemht], ...
        'String','Edit', ...
        'Visible', 'on',...
        'Tag','procedit_btn',...
        'Units', unitDesignation);
    
    % 	% button for updating the code for this tool
    % 	% shift-click to edit server parameters
    % 	spos = get(ui.status_field, 'Position');
    % 	butl = figwid-(edwid+hspc);
    % 	butb = spos(2);
    % 	ui.update_btn = uicontrol('Parent',rextool_fig, ...
    % 		'Units','pixels', ...
    % 		'BackgroundColor',0.85.*gry, ...
    % 		'BusyAction', 'cancel',...
    % 		'CallBack','scriptTool(''update_btn'');',...
    % 		'HorizontalAlignment', 'left',...
    % 		'Interruptible', 'off',...
    % 		'KeyPressFcn', @fig_keypress,...
    % 		'Position',[butl, butb, edwid, itemht], ...
    % 		'String','Update', ...
    % 		'Tag','update_btn',...
    %         'Units', unitDesignation);
    
    
    ch = get(rextool_fig, 'Children');
    for x = 1:length(ch)
        set(ch(x), 'Units', 'normalized')
    end
    
    drawnow;
    % attach the user interface items to the user data
    ud.ui = ui;
    
    % 	% make the timer to periodocally check the working dir
    % 	tmr = timer('TimerFcn', 'scriptTool(''checkworkingdir'');', 'Period', 3,...
    % 		'BusyMode', 'drop', 'ExecutionMode', 'fixedSpacing',...
    % 		'StartDelay', 2, 'TasksToExecute', Inf,...
    % 		'Name', 'REXWorkingDirTimer');
    %
    % 	% save timer and status in user data struct
    % 	ud.tmr = tmr;
    ud.currkey = '';
    % 	ud.oldworkingdir = pwd;
    
    % save the user data into the figure
    set(rextool_fig, 'UserData', ud);
    
    % make the ui controls scale with the figure
    set(rextool_fig, 'Units', 'normalized');
    
    % see what we can load into the menus
    % try to abort nicely if we fail
    % 	status({'Initializing', 'Please wait'});
    
    % 	try
    % 		workingdir;
    % 		drawnow;
    % 	catch
    % 		status('Initialization failed - set up working directories.');
    % 	end
    try
        getDataDir;
        drawnow;
    catch
        status('Initialization failed - set up raw data directories.');
    end
    % 	try
    % 		load_ecodes;
    % 		drawnow;
    % 	catch
    % 		status('Initialization failed - set up ecode files.');
    % 	end
    try
        allUserFuncs;
        drawnow;
    catch
        status('Initialization failed in ''userfunc''.');
    end
    
    try
        set(rextool_fig, 'CloseRequestFcn', 'scriptTool(''close_btn'');');
        status('Ready');
        % 		scriptTool('start_btn');
    catch
        status('Initialization failed starting timer.');
        % 		stop(tmr);
        set(rextool_fig, 'CloseRequestFcn', 'delete(gcf);');
    end
    
    scriptTool('filter_btn');

    return;
else  % argument passed
    %retrieve handles from userdata
    try
        ud = get(toolfig, 'UserData');
        ui = ud.ui;
    catch
        set(toolfig, 'CloseRequestFcn', 'delete(gcf)');
        status('Error geting REXtool UserData');
    end
end


switch cmd
    % 	case 'start_btn'
    % 		running = strcmp(get(ui.status_tag, 'String'), 'Running');
    % 		if running
    % 			scriptTool('pause');
    %
    % 		else  % paused
    % 			scriptTool('resume');
    % 			scriptTool('filter_btn');
    % 		end
    %
    % 	case 'workdir_menu_edit_btn'
    % 		workdir_edit;
    % 	case 'workdir_menu'
    % 		% workdir chosen - save the old one and make sure the timer's going
    % 		if ~strcmpi(ud.oldworkingdir, workingdir)
    % 			ud.oldworkingdir = workingdir;
    % 			set(toolfig, 'UserData', ud);
    % 			scriptTool('reload_datadir');
    % 			scriptTool('reload_ecodes');
    % 			scriptTool('reload_userfuncs');
    % 			status(['Changed working directory to ',pwd]);
    % 			timer_ping(ud.tmr);
    % 		end
    % 	case 'checkworkingdir'  % timer sends us here
    % 		% only if we've moved do the following
    % 		if ~strcmpi(pwd, ud.oldworkingdir)
    % 			status('Changing working directory');
    % 			% save the old working dir into the userdata
    % 			ud.oldworkingdir = workingdir(pwd);
    % 			set(toolfig, 'UserData', ud);
    % 			% re-initialize the ecode and raw data menus
    % 			scriptTool('reload_datadir');
    % 			scriptTool('reload_ecodes');
    % 			scriptTool('reload_userfuncs');
    % 			status(['Changed working directory to ',pwd]);
    % 		end
    % 		ud.currkey = '';
    % 		set(toolfig, 'UserData', ud);
    case 'datadir_menu_edit_btn'
        % 		datadir_edit;
        
        list_edit(rdname, 'Raw data directory', 1, '*', 'rextool(''reload_datadir'');');
        
        %             list_edit(udname, 'User defined processes', 0, '*.m', 'rextool(''reload_userfuncs'');');
        
        getDataDir;
    case 'datadir_menu'
        getDataDir;
        scriptTool('filter_btn');
    case 'rawdata_list'
        % double click sends the selected item to the chosen list
        if strcmp(get(toolfig,'SelectionType'),'open')  % double-click
            scriptTool('add_btn');
        end
    case 'chosendata_list'
        % double click removes the selected item from the list
        if strcmp(get(toolfig,'SelectionType'),'open')  % double-click
            currval = get(ui.chosendata_list, 'Value');
            scriptTool('remove_btn');
            % make sure the selection doesn't go past the end
            nchosen = length(get(ui.chosendata_list, 'String'));
            currval = min(currval, nchosen);
            set(ui.chosendata_list, 'Value', currval);
        end
        
        
        % 	case 'ecode_menu_edit_btn'
        % 		ecodefile_edit;
        % 	case 'ecode_menu'
        % 		load_ecodes;
        
        % 	case 'getdata_btn'
        % 		% shift-click changes server info
        % 		if strcmpi(ud.currkey, 'shift')
        % % 			disp('foo')
        % 			change_server_info('ftp_data_retrieve_params');
        % 			ud.currkey = '';
        % 			set(toolfig, 'UserData', ud);
        % 		else  % regular click downloads from the lab
        % 			% get_new_data temporarily changes the current directory so we
        % 			% need to stop the timer so scriptTool doesn't try to update the
        % 			% working directory
        % 			stop(ud.tmr);
        % 			get_new_data;
        % 			timer_ping(ud.tmr);
        % 			scriptTool('filter_btn');
        % 		end
        % 	case 'backupdata_btn'
        % 		% shift-click changes server info
        % 		if strcmpi(ud.currkey, 'shift')
        % 			change_server_info('ftp_data_backup_params');
        % 			ud.currkey = '';
        % 			set(toolfig, 'UserData', ud);
        % 		else  % regular click uploads to the server
        % 			% backup_data temporarily changes the current directory so we
        % 			% need to stop the timer so scriptTool doesn't try to update the
        % 			% working directory
        % 			stop(ud.tmr);
        % 			backup_data;
        % 			timer_ping(ud.tmr);
        % 		end
        
    case 'close_btn'
        % delete the timer and the figure
        % 		stop(ud.tmr);
        % 		delete(ud.tmr);
        delete(toolfig);
        
    case {'filter_btn','date_check'}
        % save the old values and put a WAIT message
        set(ui.rawdata_list,'Value',1);
        set(ui.rawdata_list,'String','PLEASE WAIT');
        drawnow;
        
        % get the list from the raw data directory
        fstr = get(ui.filter_string, 'String');
        bydate = get(ui.date_check, 'Value');
        
        fnm = datafilenames(bydate, fstr, ui);
        
        % put the new info in the list
        set(ui.rawdata_list,'String',fnm);
        set(ui.rawdata_list,'Value',1);
    case 'add_btn'
        % this is also called by double-clicking in the raw data list
        whichexpt = get(ui.rawdata_list,'Value');
        exptstrs = get(ui.rawdata_list,'String');
        
        % make a list of names to add
        toadd = cell(length(whichexpt), 1);
        [toadd{:}] = exptstrs{whichexpt};
        toadd = strvcat(toadd);
        
        % add the new names to the old list
        oldchosen = strvcat(get(ui.chosendata_list, 'String'));
        newchosen = deblank(cellstr(strvcat(oldchosen, toadd)));
        
        refresh;
        % put the new list back into the chosel files list
        set(ui.chosendata_list, 'String', newchosen);
        
    case 'clear_btn'
        % just empty the chosen data list
        set(ui.chosendata_list, 'Value', 1);
        set(ui.chosendata_list, 'String', '');
    case 'remove_btn'
        exptstrs = strvcat(get(ui.chosendata_list,'String'));
        if isempty(exptstrs)  % return if nothing to remove
            return;
        end
        % figure out which expt names to KEEP
        whichexpt = get(ui.chosendata_list,'Value');
        numlines = size(exptstrs,1);
        keep = setxor(1:numlines,whichexpt);
        
        % set the new list to the files we're keeping
        set(ui.chosendata_list, 'Value', 1);
        set(ui.chosendata_list, 'String', cellstr(exptstrs(keep, :)));
        
    case 'save_btn'
        % save a list of filenames
        exptnames = get(ui.chosendata_list,'String');
        put_expt_names(exptnames, '');
    case 'load_btn'
        if strcmpi(ud.currkey, 'shift')
            % get a list from an Excel spreadsheet
            fnm = uigetfile('*.xls');
            
            if ~fnm
                return;
            end
            
            [~, txt] = xlsread(fnm, -1);
            
            % if we did it right, txt should contain the filenames
            % no error checking for this yet
            exptnames = txt;
        else
            % load a list of filenames
            exptnames = get_expt_names();
        end
        
        % loaded list gets appended to the chosen file list
        old_list = get(ui.chosendata_list, 'String');
        if isempty(old_list)
            old_list = {};
        end
        if ~isempty(exptnames)
            exptnames = cellstr(exptnames);
            % 			allnames = {old_list{:}, exptnames{:}};
            allnames = [old_list, exptnames];
            set(ui.chosendata_list, 'String', allnames);
        end
        
    case 'proc_menu'
        mstr = get(ui.proc_menu, 'String');
        choice = mstr{get(ui.proc_menu, 'Value')};
        set(ui.procedit_btn, 'Visible', 'on');
        set(ui.userproc_menu, 'Visible', 'on');
    case 'proc_btn'
        % process the chosen files
        % see which option is chosen in menu
        %         choice = get(ui.proc_menu, 'Value');
        enames = get(ui.chosendata_list, 'String');
        if ~isempty(enames)
            uf = userfunc(ui);
            estr = [uf,'(enames);'];
            eval(estr);
        else
            % warn if nothing's been chosen
            warndlg('The right-hand list must not be empty', 'File names needed.');
        end
    case 'procedit_btn'
        if strcmpi(ud.currkey, 'shift')
            ud.currkey = '';
            set(toolfig, 'UserData', ud);
            uf = userfunc(ui);
            estr = ['edit ',uf];
            eval(estr);
        else
            
            list_edit(udname, 'User defined processes', 0, '*.m', 'rextool(''reload_userfuncs'');');
            
            % 			proclist_edit();
        end
        % 	case 'update_btn'
        % 		if strcmpi(ud.currkey, 'shift')
        % 			change_server_info('ftp_code_backup_params');
        % 			ud.currkey = '';
        % 			set(toolfig, 'UserData', ud);
        % 		else
        % 			stop(ud.tmr);
        %  			success = update_my_code;
        % 			if success
        % 				scriptTool('close_btn');
        % 				refresh;
        % 				clear scriptTool;
        % 				rehash toolboxcache;
        % 				scriptTool;
        % % 				warndlg({'Code updated.','Restart REXtool as soon as possible.'},...
        % % 					'Restart REXtool');
        % 			else
        % 				warndlg({'PROBLEM UPDATING CODE.','Please try again.'},...
        % 					'UPDATE ERROR');
        % 				timer_ping(ud.tmr);
        % 			end
        % 		end
        
        
        % 	case 'reload_workingdir'
        % 		currdir = pwd;
        % 		set(ui.workdir_menu, 'Value', 1);
        % 		set(ui.workdir_menu, 'String', {'empty list'});
        % 		workingdir(currdir);
        % 	case 'reload_ecodes'
        % 		load_ecodes(0);
        % 		set(ui.ecode_menu, 'Value', 1);
        % 		set(ui.ecode_menu, 'String', {'empty list'});
        % 		load_ecodes();
    case 'reload_datadir'
        set(ui.datadir_menu, 'Value', 1);
        set(ui.datadir_menu, 'String', {'empty list'});
        getDataDir();
        ud.oldworkingdir = pwd;
        set(toolfig, 'UserData', ud);
        scriptTool('clear_btn');
        scriptTool('filter_btn');
    case 'reload_userfuncs'
        set(ui.userproc_menu, 'Value', 1);
        set(ui.userproc_menu, 'String', {'empty list'});
        userfunc(ui);
        % 	case 'pause'
        % 		set(ui.status_tag, 'String', 'Paused');
        % 		stop(ud.tmr);
        % 	case 'resume'
        % 		timer_ping(ud.tmr);
        % 		set(ui.status_tag, 'String', 'Running');
    otherwise
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function timer_ping(tmr)
return;
try
    if strcmp(get(tmr, 'Running'), 'off')
        start(tmr);
    end
catch
    status('Error pinging timer');
end

function fig_keypress(~, evt)
ud = get(gcbf, 'UserData');
ud.currkey = evt.Modifier;
set(gcbf, 'UserData', ud);

function status(str)
disp(str);

function dd = getDataDir(cname)

rhdl = find_rextool();


hdl = findobj(rhdl, 'Tag', 'datadir_menu');
if isempty(hdl)
    error('Data directory menu not found in tool window.');
elseif length(hdl) > 1
    error('Too many data directory menu handles found in tool.');
else
    mnu = hdl;
    
    ddnm = {};
    if strcmp('empty list', get(mnu, 'String'))
        if exist(rdname, 'file')
            fpath = fileparts(which(rdname));
            if strcmp(pwd, fpath)
                load(rdname);
                set(mnu, 'String', ddnm);
            end
        end
        if isempty(ddnm)
            set(mnu, 'String', 'empty list');
        end
    else
        ddnm = get(mnu, 'String');
    end
    
    if ~isempty(ddnm)
        % check for old file format
        firstentry = ddnm{1};
        if firstentry(1) == '-'
            % old format - remove first line
            strs = strvcat(ddnm);
            if size(strs,1) == 1
                ddnm = {};
            else
                ddnm = cellstr(strs(2:end,:));
            end
            set(mnu, 'String', ddnm);
            % save the file with the first line removed
            save(rdname, 'ddnm');
        end
    end
    
    menuval = get(mnu, 'Value');  % line selected in menu
    if ~isempty(ddnm)
        menuchoice = ddnm{menuval};
    else
        return;
    end
    
    if ~nargin
        dd = menuchoice;
    else
        % argument was passed
        % if it's in the menu, eval
        % ignore final filesep in cname
        if cname(end) == filesep
            cname = cname(1:(end-1));
        end
        idx = strmatch(lower(cname), lower(ddnm));
        if ~isempty(idx)  % found a match
            set(mnu, 'Value', idx);
            dd = ddnm{idx};
        else  % no match
            warndlg({['Raw data directory ''',cname,''' not found in list.'],...
                'Use Edit button to add to list.',...
                ['Selecting last chosen raw data directory: ',menuchoice]},...
                'Data directory not found');
            dd = menuchoice;
        end
    end
    if dd(end) ~= filesep
        dd = [dd,filesep];
    end
end

function fhdl = find_rextool()


% returns the handle to the rextool window.  Creates it if it's not there

h_fig_list = findobj(get(0,'children'),'flat',...
    'tag','rextool_fig');

if isempty(h_fig_list)
    fhdl = [];
elseif length(h_fig_list) >= 1
    fhdl = h_fig_list(1);
end

function fnm = datafilenames(bydate, filtstr, ui)

if ~nargin
    bydate = 0;
    filtstr = [];
elseif nargin == 1
    if isstr(bydate)
        filtstr = bydate;
        bydate = 0;
    end
end

rddir = rawdatadir(ui);
if rddir(end) ~= filesep
    rddir = [rddir,filesep];
end

if isempty(filtstr)
    fls = dir([rddir,'*.txt']);
elseif strcmpi(filtstr, 'all')
    flsa = dir([rddir,'*.txt']);
    fls = [flsa; flse];
else
    %     if filtstr(end) ~= '*'
    %         filtstr = [filtstr,'*'];
    %     end
    fls = dir([rddir,filtstr]);
end

numf = length(fls);

if numf==0
    fnm='';
    return;
end

fnm = cellstr(char(fls.name));

% remove trailing 'A'
if ~strcmpi(filtstr, 'all')
    fnm = deblank(strrep(fnm, 'A', ' '));
end

fnm = char(fnm);


if bydate
    
    fdt = {fls.date};
    fdt = datenum(fdt);
    
    [ignore, idx] = sort(fdt);
    
    fnm = char(fnm);
    fnm = fnm(idx(end:-1:1),:);
else
    fnm = sortrows(fnm);
end

fnm = cellstr(fnm);



function list_edit(fname, dname, dodir, filtstr, cleanup)

if nargin == 1
    cmd = fname;
end

wintag = 'list_edit_window';

if nargin == 0  % new window
    cmd = 'init';
elseif nargin >= 2
    cmd = 'new';
    if nargin < 5
        cleanup = [];
    end
    if nargin < 4
        filtstr = '*.*';
    end
    if nargin < 3
        dodir = 0;
    end
end


if strcmp(lower(cmd),'new')
    command = 0;
elseif strcmp(lower(cmd),'fnames')
    command = 1;
elseif strcmp(lower(cmd),'addfile_btn')
    command = 2;
elseif strcmp(lower(cmd),'removefile_btn')
    command = 3;
elseif strcmp(lower(cmd),'save_btn')
    command = 5;
elseif strcmp(lower(cmd),'init')
    list_edit('new');
    command = -1;
end

if command ~= 0
    h_fig_list = findobj(get(0,'children'),'flat',...
        'tag',wintag);
    if length(h_fig_list) > 1
        h_fig_list = gcf;
    elseif length(h_fig_list) == 0
        if command ~= 9 % no error if closing non-existent window
            error(['There are no figures with Tag = ',wintag,'.']);
        end
    end
    
    ud = get(h_fig_list,'userdata');
    mastfname = ud.savename;
end

if command == 0
    mastfname = fname;
    
    if dodir
        btag = 'directory';
    else
        btag = 'file';
    end
    
    % make the figure and ui
    top_margin = 50;
    ssz = get(0, 'ScreenSize');
    scrtop = ssz(4);
    scr_mid = ssz(3)/2;
    
    fig_ht = 495;
    fig_wid = 492;
    
    fig_bot = scrtop - (top_margin+fig_ht);
    fig_left = round(scr_mid - (fig_wid/2));
    ewin = figure('Color',[0.8 0.8 0.8], ...
        'MenuBar','none',...
        'Position',[fig_left fig_bot fig_wid fig_ht], ...
        'NumberTitle','off', ...
        'Name',dname, ...
        'CloseRequestFcn', 'delete(gcf);',...
        'Tag',wintag);
    figure(ewin);
    refresh;
    
    figtag = uicontrol('Parent',ewin, ...
        'Units','pixels', ...
        'BackgroundColor',[.9 .9 .9], ...
        'FontName','Helvetica', ...
        'FontSize',10, ...
        'Position',[40 450 405 20], ...
        'String',dname, ...
        'Style','text', ...
        'Tag','figtag');
    
    ud.ui.fnames = uicontrol('Parent',ewin, ...
        'Units','pixels', ...
        'BackgroundColor',[1 1 1], ...
        'Callback','list_edit(''fnames'')', ...
        'FontName','Courier', ...
        'Position',[40 40 300 400], ...
        'String',[], ...
        'Style','listbox', ...
        'Tag','fnames_win', ...
        'Value',1);
    
    ud.ui.addfile_btn = uicontrol('Parent',ewin, ...
        'Units','pixels', ...
        'BackgroundColor',[0.733333 0.733333 0.733333], ...
        'Callback','list_edit(''addfile_btn'')', ...
        'Position',[355 415 100 25], ...
        'String',['Add ',btag], ...
        'Tag','addfile_btn');
    
    ud.ui.removefile_btn = uicontrol('Parent',ewin, ...
        'Units','pixels', ...
        'BackgroundColor',[0.733333 0.733333 0.733333], ...
        'Callback','list_edit(''removefile_btn'')', ...
        'Position',[355 380 100 25], ...
        'String',['Remove ', btag], ...
        'Tag','removefile_btn');
    
    ud.ui.save_btn = uicontrol('Parent',ewin, ...
        'Units','pixels', ...
        'BackgroundColor',[0.733333 0.733333 0.733333], ...
        'Callback','list_edit(''save_btn'')', ...
        'Position',[365 200 80 25], ...
        'String','Save changes', ...
        'Tag','save_btn');
    
    % load the master file and display the filenames
    if exist(mastfname, 'file')
        load(mastfname);
    else
        ddnm = {};
    end
    
    % remove any empty lines, duplicates, and save back to master file
    newnm = {};
    newidx = 1;
    if ~isempty(ddnm)
        for n = 1:length(ddnm)
            if ~isempty(deblank(ddnm{n})) &...
                    isempty(strmatch(ddnm{n}, newnm, 'exact'))
                % not found - add it
                newnm{newidx} = ddnm{n};
                newidx = newidx + 1;
            end
        end
    end
    ddnm = newnm;
    save(mastfname, 'ddnm');
    
    % reload master file and set figure list box and userdata
    
    set(ud.ui.fnames, 'String', ddnm);
    ud.savename = mastfname;
    ud.dodir = dodir;
    ud.filtstr = filtstr;
    ud.cleanup = cleanup;
    set(ewin, 'UserData', ud);
    
elseif command == 1  % fnames
    if strcmp(get(gcf,'SelectionType'),'open')  % double-click
        % double-click on list item
    end
elseif command == 2  % addfile_btn
    ddnm = get(ud.ui.fnames, 'String');
    if ud.dodir
        fnm = uigetdir('/', 'Select a directory');
    else
        fnm = uigetfile(ud.filtstr, 'Select a file');
    end
    
    if fnm
        if ~ud.dodir
            extpos = strfind(fnm, '.');
            if ~isempty(extpos)
                fnm = fnm(1:(extpos(1)-1));
            end
        end
        if ~isempty(ddnm)
            ddnm = {ddnm{:}, fnm};
        else
            ddnm = {fnm};
        end
        set(ud.ui.fnames, 'String', ddnm);
        set(ud.ui.fnames, 'Value', length(ddnm));
        save(mastfname, 'ddnm');
    end
elseif command == 3  % removefile_btn
    ddnm = get(ud.ui.fnames, 'String');
    if length(ddnm) > 0
        % find the selected name
        whichline = get(ud.ui.fnames, 'Value');
        toremove = ddnm{whichline};
        
        % remove it from the list
        c = char(ddnm);
        c([whichline:(end-1)],:) = c([(whichline+1):end],:);
        c = c([1:(end-1)],:);
        ddnm = cellstr(c);
        if size(c, 1) == 0
            ddnm = {};
        end
        newval = whichline;
        if newval > length(ddnm)
            newval = length(ddnm);
        end
        set(ud.ui.fnames, 'Value', newval);
        set(ud.ui.fnames, 'String', ddnm);
    end
elseif command == 5   % save btn
    % save current list as master file
    
    ddnm = get(ud.ui.fnames, 'String');
    save(mastfname, 'ddnm');
    if ~isempty(ud.cleanup)
        try
            eval(ud.cleanup);
        catch
        end
    end
    close(gcf);
end


function rn = rdname()

rn = 'rawddirnamesnames.mat';



function ud = udname()

ud = 'userdefinedfunctions.mat';


function nms = get_expt_names(fname)

ext = '.exptlist';

if nargin
    fnm = [fname,ext];
    indata = load('-mat',fnm);
    onms = indata.exptnames;
    
    if nargout
        nms = onms;
    end
else
    
    % make the figure
    fight = 795;
    boxht = fight-95;
    tagy = boxht+20;
    btny = tagy+40;
    sp = get(0, 'ScreenSize');
    
    get_expt_names_fig = figure('Color',[0.8 0.8 0.8], ...
        'MenuBar','none',...
        'Position',[sp(3)-700 sp(4)-(fight)-200 600 fight], ...
        'NumberTitle','off', ...
        'Name','Get experiment names', ...
        'Tag','get_expt_names_fig',...
        'WindowStyle', 'modal');
    
    assignin('base', 'get_expt_names_fig', get_expt_names_fig);
    
    file_names_tag = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'Position',[10 tagy 100 20], ...
        'String','File name', ...
        'Style','text', ...
        'Tag','file_names_tag');
    
    file_names_list = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[1 1 1], ...
        'Callback','set(get_expt_names_fig,''UserData'',''file_names_list'');',...
        'FontName','Courier', ...
        'Position',[10 10 175 boxht], ...
        'String','', ...
        'Style','listbox', ...
        'Tag','file_names_list', ...
        'Value',1);
    
    expt_names_tag = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'Position',[200 tagy 100 20], ...
        'String','Experiment names', ...
        'Style','text', ...
        'Tag','expt_names_tag');
    
    expt_names_list = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[1 1 1], ...
        'FontName','Courier', ...
        'Position',[200 10 175 boxht], ...
        'String','', ...
        'Style','listbox', ...
        'Tag','expt_names_list', ...
        'Value',1);
    
    comment_tag = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'Position',[390 tagy 100 20], ...
        'String','Comments', ...
        'Style','text', ...
        'Tag','comment_tag');
    
    comment_text = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[1 1 1], ...
        'FontName','Courier', ...
        'HorizontalAlignment','left',...
        'Min',0,...
        'Max',12,...
        'Position',[390 10 200 boxht], ...
        'String','', ...
        'Style','text', ...
        'Tag','comment_text');
    
    load_btn = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[.2 0.4 .2], ...
        'Callback','set(get_expt_names_fig,''UserData'',''load_btn'');',...
        'Position',[180 btny 260 30], ...
        'String','Load file names', ...
        'Tag','load_btn');
    
    cancel_btn = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[0.733333 0.733333 0.733333], ...
        'Callback','set(get_expt_names_fig,''UserData'',''cancel_btn'');',...
        'Position',[550 btny 40 30], ...
        'String','Cancel', ...
        'Tag','cancel_btn');
    
    delete_btn = uicontrol('Parent',get_expt_names_fig, ...
        'Units','pixels', ...
        'BackgroundColor',[1 0.5 0.5], ...
        'ForegroundColor',[.6 .2, .2], ...
        'Callback','set(get_expt_names_fig,''UserData'',''delete_btn'');',...
        'Position',[10 btny 100 30], ...
        'String','Delete selection', ...
        'Tag','delete_btn');
    
    drawnow;
    pause(0.1);
    
    % initialize
    % get files with list of expt names
    dd = dir(['*',ext]);
    
    % make a cell to hold names only
    numf = length(dd);
    fnm = cell(1,numf);
    
    % put filenames into cell
    [fnm{:}] = deal(dd.name);
    fnm = strrep(fnm, ext, '');
    
    % put names in the list
    set(file_names_list, 'String', fnm);
    
    % initialize other fields from first file
    str = get(file_names_list, 'String');
    fnm = str{1};
    
    % add the extension and load it
    fnm = [fnm,ext];
    indata = load('-mat',fnm);
    if ~isfield(indata, 'comment')
        indata.comment = '';
    end
    
    % set expt name and comment fields
    set(expt_names_list, 'String', indata.exptnames);
    set(comment_text, 'String', indata.comment);
    
    
    done = 0;
    while ~done
        % wait for a click
        
        waitfor(get_expt_names_fig, 'UserData');
        cmd = get(get_expt_names_fig, 'UserData');
        set(get_expt_names_fig, 'UserData', []);
        
        switch cmd,
            
            case 'file_names_list',
                % get the name of the expt
                itm = get(file_names_list, 'Value');
                str = get(file_names_list, 'String');
                fnm = str{itm};
                
                % add the extension and load it
                fnm = [fnm,ext];
                indata = load('-mat',fnm);
                if ~isfield(indata, 'comment')
                    indata.comment = '';
                end
                
                % set expt name and comment fields
                set(expt_names_list, 'String', indata.exptnames);
                set(comment_text, 'String', indata.comment);
                
                if strcmp(get(gcf,'SelectionType'),'open')  % double-click
                    onms = get(expt_names_list,'String');
                    done = 1;
                end
                
            case 'cancel_btn',
                onms = [];
                done = 1;
            case 'load_btn',
                onms = get(expt_names_list,'String');
                done = 1;
            case 'delete_btn',
                % get the name of the expt
                itm = get(file_names_list, 'Value');
                str = get(file_names_list, 'String');
                fnm = str{itm};
                
                % add the extension
                fnm = [fnm,ext];
                
                confirm = questdlg(['Really delete ',fnm,'?']);
                if strcmp(lower(confirm), 'yes')
                    % back it up
                    bkpnm = [fnm,'.deleted'];
                    exptnames = get(expt_names_list,'String');
                    comment = get(comment_text,'String');
                    save(bkpnm, 'exptnames', 'comment', '-mat');
                    
                    % now delete it
                    delete(fnm);
                    
                    % reload a new list
                    % get files with list of expt names
                    dd = dir(['*',ext]);
                    
                    % make a cell to hold names only
                    numf = length(dd);
                    fnm = cell(1,numf);
                    
                    % put filenames into cell
                    [fnm{:}] = deal(dd.name);
                    fnm = strrep(fnm, ext, '');
                    
                    % put names in the list
                    set(file_names_list, 'String', fnm);
                    
                    % initialize other fields from first file
                    str = get(file_names_list, 'String');
                    fnm = str{1};
                    
                    % add the extension and load it
                    fnm = [fnm,ext];
                    indata = load('-mat',fnm);
                    if ~isfield(indata, 'comment')
                        indata.comment = '';
                    end
                    
                    % set expt name and comment fields
                    set(expt_names_list, 'String', indata.exptnames);
                    set(comment_text, 'String', indata.comment);
                end
                
        end
    end
    
    close;
    
    if nargout
        nms = onms;
    end
    
end


function didok = put_expt_names(nms, comment)


if ~nargin
    if nargout
        didok = 0;
    end
    return;
elseif nargin < 2
    comment = '';
end

ext = '.exptlist';

% make the figure
fight = 595;
boxht = fight-95;
tagy = boxht+20;
btny = tagy+40;
sp = get(0, 'ScreenSize');

put_expt_names_fig = figure('Color',[0.8 0.8 0.8], ...
    'MenuBar','none',...
    'Position',[sp(3)-700 sp(4)-(fight)-200 600 fight], ...
    'NumberTitle','off', ...
    'Name','Put experiment names', ...
    'Tag','put_expt_names_fig');

assignin('base', 'put_expt_names_fig', put_expt_names_fig);

file_names_tag = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'Position',[10 tagy 100 20], ...
    'String','Existing files', ...
    'Style','text', ...
    'Tag','file_names_tag');

file_names_list = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'BackgroundColor',[1 1 1], ...
    'Callback','set(put_expt_names_fig,''UserData'',''file_names_list'');',...
    'FontName','Courier', ...
    'Position',[10 10 175 boxht], ...
    'String','', ...
    'Style','listbox', ...
    'Tag','file_names_list', ...
    'Value',1);

expt_names_tag = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'Position',[200 tagy 100 20], ...
    'String','Experiment names', ...
    'Style','text', ...
    'Tag','expt_names_tag');

expt_names_list = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'BackgroundColor',[1 1 1], ...
    'FontName','Courier', ...
    'Position',[200 10 175 boxht], ...
    'String',nms, ...
    'Style','listbox', ...
    'Tag','expt_names_list', ...
    'Value',1);

comment_tag = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'Position',[390 tagy 100 20], ...
    'String','Comments', ...
    'Style','text', ...
    'Tag','comment_tag');

comment_text = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'BackgroundColor',[1 1 1], ...
    'FontName','Courier', ...
    'HorizontalAlignment','left',...
    'Min',0,...
    'Max',12,...
    'Position',[390 10 200 boxht], ...
    'String',comment, ...
    'Style','edit', ...
    'Tag','comment_text');

save_btn = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'BackgroundColor',[0.733333 0.733333 0.733333], ...
    'Callback','set(put_expt_names_fig,''UserData'',''save_btn'');',...
    'Position',[80 btny 100 30], ...
    'String','Save as:', ...
    'Tag','load_btn');

cancel_btn = uicontrol('Parent',put_expt_names_fig, ...
    'Units','pixels', ...
    'BackgroundColor',[0.733333 0.733333 0.733333], ...
    'Callback','set(put_expt_names_fig,''UserData'',''cancel_btn'');',...
    'Position',[550 btny 40 30], ...
    'String','Cancel', ...
    'Tag','cancel_btn');

saveas_text = uicontrol('Parent',put_expt_names_fig, ...
    'BackgroundColor',[1 1 1], ...
    'Units','pixels', ...
    'HorizontalAlignment','left',...
    'Position',[200 btny 300 30], ...
    'String','', ...
    'Style','edit', ...
    'Tag','saveas_text');

drawnow;
pause(0.1);

% initialize
% get files with list of expt names
dd = dir(['*',ext]);

% make a cell to hold names only
numf = length(dd);
fnm = cell(1,numf);

% put filenames into cell
[fnm{:}] = deal(dd.name);
fnm = strrep(fnm, ext, '');

% put names in the list
set(file_names_list, 'String', fnm);

done = 0;
while ~done
    % wait for a click
    
    waitfor(put_expt_names_fig, 'UserData');
    cmd = get(put_expt_names_fig, 'UserData');
    set(put_expt_names_fig, 'UserData', []);
    
    switch cmd,
        case 'cancel_btn',
            exptnames = [];
            done = 1;
        case {'file_names_list', 'save_btn'},
            % get the name of the expt
            itm = get(file_names_list, 'Value');
            str = get(file_names_list, 'String');
            fnm = str{itm};
            
            % set file name to save as
            if strcmp(cmd, 'file_names_list')
                set(saveas_text, 'String', fnm);
            end
            
            if strcmp(get(gcf,'SelectionType'),'open') | strcmp(cmd, 'save_btn')  % double-click
                outfile = get(saveas_text, 'String');
                outfile = [outfile, ext];
                if isempty(outfile)
                    exptnames = [];
                    done = 1;
                else
                    exptnames = get(expt_names_list,'String');
                    comment = get(comment_text, 'String');
                    if exist(outfile, 'file')
                        overwrite = questdlg('File exists. Overwrite?');
                        if strcmp(lower(overwrite), 'yes')
                            save(outfile, 'exptnames', 'comment', '-mat');
                            done = 1;
                        end
                    else
                        save(outfile, 'exptnames', 'comment', '-mat');
                        done = 1;
                    end
                end
            end
    end
end

close;

if nargout
    didok = ~isempty(exptnames);
end




function d = rawdatadir(ui)
d = '';
menuval = get(ui.datadir_menu, 'Value');  % line selected in menu
ddnm = get(ui.datadir_menu, 'String');
if ~isempty(ddnm)
    d = ddnm{menuval};
end
    
function f = userfunc(ui)
f = '';
menuval = get(ui.userproc_menu, 'Value');  % line selected in menu
ddnm = get(ui.userproc_menu, 'String');
if ~isempty(ddnm)
    f = ddnm{menuval};
end
    
function uf = allUserFuncs(wname)

rhdl = find_rextool();

hdl = findobj(rhdl, 'Tag', 'userproc_menu');
if isempty(hdl)
    error('Working directory menu not found in REX tool window.');
elseif length(hdl) > 1
    error('Too many working directory menu handles found in REX tool window.');
else
    mnu = hdl;

	ddnm = {};
	if strcmp('empty list', get(mnu, 'String'))
		if exist(udname, 'file')
			fpath = fileparts(which(udname));
			if strcmp(pwd, fpath)
				load(udname);
				set(mnu, 'String', ddnm);
			end;
		end;
		if isempty(ddnm)
			set(mnu, 'String', 'empty list');
		end;
	else
		ddnm = get(mnu, 'String');
	end;

    if ~isempty(ddnm)
        % check for old file format
        firstentry = ddnm{1};
        if firstentry(1) == '-'
            % old format - remove first line
            strs = strvcat(ddnm);
            if size(strs,1) == 1
                ddnm = {};
            else
                ddnm = cellstr(strs(2:end,:));
            end;
            set(mnu, 'String', ddnm);
            % save the file with the first line removed
            save(udname, 'ddnm');
        end;
    end;

    if isempty(ddnm)
        set(mnu, 'String', ' ');
        if nargout
            uf = '';
        end;
        return;
    end;

    % if the master file exists, load it
    menuval = get(mnu, 'Value');  % line selected in menu
    menuchoice = ddnm{menuval};

    if ~nargin
        uf = menuchoice;
    else
        % argument was passed
        % if it's in the menu, eval
        idx = strmatch(lower(wname), lower(ddnm));
        if ~isempty(idx)  % found a match
            set(mnu, 'Value', idx);
            mnustr = get(mnu, 'String');
            uf = mnustr{idx};
        else  % no match
			warndlg({['User defined function ''',wname,''' not found in list.'],...
				'Use Edit button to add to list.',...
				['Selecting last chosen function: ',menuchoice]},...
				'User function not found');
            uf = menuchoice;
        end;
    end;
end;

