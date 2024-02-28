% Clear workspace
clear; clc;

% directory where data is stored
DIR = '';

param_file = fullfile(DIR,'airfoilDNS_parameters.h5');
data_file = fullfile(DIR,'airfoilDNS_a25f0p05.h5');
filename_grid = fullfile(DIR,'airfoilDNS_grid.h5');

dt_field = h5read(param_file,'/dt_field'); % timestep for field variables (velocity and vorticity)
Re = h5read(param_file,'/Re');
alpha_p = h5read(param_file,'/alpha_p'); % pitching amplitude (deg)


x = h5read(filename_grid,'/x');
y = h5read(filename_grid,'/y');
nx = length(x);
ny = length(y);

%% Run simple dmd code

% run dmd

meanSub = 1; % subtract mean prior to performing DMD

filename = fullfile(DIR,data_file);
ux = h5read(filename,'/ux'); % streamwise velocity
uy = h5read(filename,'/uy'); % transverse velocity

xa = h5read(filename,'/xa');
ya = h5read(filename,'/ya');
t_field = h5read(filename,'/t_field');
nt = length(t_field);

uxreshape = reshape(ux,nx*ny,nt);
uyreshape = reshape(uy,nx*ny,nt);

data = [uxreshape;uyreshape];

if meanSub
    dataMean = mean(data,2);
    data = data-dataMean*ones(1,nt);
end

r= 100; % optional truncation of SVD (set to 0 for no truncation)
[Phi, Lambda,Atilde,Amplitudes] = calc_dmd(data,r);

Eigscts = log(Lambda)/dt_field; % find eignevalues in continuous time

figure
stem(imag(Eigscts)/(2*pi), abs(Amplitudes.*(Lambda))/max(abs(Amplitudes.*(Lambda))),'-','linewidth',2)

xlim([-0.02 1.5])
ylim([0 1.2])

xlabel('Frequency'), ylabel('DMD mode amplitude (scaled)')

plot_darkmode

% contour plot parameters
MM = 0.01';
v = -1:0.1:1;
v(11)=[];

modeInd = 1;
figure
for ii = 1:6
    subplot(2,3,ii)
    % for velocity form
    [cv,ch] = contourf(x,y,transpose(reshape(real(Phi(end/2+1:end,modeInd)),nx,ny)),MM*v);
    caxis([-MM MM]);
    colormap("cool")
    set(gca,'fontsize',14)
    title(['$f = ',num2str(abs(imag(Eigscts(modeInd)))/(2*pi)),'$'],'interpreter','latex','fontsize',22)
    
    if abs(imag(Eigscts(modeInd))) > 0
        modeInd = modeInd +1; % skip plotting complex conjugate modes
    end
    modeInd = modeInd +1;
    axis equal
    axis off
    hold on
    plot(xa(:,:),ya(:,:),'k-')  % plot all airfoil locations
    %set(gcf,'position',[100 100 300 700])
    
end
sgtitle({'Leading DMD modes', 'Real($u_x$)'},'Fontsize',22,'Interpreter','latex','Color','white')
plot_darkmode
