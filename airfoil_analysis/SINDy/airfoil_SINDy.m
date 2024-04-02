DIR = 'C:\Users\Sohaib\Documents\GitHub\data_driven_control\airfoil_analysis\proper_orthogonal_decomposition';

temporal_amplitudes = fullfile(DIR,'temporal_amplitudes.h5');
params_file = fullfile(DIR,'airfoilDNS_parameters.h5');
data_file = fullfile(DIR,'airfoilDNS_a25f0p05.h5');

temporal_amplitudes = h5read(temporal_amplitudes,'/temporal_amplitudes'); 
dt = h5read(params_file,'/dt_field');

dXdt = diff(temporal_amplitudes)/dt;
X = temporal_amplitudes(1:400,:);

n = size(X, 2);
lambda = 0.3;

theta = construct_library(X, n, 2);
Xi  = least_squares(theta, dXdt, lambda, n);

cols = 2;
rows = 3;

num_plots = cols * rows;
t = h5read(data_file,'/t_field');
t = t(1:400);

X_dot_sindy = theta*Xi;
X_sindy = zeros(length(t), size(X, 2));
for i = 1:size(X, 2)
    X_sindy(:,i) = cumtrapz(t, X_dot_sindy(:,i)) + X(1,i);
end


for i = 1:num_plots
    subplot(rows, cols, i)
    plot(t, X(:, i))
    hold on
    plot(t, X_sindy(:, i))
end

legend("original", "SINDY approximation")

function Xi = least_squares(Theta, dXdt, lambda, n)
    Xi = Theta\dXdt;
    for k = 1:10
        smallinds = (abs(Xi)<lambda);
        Xi(smallinds) = 0;
        for ind = 1:n
            biginds = ~smallinds(:,ind);
            Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind);
        end
    end
end

function yout = construct_library(yin, nVars,polyorder)
    n = size(yin,1);

    ind = 1;
    % poly order 0
    yout(:,ind) = ones(n,1);
    ind = ind+1;

    % poly order 1
    for i=1:nVars
        yout(:,ind) = yin(:,i);
        ind = ind+1;
    end

    if(polyorder>=2)    % poly order 2
        for i=1:nVars
            for j=i:nVars
                yout(:,ind) = yin(:,i).*yin(:,j);
                ind = ind+1;
            end
        end
    end
end


