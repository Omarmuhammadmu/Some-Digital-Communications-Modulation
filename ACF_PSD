%{
* FILE DESCRIPTION
* File: ACF_PSD.m
* Digital communication project-assignment
* Description: Statstical autocorrelation function and PSD of the ensemble
* Author:Omar Muhammad Mustafa, Omar Muhammad Tolba
* Date: 9th May 2023
%}

%% statistical autocorrelation
stat_autocorrelation = zeros(1, 700);

%Implementation of statstical autocorrelation function
Number_of_realizations = 500;               %Number of generated realization.
Number_of_bits = length(Sampled_signal);    %Number of bits per realization.

%Loop to find the statistical autocorrelation by fixing the first column 
%and vary the index of the other column to
%get R_{tau} as tau varies from tau = 0 to tau = Numbers of column in the
%ensemble(i.e., the numbers of bits per one realization)
for column = 1:Number_of_bits
    %NOTE: Relplace Tx_CompleteEnsemble_shifted with the created ensamble. (Remove this line after you edit)
    stat_autocorrelation(column) =(1/Number_of_realizations) * sum(Tx_CompleteEnsemble_shifted(:, 1) .* conj(Tx_CompleteEnsemble_shifted(:, column)));
end

%Flipping R_{tau} as tau varies from 0 to Numbers of column in the ensemble since R_{tau} is even
R_tau = [fliplr(conj(stat_autocorrelation(2:end))) stat_autocorrelation]; %Flipping R_tau to the -ve quad

%NOTE: Take a look at the names of the variables. (Remove this line after you edit)
%Plotting statistical autocorrelation
figure;
frequency_axis = ((-Number_of_bits) : (1/Tb) : (Number_of_bits));
plot(frequency_axis,abs(R_tau),'r','LineWidth',2)
title('Statistical autocorrelation');
xlabel('tau');
ylabel('R(tau)');

%% PSD (By taking the Fourier transform of ACF)
PSD = abs(fftshift(fft(R_tau))); %Power spectral density of the ACF
% Plotting the PSD
figure;
plot(frequency_axis,PSD);
