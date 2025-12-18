
clear; close all; clc;

fprintf('==============================================\n');
fprintf('  Wireless Communication System Simulation and Performance Atlas\n');
fprintf('==============================================\n\n');

%  Part 1: Single-scenario Monte Carlo Simulation (QPSK + AWGN)


% Simulation parameters
num_bits_per_snr = 1e5;          % Number of bits per SNR point
max_errors       = 100;          % Early stopping: max error count
EbN0_dB          = 0:2:16;       % SNR range in dB (Eb/N0)

modulation_type  = 'QPSK';       % 'BPSK' or 'QPSK'
channel_type     = 'AWGN';       % 'AWGN' or 'Rayleigh'

% Rayleigh channel parameters (also used in Part 3 plots)
fd = 50;                         % Maximum Doppler shift (Hz)
fs = 1000;                       % Sampling frequency (Hz)

BER_simulated    = zeros(size(EbN0_dB));
BER_theoretical  = zeros(size(EbN0_dB));

fprintf('========= Part 1: Single-scenario Monte Carlo Simulation =========\n');
fprintf('Modulation: %s\n', modulation_type);
fprintf('Channel   : %s\n\n', channel_type);

for idx = 1:length(EbN0_dB)
    snr_db = EbN0_dB(idx);

    num_errors           = 0;
    num_bits_transmitted = 0;

    fprintf('SNR = %d dB ... ', snr_db);
    tic;

    % ---- Monte Carlo loop: transmit/receive until conditions are met ----
    while (num_bits_transmitted < num_bits_per_snr) && (num_errors < max_errors)
        % 1. Bit source
        tx_bits = generate_random_bits(100);
        % 2. Modulation (Tx)
        tx_signal = modulate_signal(tx_bits, modulation_type);
        % 3. Channel
        if strcmp(channel_type, 'AWGN')
            rx_signal = awgn_channel(tx_signal, snr_db, modulation_type);
        else
            [rx_signal, ~] = rayleigh_channel(tx_signal, snr_db, modulation_type, fd, fs);
        end
        % 4. Demodulation (Rx)
        rx_bits = demodulate_signal(rx_signal, modulation_type);
        % 5. Error counting
        num_errors           = num_errors + sum(tx_bits ~= rx_bits);
        num_bits_transmitted = num_bits_transmitted + length(tx_bits);
    end

    BER_simulated(idx)   = num_errors / num_bits_transmitted;
    BER_theoretical(idx) = theoretical_ber(snr_db, modulation_type, channel_type);
    elapsed_time = toc;
    fprintf('BER = %.2e (bits: %d, errors: %d, time: %.2f s)\n', ...
        BER_simulated(idx), num_bits_transmitted, num_errors, elapsed_time);
end

plot_results(EbN0_dB, BER_simulated, BER_theoretical, modulation_type, channel_type);

results.EbN0_dB          = EbN0_dB;
results.BER_simulated    = BER_simulated;
results.BER_theoretical  = BER_theoretical;
results.modulation_type  = modulation_type;
results.channel_type     = channel_type;
results.num_bits_per_snr = num_bits_per_snr;

filename = sprintf('simulation_results_%s_%s.mat', modulation_type, channel_type);
save(filename, 'results');
fprintf('\n[Part 1] Results saved to file: %s\n\n', filename);

%% ====================================================================
%  Part 2: Multi-scenario BER Comparison (BPSK/QPSK over AWGN/Rayleigh)


fprintf('========= Part 2: Multi-scenario BER Comparison =========\n');

% Scenarios to compare
scenarios = {
    'BPSK', 'AWGN';
    'QPSK', 'AWGN';
    'BPSK', 'Rayleigh';
    'QPSK', 'Rayleigh'
};

all_results = cell(size(scenarios, 1), 1);

for scenario_idx = 1:size(scenarios, 1)
    modulation_type_s = scenarios{scenario_idx, 1};
    channel_type_s    = scenarios{scenario_idx, 2};

    fprintf('\n--- Scenario %d: %s over %s channel ---\n', ...
        scenario_idx, modulation_type_s, channel_type_s);

    BER_sim = zeros(size(EbN0_dB));
    BER_theo= zeros(size(EbN0_dB));

    for idx = 1:length(EbN0_dB)
        snr_db = EbN0_dB(idx);
        num_errors           = 0;
        num_bits_transmitted = 0;

        fprintf('SNR = %d dB ... ', snr_db);

        while (num_bits_transmitted < num_bits_per_snr) && (num_errors < max_errors)
            tx_bits   = generate_random_bits(100);
            tx_signal = modulate_signal(tx_bits, modulation_type_s);

            if strcmp(channel_type_s, 'AWGN')
                rx_signal = awgn_channel(tx_signal, snr_db, modulation_type_s);
            else
                [rx_signal, ~] = rayleigh_channel(tx_signal, snr_db, modulation_type_s, fd, fs);
            end

            rx_bits   = demodulate_signal(rx_signal, modulation_type_s);
            num_errors           = num_errors + sum(tx_bits ~= rx_bits);
            num_bits_transmitted = num_bits_transmitted + length(tx_bits);
        end

        BER_sim(idx)  = num_errors / num_bits_transmitted;
        BER_theo(idx) = theoretical_ber(snr_db, modulation_type_s, channel_type_s);

        fprintf('BER = %.2e\n', BER_sim(idx));
    end

    all_results{scenario_idx}.modulation = modulation_type_s;
    all_results{scenario_idx}.channel    = channel_type_s;
    all_results{scenario_idx}.BER_sim    = BER_sim;
    all_results{scenario_idx}.BER_theo   = BER_theo;
end

% -------- Plot multi-scenario comparison --------
figure('Position', [100, 100, 1000, 700]);
colors      = {'b', 'r', 'g', 'm'};
markers     = {'o', 's', '^', 'd'};
line_styles = {'-', '-', '--', '--'};

for i = 1:length(all_results)
    semilogy(EbN0_dB, all_results{i}.BER_sim, ...
        [colors{i} markers{i} line_styles{i}], ...
        'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', colors{i});
    hold on;
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 13, 'FontWeight', 'bold');
title('Performance Comparison of Modulation Schemes and Channel Types', ...
    'FontSize', 14, 'FontWeight', 'bold');

legend_entries = cell(1, length(all_results));
for i = 1:length(all_results)
    legend_entries{i} = sprintf('%s over %s', ...
        all_results{i}.modulation, all_results{i}.channel);
end
legend(legend_entries, 'Location', 'southwest', 'FontSize', 11);

ylim([1e-5 1]);
set(gca, 'FontSize', 11);

% Simple performance statistics
fprintf('\n========= Multi-scenario Performance Summary =========\n');
for i = 1:length(all_results)
    fprintf('\n%s over %s:\n', ...
        all_results{i}.modulation, all_results{i}.channel);

    target_ber = 1e-3;
    ber_vals   = all_results{i}.BER_sim;
    idx_below  = find(ber_vals <= target_ber, 1, 'first');

    if ~isempty(idx_below)
        fprintf('  SNR to reach BER = 1e-3: ~%d dB\n', EbN0_dB(idx_below));
    else
        fprintf('  BER = 1e-3 not reached in simulated SNR range\n');
    end

    fprintf('  Minimum BER: %.2e (at SNR = %d dB)\n', ...
        min(ber_vals), EbN0_dB(end));
end

%  Part 3: Wireless Channel Characteristic Plots (Atlas)


fprintf('\n========= Part 3: Wireless Channel Characteristic Plots =========\n');
plot_wireless_channel_figures(fd, fs);
fprintf('\nAll channel plots generated.\n');
fprintf('\n========== All simulations and plots finished ==========\n');


function bits = generate_random_bits(num_bits)
    % Generate a random binary sequence of length num_bits
    bits = randi([0, 1], 1, num_bits);
end

function modulated_signal = modulate_signal(bits, modulation_type)
    % Map bits to BPSK or QPSK symbols
    switch modulation_type
        case 'BPSK'
            % 0 -> -1, 1 -> +1
            modulated_signal = 2*bits - 1;
        case 'QPSK'
            % Make sure bit length is even
            if mod(length(bits), 2) ~= 0
                bits = [bits, 0];
            end
            bits_I = bits(1:2:end);
            bits_Q = bits(2:2:end);
            I = 2*bits_I - 1;
            Q = 2*bits_Q - 1;
            % Normalize so that average symbol power is 1
            modulated_signal = (I + 1j*Q) / sqrt(2);
        otherwise
            error('Unsupported modulation type: %s', modulation_type);
    end
end

function rx_signal = awgn_channel(tx_signal, snr_db, modulation_type)
    % AWGN channel with Eb/N0 input in dB (consistent for real/complex)
    Es   = mean(abs(tx_signal).^2);       % average symbol energy (power)
    EbN0 = 10^(snr_db/10);

    % bits per symbol
    switch modulation_type
        case 'BPSK', k = 1;
        case 'QPSK', k = 2;
        otherwise,   k = 1;
    end

    % Eb = Es/k  =>  N0 = Eb / (Eb/N0) = Es/(k*EbN0)
    N0 = Es / (k * EbN0);

    if isreal(tx_signal)
        n = sqrt(N0/2) * randn(size(tx_signal));  % variance N0/2
    else
        n = sqrt(N0/2) * (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));
    end

    rx_signal = tx_signal + n;
end


function [rx_signal, h] = rayleigh_channel(tx_signal, snr_db, modulation_type, fd, fs)
    % Rayleigh fading channel using a simple Jakes-like model, followed by AWGN
    N = length(tx_signal);
    t = (0:N-1) / fs;
    num_oscillators = 8;

    h = zeros(1, N);
    for n = 1:num_oscillators
        theta_n = 2*pi*n/num_oscillators;
        phi_n   = 2*pi*rand();
        h = h + exp(1j*(2*pi*fd*t*cos(theta_n) + phi_n));
    end
    h = h / sqrt(num_oscillators);          % Normalize

    faded_signal = tx_signal .* h;
    faded_signal = faded_signal / sqrt(mean(abs(h).^2));  % Re-normalize average power

    rx_signal = awgn_channel(faded_signal, snr_db, modulation_type);
end

function bits = demodulate_signal(rx_signal, modulation_type)
    % Hard-decision demodulation for BPSK/QPSK
    switch modulation_type
        case 'BPSK'
            bits = real(rx_signal) > 0;
        case 'QPSK'
            I = real(rx_signal);
            Q = imag(rx_signal);
            bits_I = I > 0;
            bits_Q = Q > 0;
            num_symbols = length(rx_signal);
            bits = zeros(1, 2*num_symbols);
            bits(1:2:end) = bits_I;
            bits(2:2:end) = bits_Q;
        otherwise
            error('Unsupported modulation type: %s', modulation_type);
    end
end

function ber = theoretical_ber(snr_db, modulation_type, channel_type)
    % Theoretical BER for BPSK/QPSK over AWGN/Rayleigh, snr_db is Eb/N0 in dB
    snr_linear = 10^(snr_db/10);

    switch channel_type
        case 'AWGN'
            switch modulation_type
                case 'BPSK'
                    ber = 0.5 * erfc(sqrt(snr_linear));
                case 'QPSK'
                    ber = 0.5 * erfc(sqrt(snr_linear));
                otherwise
                    error('Unsupported modulation type: %s', modulation_type);
            end
        case 'Rayleigh'
            switch modulation_type
                case 'BPSK'
                    ber = 0.5 * (1 - sqrt(snr_linear / (1 + snr_linear)));
                case 'QPSK'
                    ber = 0.5 * (1 - sqrt(snr_linear / (1 + snr_linear)));
                otherwise
                    error('Unsupported modulation type: %s', modulation_type);
            end
        otherwise
            error('Unsupported channel type: %s', channel_type);
    end
end

function plot_results(EbN0_dB, BER_simulated, BER_theoretical, modulation_type, channel_type)
    % -------- Main BER figure --------
    figure('Position', [100, 100, 800, 600]);
    semilogy(EbN0_dB, BER_simulated, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    semilogy(EbN0_dB, BER_theoretical, 'r--', 'LineWidth', 2);
    grid on;

    xlabel('E_b/N_0 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('BER Performance of %s over %s Channel', modulation_type, channel_type), ...
        'FontSize', 14, 'FontWeight', 'bold');
    legend('Monte Carlo Simulation', 'Theoretical', ...
        'Location', 'southwest', 'FontSize', 11);
    ylim([1e-5 1]);
    set(gca, 'FontSize', 11);

    % -------- Combined analysis figure (4 subplots) --------
    figure('Position', [150, 150, 900, 700]);

    % Subplot 1: BER comparison
    subplot(2,2,1);
    semilogy(EbN0_dB, BER_simulated, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
    semilogy(EbN0_dB, BER_theoretical, 'r--', 'LineWidth', 2);
    grid on;
    xlabel('E_b/N_0 (dB)', 'FontSize', 10);
    ylabel('Bit Error Rate (BER)', 'FontSize', 10);
    title('BER Performance Comparison', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Simulation', 'Theoretical', 'Location', 'southwest');

    % Subplot 2: Relative error (%)
    subplot(2,2,2);
    error_percentage = abs(BER_simulated - BER_theoretical) ./ BER_theoretical * 100;
    error_percentage(BER_theoretical == 0) = 0;
    plot(EbN0_dB, error_percentage, 'g^-', 'LineWidth', 2, 'MarkerSize', 8);
    grid on;
    xlabel('E_b/N_0 (dB)', 'FontSize', 10);
    ylabel('Relative Error (%)', 'FontSize', 10);
    title('Relative Error between Simulation and Theory', ...
        'FontSize', 11, 'FontWeight', 'bold');

    % Subplot 3: Single constellation or waveform
    subplot(2,2,3);
    if strcmp(modulation_type, 'QPSK')

        axis off;
        text(0.05, 0.6, 'QPSK constellation is shown in the wireless channel atlas (Part 3).', ...
            'FontSize', 10);
    else
        sample_bits    = randi([0, 1], 1, 200);
        sample_symbols = modulate_signal(sample_bits, 'BPSK');
        snr_sample     = 10;
        noisy_symbols  = awgn_channel(sample_symbols, snr_sample, 'BPSK');

        plot(1:length(noisy_symbols), noisy_symbols, 'b-', 'LineWidth', 1); hold on;
        plot(1:length(sample_symbols), sample_symbols, 'r--', 'LineWidth', 2);
        grid on;
        xlabel('Symbol Index', 'FontSize', 10);
        ylabel('Amplitude', 'FontSize', 10);
        title(sprintf('BPSK Waveform (AWGN, SNR = %d dB)', snr_sample), ...
            'FontSize', 11, 'FontWeight', 'bold');
        legend('Received Signal', 'Transmitted Signal', 'Location', 'best');
    end

    % Subplot 4: System parameters and performance summary
    subplot(2,2,4);
    axis off;
    param_text = {
        '========== System Parameters ===========' ...
        sprintf('Modulation: %s', modulation_type) ...
        sprintf('Channel   : %s', channel_type) ...
        ' ' ...
        '========== Performance Metrics ===========' ...
        sprintf('Min SNR: %d dB', EbN0_dB(1)) ...
        sprintf('Max SNR: %d dB', EbN0_dB(end)) ...
        sprintf('Min BER: %.2e', min(BER_simulated)) ...
        sprintf('Max BER: %.2e', max(BER_simulated)) ...
        ' ' ...
        sprintf('Average Relative Error: %.2f%%', mean(error_percentage))
    };
    text(0.1, 0.9, param_text, 'FontSize', 10, 'VerticalAlignment', 'top', ...
        'FontName', 'Courier New');
end

function plot_wireless_channel_figures(fd, fs)


    snr_db = 10;
    Nsym_q = 2000;

    %% 1) QPSK constellation: AWGN vs Rayleigh
    bits_q = randi([0 1], 1, 2*Nsym_q);
    s_q    = modulate_signal(bits_q, 'QPSK');
    rx_awgn_q = awgn_channel(s_q, snr_db, 'QPSK');
    [rx_ray_q, h_q] = rayleigh_channel(s_q, snr_db, 'QPSK', fd, fs);

    figure('Position', [100, 100, 900, 400]);
    subplot(1,2,1);
    plot(real(rx_awgn_q), imag(rx_awgn_q), 'b.', 'MarkerSize', 3); hold on;
    ideal = [-1-1j, -1+1j, 1-1j, 1+1j] / sqrt(2);
    plot(real(ideal), imag(ideal), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    grid on; axis equal;
    xlabel('In-phase (I)');
    ylabel('Quadrature (Q)');
    title('QPSK Constellation - AWGN (SNR = 10 dB)');
    legend('Received Symbols', 'Ideal Symbols', 'Location', 'best');

    subplot(1,2,2);
    plot(real(rx_ray_q), imag(rx_ray_q), 'b.', 'MarkerSize', 3); hold on;
    plot(real(ideal), imag(ideal), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    grid on; axis equal;
    xlabel('In-phase (I)');
    ylabel('Quadrature (Q)');
    title('QPSK Constellation - Rayleigh + AWGN (SNR = 10 dB)');
    legend('Received Symbols', 'Ideal Symbols', 'Location', 'best');

    %% 2) Rayleigh fading |h(t)| vs time
    %    and 3) |h| histogram + theoretical Rayleigh PDF
    t = (0:length(h_q)-1) / fs;
    figure('Position', [150, 150, 900, 500]);
    subplot(2,1,1);
    plot(t, abs(h_q), 'b-');
    grid on;
    xlabel('Time (s)');
    ylabel('|h(t)|');
    title('Rayleigh Fading Channel Gain vs Time');

    subplot(2,1,2);
    mag_h = abs(h_q);
    histogram(mag_h, 40, 'Normalization', 'pdf'); hold on;
    % Theoretical Rayleigh PDF fit
    sigma_hat = sqrt(mean(mag_h.^2)/2);
    r = linspace(0, max(mag_h)*1.1, 200);
    ray_pdf = (r./sigma_hat.^2) .* exp(-r.^2 ./ (2*sigma_hat.^2));
    plot(r, ray_pdf, 'r', 'LineWidth', 2);
    grid on;
    xlabel('|h|');
    ylabel('Probability Density');
    title('Rayleigh Magnitude Histogram and Theoretical PDF');
    legend('Simulated Histogram', 'Theoretical Rayleigh PDF', 'Location', 'best');

    %% 4) Simple BPSK eye diagram (oversampled)

Nbits_b = 1000; 
bits_b = randi([0 1], 1, Nbits_b); 
s_b = 2*bits_b - 1;

L = 8;           
rolloff = 0.5;   
span = 6;           

h_rrc = rcosdesign(rolloff, span, L, 'sqrt'); 


tx_shaped = upfirdn(s_b, h_rrc, L);


rx_b_ov = awgn_channel(tx_shaped, snr_db, 'BPSK');

figure('Position', [200, 200, 600, 400]); 
hold on;
Tspan = 2;          
delay = span * L / 2; 

for k = 1 : (floor(length(rx_b_ov)/L) - Tspan - span)

    start_idx = delay + (k-1)*L + 1;
    end_idx   = start_idx + Tspan*L - 1;
    
    if end_idx > length(rx_b_ov), break; end
    
    seg = rx_b_ov(start_idx : end_idx);
    t_eye = (0:length(seg)-1) / L; 
    
    plot(t_eye, seg, 'b-', 'LineWidth', 0.1, 'Color', [0 0 1 0.3]); % 设置透明度，效果更好
end

grid on;
title(sprintf('BPSK Eye Diagram (Raised Cosine, \\alpha=%.1f, SNR=%d dB)', rolloff, snr_db));
xlabel('Normalized Time (symbol periods)');
ylabel('Amplitude');
axis([0 Tspan -1.5 1.5]); 
    %% 5) AWGN noise histogram + Gaussian PDF
    noise_q = rx_awgn_q - s_q;
    noise_real = real(noise_q);

    figure('Position', [250, 250, 600, 400]);
    histogram(noise_real, 40, 'Normalization', 'pdf'); hold on;
    mu_hat    = mean(noise_real);
    sigma2_hat= var(noise_real);
    sigma_hat = sqrt(sigma2_hat);
    x = linspace(mu_hat - 4*sigma_hat, mu_hat + 4*sigma_hat, 200);
    gauss_pdf = 1/(sqrt(2*pi)*sigma_hat) * exp(-(x-mu_hat).^2 / (2*sigma2_hat));
    plot(x, gauss_pdf, 'r', 'LineWidth', 2);
    grid on;
    xlabel('Noise Sample (Real Part)');
    ylabel('Probability Density');
    title('AWGN Noise Histogram and Gaussian PDF Fit');
    legend('Noise Histogram', 'Gaussian PDF', 'Location', 'best');
end
