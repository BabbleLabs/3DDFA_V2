clear; clc; close all;

%% Settings 
% audioPath = 'inputs/';
% audioName = 'example_3.wav';
% singleTalk = [2.49 13.14; 25.19 37.79];
% doubleTalk = [13.22 25.01];

% audioPath = 'inputs/';
% audioName = 'example_4.wav';
% singleTalk = [0.85 12.44; 13.82 23.34; 24.58 37.29; 38.36 49.38; 50.30 52.61];
% doubleTalk = [];

% audioPath = 'inputs/';
% audioName = 'example_7.wav';
% singleTalk = [0.72 3.32; 11.32 15.19; 36.27 38.98; 40.99 44.17];
% doubleTalk = [3.36 11.19; 15.57 36.16; 39.14 40.99; 44.33 53.81];

% audioPath = 'inputs/';
% audioName = 'example_9.wav';
% singleTalk = [0.01 1.04; 12.32 20.22; 33.09 34.13; 35.54 41.49; 47.59 49.76];
% doubleTalk = [1.14 11.8; 20.41 32.67; 43.42 47.48];

% audioPath = 'inputs/';
% audioName = 'example_10.wav';
% singleTalk = [0.01 1.14; 8.5 11.30; 17.49 19.04; 23.94 31.25; 36.55 38.24; 47.74 56.22; 59.08 61.46];
% doubleTalk = [11.46 17.35; 19.19 23.00; 31.34 35.25; 38.31 45.47; 57.69 58.99];

% audioPath = 'inputs/';
% audioName = 'example_11.wav';
% singleTalk = [0.52 24.80];
% doubleTalk = [];

% audioPath = 'inputs/';
% audioName = 'example_12.wav';
% singleTalk = [1.28 18.02];
% doubleTalk = [];

% audioPath = 'inputs/';
% audioName = 'example_13.wav';
% singleTalk = [0.30 6.27; 19.40 27.43];
% doubleTalk = [6.84 19.2];

% audioPath = 'inputs/';
% audioName = 'example_14.wav';
% singleTalk = [0.30 6.88; 21.78 27.23];
% doubleTalk = [6.92 21.37];

% audioPath = 'inputs/';
% audioName = 'example_15.wav';
% singleTalk = [1.13 8.82; 28.85 37.92];
% doubleTalk = [8.86 28.21];

% audioPath = 'inputs/';
% audioName = 'example_16.wav';
% singleTalk = [0.57 9.20; 29.20 39.32];
% doubleTalk = [9.76 28.93];

% audioPath = 'inputs/';
% audioName = 'example_17.wav';
% singleTalk = [0.35 5.85; 18.13 25.46];
% doubleTalk = [7.01 17.91];

% audioPath = 'inputs/';
% audioName = 'example_18.wav';
% singleTalk = [0.75 6.98; 18.89 27.12];
% doubleTalk = [7.02 18.49];

% audioPath = 'inputs/';
% audioName = 'example_19.wav';
% singleTalk = [14.28 27.25; 41.38 46.25];
% doubleTalk = [0.80 13.37; 27.84 39.51];

% audioPath = 'inputs/';
% audioName = 'example_20.wav';
% singleTalk = [10.93 20.73; 33.77 39.94];
% doubleTalk = [1.08 10.20; 22.50 32.26];

% audioPath = 'inputs/';
% audioName = 'example_21.wav';
% singleTalk = [10.99 18.28; 32.07 41.20];
% doubleTalk = [1.30 10.33; 20.12 30.44];

% audioPath = 'inputs/';
% audioName = 'example_22.wav';
% singleTalk = [11.99 23.13; 36.65 40.41];
% doubleTalk = [0.82 9.39; 25.00 35.82];

% audioPath = 'inputs/';
% audioName = 'example_23.wav';
% singleTalk = [7.25 22.5; 32.23 49.35];
% doubleTalk = [0.36 6.57; 23.47 32.08];

audioPath = 'inputs/';
audioName = 'example_24.wav';
singleTalk = [7.32 21.82; 32.77 48.34];
doubleTalk = [0.55 6.07; 23.16 32.25];

%% Processing
fsTarget = 48e3;
[audio, fs] = audioread([audioPath audioName]);
[P, Q] = rat(fsTarget/fs);
audio = resample(audio, P, Q);

vad = zeros(length(repelem(audio, P, Q)), 1);
if ~isempty(singleTalk)
    for singleIdx = 1:size(singleTalk, 1)
        vad(round(singleTalk(singleIdx, 1)*fsTarget):round(singleTalk(singleIdx, 2)*fsTarget)) = 1;
    end
end
if ~isempty(doubleTalk)
    for doubleIdx = 1:size(doubleTalk, 1)
        vad(round(doubleTalk(doubleIdx, 1)*fsTarget):round(doubleTalk(doubleIdx, 2)*fsTarget)) = 2;
    end
end
save([audioPath audioName(1:end-4) '_manualVAD.mat'], 'vad');

%% Plotting
fontSize = 18;
figure(); hold on;
set(gcf, 'Position', [50 50 1800 600]);
set(gca, 'FontSize', fontSize - 2);
t = (0:length(audio)-1) / fsTarget;   % time axis in seconds
fill([-2 -1 -1 -2], [0 0 1 1], [0.0 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.25);
fill([-2 -1 -1 -2], [0 0 1 1], [1.0 0.70 0.30], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
yBot = -1.05 * max(abs(audio));
yTop =  1.05 * max(abs(audio));
for k = 1:size(singleTalk, 1)
    x1 = singleTalk(k, 1);
    x2 = singleTalk(k, 2);
    fill([x1 x2 x2 x1], [yBot yBot yTop yTop], [0.0 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.25);
end
for k = 1:size(doubleTalk, 1)
    x1 = doubleTalk(k, 1);
    x2 = doubleTalk(k, 2);
    fill([x1 x2 x2 x1], [yBot yBot yTop yTop], [1.0 0.70 0.30], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
end
plot(t, audio, 'k');
xlim([0 t(end)]);
ylim([yBot yTop]);
xlabel('Time [s]', 'FontSize', fontSize); 
ylabel('Amplitude', 'FontSize', fontSize);
title('Audio waveform with VAD labels', 'FontSize', fontSize + 4);
legend('\bfSingle speaker', '\bfDouble talk', 'Location', 'NorthEast', 'FontSize', fontSize);
