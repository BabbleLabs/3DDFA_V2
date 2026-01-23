% Multi-face version of import_helper_script.m
% This script plots data for all detected faces, not just face 0
% Each face is assigned a different color for easy distinction

% Set the directory where the .csv files reside
csv_file_directory = "../dumps/";

% Import data from CSV files
facecount = import_csv_data(csv_file_directory + "face_count.csv", "face_count");
faceposition = import_csv_data(csv_file_directory + "face_position.csv", "face_orientation");
mouthposition = import_csv_data(csv_file_directory + "mouth_position.csv", "mouth_position");

% Find all unique face indices in the data
unique_face_indices = unique(faceposition.face_idx);
num_faces = length(unique_face_indices);

fprintf('Found %d unique face indices: ', num_faces);
fprintf('%d ', unique_face_indices);
fprintf('\n');

% Define colors for different faces (up to 10 faces)
face_colors = [
    0.1216, 0.4667, 0.7059;  % Blue
    1.0000, 0.4980, 0.0549;  % Orange
    0.1725, 0.6275, 0.1725;  % Green
    0.8392, 0.1529, 0.1569;  % Red
    0.5804, 0.4039, 0.7412;  % Purple
    0.5490, 0.3373, 0.2941;  % Brown
    0.8902, 0.4667, 0.7608;  % Pink
    0.4980, 0.4980, 0.4980;  % Gray
    0.7373, 0.7412, 0.1333;  % Olive
    0.0902, 0.7451, 0.8118   % Cyan
];

% Set some properties of the plots
title_font_size = 20;
axis_description_font_size = 16;
tick_font_size = 13;
marker_size = 6;

% Create a figure with white background
figure('Color',[1 1 1]);

% ============================================================
% Subplot 1 (top-left): Face count on screen
% ============================================================
subplot(2,2,1);
plot(facecount.seconds, facecount.face_count);
title("Face count on screen", 'FontSize', title_font_size)
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("# of faces", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
max_number_of_faces = max(facecount.face_count);
ylim([-0.1 0.1+max_number_of_faces])
grid on

% ============================================================
% Subplot 2 (top-right): Face proximity (z coordinate)
% ============================================================
subplot(2,2,2);
hold on;
legend_entries = {};
for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    
    % Get mouth position for point type 63 for this face
    mouthposition_face = mouthposition((mouthposition.face_idx == current_face_id) & ...
                                       (mouthposition.point_type == 63), :);
    
    if ~isempty(mouthposition_face)
        color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
        plot(mouthposition_face.seconds, 100./mouthposition_face.z, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", face_colors(color_idx, :));
        legend_entries{end+1} = sprintf("Face %d", current_face_id);
    end
end

if num_faces == 1
    title(sprintf("Face %d proximity", unique_face_indices(1)), 'FontSize', title_font_size)
else
    title("Face proximity", 'FontSize', title_font_size)
    legend(legend_entries, 'Location', 'best', 'FontSize', 10)
end
xlabel("time [s]", 'FontSize', axis_description_font_size);
set(gca, 'FontSize', tick_font_size)
hold off;
grid on

% ============================================================
% Subplot 3 (bottom-left): Point 63 position (x and y)
% ============================================================
subplot(2,2,3);
hold on;
legend_entries = {};

for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    
    % Get mouth position for point type 63 for this face
    mouthposition_face = mouthposition((mouthposition.face_idx == current_face_id) & ...
                                       (mouthposition.point_type == 63), :);
    
    if ~isempty(mouthposition_face)
        color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
        base_color = face_colors(color_idx, :);
        
        % Plot X coordinate with base color
        plot(mouthposition_face.seconds, mouthposition_face.x, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", base_color);
        
        % Plot Y coordinate with darker shade of the same color
        darker_color = base_color * 0.7;
        plot(mouthposition_face.seconds, mouthposition_face.y, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", darker_color);
        
        if num_faces == 1
            legend_entries{end+1} = "x (horizontal)";
            legend_entries{end+1} = "y (vertical)";
        else
            legend_entries{end+1} = sprintf("Face %d x", current_face_id);
            legend_entries{end+1} = sprintf("Face %d y", current_face_id);
        end
    end
end

if num_faces == 1
    title(sprintf("Face %d point 63 position", unique_face_indices(1)), 'FontSize', title_font_size)
else
    title("Face point 63 position", 'FontSize', title_font_size)
end
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("coordinate value [px]", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
legend(legend_entries, 'Location', 'northwest', 'FontSize', 10)
hold off;
grid on

% ============================================================
% Subplot 4 (bottom-right): Face roll angle
% ============================================================
subplot(2,2,4);
hold on;
legend_entries = {};

for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    
    % Get face position for this face
    faceposition_face = faceposition(faceposition.face_idx == current_face_id, :);
    
    if ~isempty(faceposition_face)
        color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
        plot(faceposition_face.seconds, faceposition_face.roll, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", face_colors(color_idx, :));
        legend_entries{end+1} = sprintf("Face %d", current_face_id);
    end
end

if num_faces == 1
    title(sprintf("Face %d roll", unique_face_indices(1)), 'FontSize', title_font_size)
else
    title("Face roll", 'FontSize', title_font_size)
    legend(legend_entries, 'Location', 'best', 'FontSize', 10)
end
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("roll angle [deg]", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
ylim([-90 90])
hold off;
grid on

% Print statistics to console
fprintf('\n========================================\n');
fprintf('STATISTICS\n');
fprintf('========================================\n');
fprintf('Total frames: %d\n', height(facecount));
fprintf('Frames with faces: %d (%.1f%%)\n', ...
    sum(facecount.face_count > 0), ...
    100 * sum(facecount.face_count > 0) / height(facecount));
fprintf('Frames with multiple faces: %d (%.1f%%)\n', ...
    sum(facecount.face_count > 1), ...
    100 * sum(facecount.face_count > 1) / height(facecount));
fprintf('\n');

fprintf('Frames per face:\n');
for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    face_frames = sum(faceposition.face_idx == current_face_id);
    fprintf('  Face %d: %d frames (%.1f%%)\n', ...
        current_face_id, face_frames, ...
        100 * face_frames / height(facecount));
end
fprintf('\n');

fprintf('Average orientation per face:\n');
for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    faceposition_face = faceposition(faceposition.face_idx == current_face_id, :);
    
    avg_roll = mean(faceposition_face.roll);
    avg_pitch = mean(faceposition_face.pitch);
    avg_yaw = mean(faceposition_face.yaw);
    
    fprintf('  Face %d: roll=%.1f°, pitch=%.1f°, yaw=%.1f°\n', ...
        current_face_id, avg_roll, avg_pitch, avg_yaw);
end
fprintf('========================================\n\n');

%% Helper function to import CSV data
function imported_data = import_csv_data(csv_file_path, csv_file_type)
%import_csv_data A helper function to assist with importing data from CSV
%files
% csv_file_path: the location of the csv file in the system
% csv_file_type: a string, the following values are supported:
%   "face_count": a file with face count information
%   "face_orientation: a file with roll, pitch, and yaw information
%   "mouth_position": a file with information about position and proximity of
%       various points of interest

    % Set up column names and their types, depending from the type of the CSV file
    
    switch csv_file_type
        case "face_count"
           variable_names = ["seconds", "face_count"];
        case "face_orientation"
           variable_names = ["seconds", "face_idx", "roll", "pitch", "yaw"];
        case "mouth_position"
           variable_names = ["seconds", "face_idx", "point_type", "x", "y", "z"];
    end
    
    % Specify the number of imported variables and their type
    number_of_imported_variables = size(variable_names, 2);
    opts = delimitedTextImportOptions("NumVariables", number_of_imported_variables);
    opts.VariableTypes = repmat("double", 1, number_of_imported_variables);
    opts.VariableNames = variable_names;
    
    
    % Specify range and delimiter
    opts.DataLines = [2, Inf]; % data line 1 is a line of headers - don't import them
    opts.Delimiter = ","; % values are separated with a comma
    
    % Specify file level properties
    opts.ExtraColumnsRule = "ignore"; % ignore any extra columns
    opts.EmptyLineRule = "read"; % read all empty lines
    
    % Import the data
    imported_data = readtable(csv_file_path, opts);

end
