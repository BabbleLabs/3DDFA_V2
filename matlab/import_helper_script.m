% Set the directory where the .csv files reside
csv_file_directory = "dumps/";

% Import data from CSV files
facecount = import_csv_data(csv_file_directory + "face_count.csv", "face_count");
faceposition = import_csv_data(csv_file_directory + "face_position.csv", "face_orientation");
mouthposition = import_csv_data(csv_file_directory + "mouth_position.csv", "mouth_position");

% Implement 'queries' to CSV files to get just the desired properties out
% of the files
% Obtain mouth position for point type 63 just for face with index 0
mouthposition_zero = mouthposition((mouthposition.face_idx == 0) & (mouthposition.point_type==63), :);

% Obtain face position just for the face with face index 0
faceposition_zero = faceposition((faceposition.face_idx == 0),:);

% Set some properties of the plots
title_font_size = 20;
axis_description_font_size = 16;
tick_font_size = 13;

% Plot a figure with graphs
figure('Color',[1 1 1]);

% Create a subplot with X and Y axis data points for face that has an index
% 0 assigned
subplot(2,2,3);
hold on;
plot(mouthposition_zero.seconds, mouthposition_zero.x, "LineStyle", "none", "Marker", ".");
plot(mouthposition_zero.seconds, mouthposition_zero.y, "LineStyle", "none", "Marker", ".");
title("Face 0 point 63 position", 'FontSize', title_font_size)
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("coordinate value [px]", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
legend("x (horizontal)", "y (vertical)", 'Location', 'northwest')
hold off;
grid on

% Create a subplot with the "proximity" coordinate of all the faces
subplot(2,2,2);
plot(mouthposition_zero.seconds, mouthposition_zero.z, "LineStyle", "none", "Marker", ".");
title("Face 0 proximity", 'FontSize', title_font_size)
xlabel("time [s]", 'FontSize', axis_description_font_size);
%ylabel("coordinate value [?]")
set(gca, 'FontSize', tick_font_size)
grid on

% Create a subplot with the number of faces present for all frames
subplot(2,2,1);
plot(facecount.seconds, facecount.face_count);
title("Face count on screen", 'FontSize', title_font_size);
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("# of faces", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
max_number_of_faces = max(facecount.face_count);
ylim([-0.1 0.1+max_number_of_faces])
grid on

% Plot the roll coordinate of face with index 0 (this shows where the face
% is looking)
subplot(2,2,4);
plot(faceposition_zero.seconds, faceposition_zero.roll, "LineStyle", "none", "Marker", ".");
title("Face 0 roll", 'FontSize', title_font_size)
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("roll angle [deg]", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
ylim([-90 90])
grid on

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