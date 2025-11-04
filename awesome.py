a=2
b=3
c=a+b
disp(c)
who
whos

rowvec=[1 2 3 4 5 6]
colvec=[1;2;3;4;5;6]

sum(colvec)
mean(rowvec)
colvec(2:4)

x1=[1 2 3;4 5 6;7 8 9]
y1=[1 2 3;4 5 6;7 8 9]

aa=x1+y1;
bb=x1.*y1;
disp(aa)
disp(bb )
transpose=x1'
inverse=inv(x1)

x=-pi:0.1:10*pi;
y=sin(x);
y3=cos(x);
figure
plot(x,y,'b',x,y3,'r')
xlabel('x')
ylabel('y')
title("sinx")
legend('sinx','cosx')

% Manual One-Hot Encoding (no toolbox needed)
Animals = ["cat"; "dog"; "bird"; "dog"; "cat"; "bird"; "cat"; "dog"];
uniqueCats = unique(Animals);
oneHot = zeros(length(Animals), length(uniqueCats));

for i = 1:length(Animals)
    oneHot(i, :) = (Animals(i) == uniqueCats);
end

encodedTable = array2table(oneHot, 'VariableNames', cellstr(uniqueCats));
finalTable = [table(Animals, 'VariableNames', {'Animal'}) encodedTable];

disp(finalTable);
writetable(finalTable, 'one_hot_strings_manual.csv');
%-----------------------------------------
classLabels = randi([1,3], 20, 1);  
T = table(classLabels, 'VariableNames', {'Class_Labels'});


numClasses = 3;
oneHot = zeros(length(classLabels), numClasses);

for i = 1:length(classLabels)
    oneHot(i, classLabels(i)) = 1;
end


encodedTable = array2table(oneHot, 'VariableNames', {'Class1', 'Class2', 'Class3'});


finalData = [T encodedTable];


writetable(finalData, 'one_hot_encoded.csv');

disp('One-hot encoded data saved as one_hot_encoded.csv');
%----------------------------------------
% Q1: Histogram Equalization and File Check
clc; clear; close all;

% Step 1: Read Image
img = imread('peppers.png');  % You can use any image file
figure, imshow(img), title('Original Image');

% Step 2: View histogram of pixel intensities (convert to grayscale first)
grayImg = rgb2gray(img);
figure, imhist(grayImg), title('Histogram of Original Image');

% Step 3: Histogram Equalization
eqImg = histeq(grayImg);
figure, imshow(eqImg), title('Histogram Equalized Image');

% Step 4: Write the adjusted image to a file
imwrite(eqImg, 'equalized_peppers.png');

% Step 5: Check the contents of the new file
fileInfo = imfinfo('equalized_peppers.png');
disp('Newly Written File Information:');
disp(fileInfo);
% Q2: Convert between image types
clc; close all;
%-----------------------------------------
img = imread('peppers.png');
figure, imshow(img), title('Original RGB Image');

% Convert RGB to Grayscale
grayImg = rgb2gray(img);
figure, imshow(grayImg), title('Grayscale Image');

% Convert Grayscale to Binary
bwImg = imbinarize(grayImg);
figure, imshow(bwImg), title('Binary Image');

% Convert RGB to Indexed image
[idx, cmap] = rgb2ind(img, 256);
figure, imshow(idx, cmap), title('Indexed Image');

% Change data type (e.g., double precision)
imgDouble = im2double(grayImg);
disp('Converted image data type:');
disp(class(imgDouble));
%--------------------------------------
% Q3: Image Resizing
clc; close all;

img = imread('peppers.png');
figure, imshow(img), title('Original Image');

% 1️⃣ Specify magnification value (e.g., 150%)
magnifiedImg = imresize(img, 1.5);
figure, imshow(magnifiedImg), title('Magnified Image (150%)');

% 2️⃣ Specify size of output image (rows x columns)
outputSize = [200 300];
resizedImg = imresize(img, outputSize);
figure, imshow(resizedImg), title('Resized Image (200x300)');

% 3️⃣ Preserve aspect ratio (using scale)
scale = 0.5;
aspectImg = imresize(img, scale);
figure, imshow(aspectImg), title('Resized (Preserved Aspect Ratio)');

% 4️⃣ Specify required height
desiredHeight = 250;
scaleH = desiredHeight / size(img,1);
heightImg = imresize(img, scaleH);
figure, imshow(heightImg), title('Resized to Desired Height (250px)');

% 5️⃣ Specify required width
desiredWidth = 400;
scaleW = desiredWidth / size(img,2);
widthImg = imresize(img, scaleW);
figure, imshow(widthImg), title('Resized to Desired Width (400px)');
%---------------------------------------------
% Q4: Rotate and Crop Image
clc; close all;

img = imread('peppers.png');
figure, imshow(img), title('Original Image');

% Rotate counterclockwise (e.g., 45 degrees)
rotatedImg = imrotate(img, 45);
figure, imshow(rotatedImg), title('Rotated Image (45° CCW)');

% Crop a region [x, y, width, height]
croppedImg = imcrop(rotatedImg, [100 100 200 200]);
figure, imshow(croppedImg), title('Cropped Image Region');
%-----------------------------------------------
% Q5: Shift Image Vertically and Horizontally
clc; close all;

img = imread('peppers.png');
figure, imshow(img), title('Original Image');

% Define shift values
xShift = 50;  % horizontal (right)
yShift = 80;  % vertical (down)

% Create translation matrix
tform = affine2d([1 0 0; 0 1 0; xShift yShift 1]);

% Apply transformation
shiftedImg = imwarp(img, tform);
figure, imshow(shiftedImg), title('Shifted Image (Right & Down)');
%---------------------------------------
% plots_demo.m
% Demonstration of various plots in MATLAB

% Clear workspace and close all figures
clc; clear; close all;

%% ---------------- PIE CHART ----------------
data = [25 35 20 10 10];
labels = {'Apples', 'Bananas', 'Grapes', 'Oranges', 'Mangoes'};

figure;
pie(data, labels);
title('Pie Chart of Fruit Distribution');
colormap(jet);  % change color scheme
legend(labels, 'Location', 'southoutside', 'Orientation', 'horizontal');

%% ---------------- BAR CHART ----------------
categories = {'A', 'B', 'C', 'D', 'E'};
values = [10 15 8 12 20];

figure;
bar(values);
title('Bar Chart Example');
xlabel('Categories');
ylabel('Values');
set(gca, 'XTickLabel', categories);  % set custom labels on x-axis
grid on;

%% ---------------- GROUPED BAR CHART ----------------
A = [4 8 6; 7 3 9; 5 6 2];
figure;
bar(A);
title('Grouped Bar Chart');
xlabel('Group');
ylabel('Value');
legend({'Type 1', 'Type 2', 'Type 3'}, 'Location', 'best');
grid on;

%% ---------------- HISTOGRAM ----------------
data = randn(1, 1000); % random normal data
figure;
histogram(data, 20); % 20 bins
title('Histogram Example');
xlabel('Value');
ylabel('Frequency');
grid on;

%% ---------------- LINE PLOT ----------------
x = 0:0.1:2*pi;
y1 = sin(x);
y2 = cos(x);

figure;
plot(x, y1, '-r', 'LineWidth', 2); hold on;
plot(x, y2, '--b', 'LineWidth', 2);
title('Line Plot of Sine and Cosine');
xlabel('x');
ylabel('y');
legend({'sin(x)', 'cos(x)'});
grid on;

%% ---------------- SCATTER PLOT ----------------
x = rand(1, 50);
y = rand(1, 50);

figure;
scatter(x, y, 70, 'filled'); % 70 = marker size
title('Scatter Plot Example');
xlabel('X Data');
ylabel('Y Data');
grid on;

%% ---------------- BOX PLOT ----------------
data1 = randn(100, 3);
figure;
boxplot(data1, {'Set 1', 'Set 2', 'Set 3'});
title('Box Plot Example');
ylabel('Values');
grid on;

%---------------------------------------
% ===============================================================
% histogram_binning_examples.m
% Demonstration of histogram binning in MATLAB
% Covers: default binning, fixed number of bins, fixed width,
% custom bin edges, retrieving counts, and comparison.
% ===============================================================

clc; clear; close all;

%% ===============================================================
% 1️⃣ DEFAULT BINNING (Automatic)
% ===============================================================
data = randn(1, 1000);   % random data from normal distribution
figure;
histogram(data);
title('Default Binning');
xlabel('Data values');
ylabel('Frequency');
grid on;


%% ===============================================================
% 2️⃣ FIXED NUMBER OF BINS
% ===============================================================
data = randn(1, 1000);
figure;
histogram(data, 10);   % 10 bins
title('Histogram with 10 Bins');
xlabel('Data values');
ylabel('Frequency');
grid on;


%% ===============================================================
% 3️⃣ FIXED BIN WIDTH
% ===============================================================
data = randn(1, 1000);
figure;
histogram(data, 'BinWidth', 0.5);
title('Histogram with Bin Width = 0.5');
xlabel('Data values');
ylabel('Frequency');
grid on;


%% ===============================================================
% 4️⃣ CUSTOM BIN EDGES
% ===============================================================
data = randn(1, 1000);
edges = -4:0.5:4;  % bins from -4 to 4 with width 0.5
figure;
histogram(data, 'BinEdges', edges);
title('Histogram with Custom Bin Edges');
xlabel('Data values');
ylabel('Frequency');
grid on;


%% ===============================================================
% 5️⃣ GET BIN COUNTS & EDGES (Without Plotting)
% ===============================================================
data = randn(1, 1000);
[counts, edges] = histcounts(data, 10);
disp('Bin counts:');
disp(counts);
disp('Bin edges:');
disp(edges);


%% ===============================================================
% 6️⃣ COMPARING DIFFERENT BINNING STYLES (Subplots)
% ===============================================================
data = randn(1, 1000);

figure;

subplot(1,3,1);
histogram(data, 10);
title('10 bins');
xlabel('Data values');
ylabel('Frequency');
grid on;

subplot(1,3,2);
histogram(data, 30);
title('30 bins');
xlabel('Data values');
ylabel('Frequency');
grid on;

subplot(1,3,3);
histogram(data, 'BinWidth', 0.2);
title('Bin Width = 0.2');
xlabel('Data values');
ylabel('Frequency');
grid on;


%% ===============================================================
% ✅ END OF SCRIPT
% ===============================================================
disp('--- All histogram binning examples executed successfully ---');
%---------------------------------------
