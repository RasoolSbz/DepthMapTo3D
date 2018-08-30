clear load 01_MorphableModel;
[model, msz] = load_model();
N = 1;% number of training samples
f = 1;
close all;
theta = deg2rad([0,45,-30]);%rotation angle
rotation_axis = [1,0,1];
img_count = 1;
for i = 1 : N
    img_count
    i
    % Generate a random head
    alpha = randn(msz.n_shape_dim, 1);
    beta  = randn(msz.n_tex_dim, 1);
    shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
    tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV );
    shape2 = reshape(double(shape) , 3 , length(shape)/3)';
    tex2 = reshape(double(tex) , 3 , length(shape)/3)';
    % compute depth map
    sub = struct('shape' , shape2 , 'texture' , tex2);
    faces{1}=sub;
    method = 'linear';
    xmax = 1;
    xmin = 0;
    normalize_faces;
    %shape2 = normalFaces{1}.shape;
    %R1 = rotationmat3D(40 , [0,1,0]);
    %shape_temp=shape2*R1;
    %R2 = rotationmat3D(0 , [0,0,1]);
    %shape_temp=shape_temp*R2;
    %R3 = rotationmat3D(0 , [1,0,0]);
    %shape_temp=shape_temp*R3;
    I = pointcloud2image(shape2(:,3) , shape2(:,1) , shape2(:,2) ,300 , 300);
    figure;imshow(I);
    saveas(gcf,strcat('test/',num2str(img_count),'_depth','.png'))
    C = depthMap_to_cloud(I);
    figure;subplot(1,2,1);title('original');plot3dFace_free(struct('shape' , shape2));axis on;
    subplot(1,2,2);title('recovered');plot3dFace_free(struct('shape' , C));axis on;
    % give pose to faces
    for h = 1 : length(theta) % there are 5 pose for each individual
        if theta(h) ~=0
            R = rotationmat3D(theta(h) , rotation_axis);
        else
            R = 1;
        end
        shape3 = shape2*R;
%       shape3(: , 3) = 0;
        sub = struct('shape' , shape3 , 'texture' , tex2);
        % render face and crop it
        figure;plot3dFace3(sub , model.tl);
        saveas(gcf,strcat('test/',num2str(img_count),'.png'))
        input_image = double(rgb2gray(imread(strcat('test/',num2str(img_count),'.png'))));
        input_image = input_image / 255;
        [tx , ty] = find( input_image ~= 1);
        cropped_input_image = input_image;
        cropped_input_image(: , 1:min(ty)) = [];
        cropped_input_image(: , max(ty)-min(ty):end) = [];
        cropped_input_image(1:min(tx), :) = [];
        cropped_input_image(max(tx)-min(tx):end, :) = [];
        cropped_input_image = imresize(cropped_input_image , [32 , 32]);
        figure;imshow(cropped_input_image);
        %             figure;imshow(training_input_image);
        face_data_X_2d(: , : , img_count) = cropped_input_image;
        face_data_X_3d(: , img_count) = shape2(:);
        face_data_X_depth(: , :  ,img_count) = I;
        img_count = img_count + 1;
        close all;
    end
end
%save('face_data_X_2d.mat','face_data_X_2d','-v7.3');
%save('face_data_X_3d.mat','face_data_X_3d','-v7.3');
%save('face_data_X_depth.mat','face_data_X_depth','-v7.3');