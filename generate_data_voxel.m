%my_addPath;
load 01_MorphableModel;
[model, msz] = load_model();
N = 5000;
f = 1;
close all;
theta = deg2rad(0);%[-45 , 0 , 45]);%rotation angle
rotation_axis = [0,1,0];
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
%     figure;plot3dFace_shape(struct('shape' , shape2));
    ptCloud = pointCloud(shape2);
    ptCloudOut = pcdownsample(ptCloud,'random',0.6);
    shape4 = ptCloudOut.Location;
%     ptCloudOut = pcdownsample(ptCloud,'gridAverage',gridStep);
%     ptCloudOut =
%     pcdownsample(ptCloud,'nonuniformGridSample',maxNumPoints);
    tex2 = reshape(double(tex) , 3 , length(shape)/3)';
    % compute depth map
    sub = struct('shape' , shape2 , 'texture' , tex2);
    faces{1}=sub;
    method = 'linear';
    xmax = 1;
    xmin = 0;
    normalize_faces;
    shape2 = normalFaces{1}.shape;
    I = pointcloud2image(shape2(:,3) , shape2(:,1) , shape2(:,2) ,200 , 200);
    figure;imshow(I);
    C = depthMap_to_cloud(I);
    figure;subplot(1,2,1);title('original');plot3dFace_free(struct('shape' , shape4));axis on;
    subplot(1,2,2);title('recovered');plot3dFace_free(struct('shape' , C));axis on;
    %% voxelize
    OUTPUTgrid = voxelize_me(200,200,200,shape2);
    figure;
    hpat = PATCH_3Darray(OUTPUTgrid , 1:200,1:200,1:200);
        view(3)
    %ptcl = convert2ptcl(permute(OUTPUTgrid , [1,2,3]));
    %%
    % give pose to faces
    for h = 1% : 3 % there are 5 pose for each individual
        if theta(h) ~=0
            R = rotationmat3D(theta(h) , rotation_axis);
        else
            R = 1;
        end
        shape3 = shape2*R;
%         shape3(: , 3) = 0;
        sub = struct('shape' , shape3 , 'texture' , tex2);
        % render face and crop it
        figure;plot3dFace3(sub , model.tl);
        saveas(gcf,'temp.png')
        input_image = double(rgb2gray(imread('temp.png')));
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
        %face_data_X_2d(: , : , img_count) = cropped_input_image;
        %face_data_X_3d(: , img_count) = shape4(:);
        %face_data_X_depth(: , :  ,img_count) = I;
        %face_data_X_vox(: , :  ,: , img_count) = OUTPUTgrid;
        %face_data_X_ptcl(: , img_count) = ptcl(:);
        save(strcat('voxel_data200/2d_',num2str(img_count),'.mat'),'cropped_input_image','-v7');
        save(strcat('voxel_data200/3d_',num2str(img_count),'.mat'),'shape4','-v7');
        save(strcat('voxel_data200/depth',num2str(img_count),'.mat'),'I','-v7');
        save(strcat('voxel_data200/vox',num2str(img_count),'.mat'),'OUTPUTgrid','-v7');
        img_count = img_count + 1;
        close all;
    end
end
%save('test/face_data_X_2d_test.mat','face_data_X_2d','-v7');
%save('test/face_data_X_3d_test.mat','face_data_X_3d','-v7');
%save('test/face_data_X_depth_test.mat','face_data_X_depth','-v7');
%save('test/face_data_X_vox_test.mat','face_data_X_vox','-v7');
%save('/Users/s2kamyab/Desktop/tmp/face_data_X_ptcl_4.mat','face_data_X_ptcl','-v7.3');