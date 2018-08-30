function h = plot3dFace3(sub , tl)
%% plotting the 3d shape
% The solution is to use Delaunay triangulation. Let's look at some
% info about the "tri" variable.
% if max(points_3D(:))>1
%     points_3D = mapminmax(points_3D , 0 , 1);
% end
% figure;
tri = tl;% delaunay(sub.shape(:, 1), sub.shape(:, 2));
% Plot it with TRISURF
if max(sub.texture(:)) > 1
    h = trisurf(tri, sub.shape(:, 1),sub.shape(:, 2), sub.shape(:, 3) , 'FaceVertexCData', sub.texture./max(sub.texture(:)), 'FaceColor' , 'interp', 'EdgeColor','none', 'FaceLighting', 'phong');%, 'LineWidth',5);%'CDataMapping','scaled'););
else
    h = trisurf(tri, sub.shape(:, 1),sub.shape(:, 2), sub.shape(:, 3) , 'FaceVertexCData', sub.texture, 'FaceColor' , 'interp', 'EdgeColor','none', 'FaceLighting', 'phong');%, 'LineWidth',5);%'CDataMapping','scaled'););
end
% axis vis3d
% axis tight
% view([-50 , -30 , 300]);
% view([0 , 0 , 300]);
view(2)
drawnow;
camlight('headlight');
material([.5, .5, .1 1  ])
%lighting none
% camlight right
% set(gca , 'Projection' , 'perspective');
%light('Position',[0 300 0],'Style','ambient')
% Clean it up
axis off

% l = light('Position',[-50 -15 29]);
% l = light('Position',[0 0.5 0.5]);
% set(gca,'CameraPosition',[208 -50 7687])
%lighting phong
% shading interp
% colorbar EastOutside
% colormap summer