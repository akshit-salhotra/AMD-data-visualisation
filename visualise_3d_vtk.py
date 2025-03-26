import vtk
import os 
import cv2
import numpy as np

bscan_dir=r"d:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data\000003162\L\20080818\124239\OPT\Carl_Zeiss_Meditec\200X1024X200\Original\B-Scans"
images=sorted(os.listdir(bscan_dir),key=lambda x:int(x.split("_")[-1].split('.')[0].strip()))
def create_volume(data):
    # Create VTK image data object
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(data.shape)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Copy data into VTK image data object
    for z in range(data.shape[2]):
        for y in range(data.shape[1]):
            for x in range(data.shape[0]):
                image_data.SetScalarComponentFromFloat(x, y, z, 0, float(data[x, y, z]))

    # Mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(image_data)

    # Volume Property
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    # Volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume
volume=[]
# for i,image in tqdm(enumerate(images)):
for i in range(0,len(images),4):
    # if i>5:
    #     break
    image_path=os.path.join(bscan_dir+os.sep+images[i])
    image=cv2.imread(image_path,0)
    image=cv2.resize(image,(image.shape[0]//4,image.shape[1]//4))
    # print(image.shape)
    cv2.imshow('bscan',image)
    cv2.waitKey(1)
    volume.append(image.tolist())
cv2.destroyAllWindows()
data=np.array(volume)


# Renderer
renderer = vtk.vtkRenderer()

# Add volume to renderer
volume = create_volume(data)
renderer.AddVolume(volume)

# Render Window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(600, 600)

# Interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Start Visualization
render_window.Render()
render_window_interactor.Start()
