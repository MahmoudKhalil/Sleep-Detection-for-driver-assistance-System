# Driver Drownsiness Detection and Car Control System for Accident Prevention
 DESCRIPTION HERE
 
## Modules
Computer Vision  
EEG Pre-processing and Feature Extraction  
Machine Learning Algorithms  
AUTOSAR Implementation  

Used [Adrian Rosebrock implementation of the Drowsiness Detection with OpenCV](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/) for the Computer Vision Module

## Installations/Dependencies needed for the Computer Vision Module
python 3.6.x/3.7.x

```bash
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install scipy
pip install --upgrade imutils
pip install pyobjc
pip install boost
pip install cmake
```
### Installing Dlib
#### For Python 3.7.x
#### 1st Option:
```bash 
pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
```

#### 2nd Option:

Download the CMake installer and install it (https://cmake.org/download/)  
Add CMake executable path to the Enviroment Variables: set PATH="%PATH%;C:\Program Files\CMake\bin"  
note: The path of the executable could be different from C:\Program Files\CMake\bin, just set the PATH accordingly.  
Restart The Cmd or PowerShell window for changes to take effect.  
Download the Dlib source(.tar.gz) from the Python Package Index: (https://pypi.org/project/dlib/#files) extract it and enter into the folder.  
Run the installation inside the Dlib directory:  
```bash 
python setup.py install
``` 

#### 3rd Option:

```bash
pip install cmake
```
Install Visual Studio build tools from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15#).  
Make Sure that Visual Studio 2017 is installed with *Desktop Development with C++*  
In Visual Studio 2017 go to the Individual Components tab, Visual C++ Tools for Cmake, and check the checkbox under the "Compilers, build tools and runtimes" section.  
In Desktop Development with C++, check Windows 8.1 SDK and Windows 10 SDK
Then: 
```bash
pip install dlib
```

#### For Python 3.6.x
```bash
pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl
```

<!-- install Dlib (http://dlib.net/files/dlib-19.6.zip) (https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10) -->

## Installations/Dependencies needed for the EPOC+

[Install CyKit Library for Python](https://github.com/CymatiCorp/CyKit/wiki/How-to-Install-CyKIT)


## Installations/Dependencies needed for the AirSim simulator  

### Install Unreal Engine  
Download the Epic Games Launcher.  
Run the Epic Games Launcher, open the Library tab on the left pane.  
Click on the Add Versions which should show the option to download Unreal 4.18 as shown below. If you have multiple versions of Unreal installed then make sure 4.18 is set to current by clicking down arrow next to the Launch button for the version.  

```bash
pip install airsim
pip install msgpack-rpc-python
```
[AirSim APIs](https://microsoft.github.io/AirSim/docs/apis/)  
[Unreal Engine Documentation](https://docs.unrealengine.com/en-US/index.html?utm_source=editor&utm_medium=docs&utm_campaign=help_menu)  
### Setting up the environment
#### These steps are for Windows, for Linux check the further references links.
Make sure you have both VS2015/2017 and CMake either as x64 or x84  
Open x64 native toold cmd vs 2017 adminstrator  
Navigate to your vs 2017 directory on the cmd using cd C:\Program Files (x86)\Microsoft Visual Studio\2017\Community  
Clone AirSim repo using git clone https://github.com/Microsoft/AirSim.git  
Navigate to AirSim Directory using cd AirSim   
Run build.cmd from cmd, This will create ready to use plugin bits in the Unreal\Plugins folder that can be dropped into any Unreal project.  
[Further references](https://microsoft.github.io/AirSim/docs/build_windows/)  

#### For Built-in Blocks Environment
Make sure above steps are done properly without errors  
Open Epic Games Launcher and press Fix Now.  
Navigate to folder AirSim\Unreal\Environments\Blocks and run update_from_git.bat from x64 native tools  
Double click on generated .sln file to open in Visual Studio 2017 or newer and wait for it to parse the files in solution.  
Make sure Blocks project is the startup project, build configuration is set to DebugGame_Editor and Win64. Hit F5 to run.  
After the build succeed, Unreal Editor will load and open. It will take sometime, so be patient.  
Wait for the remaining shaders to compile, then Press the Play button in Unreal Editor. 
[Also See How to Use AirSim](https://github.com/Microsoft/AirSim/#how-to-use-it)  
[Using Car in AirSim](https://microsoft.github.io/AirSim/docs/using_car/)
[Further references](https://microsoft.github.io/AirSim/docs/unreal_blocks/)  

You can download extra environments from [here](https://github.com/microsoft/AirSim/releases/tag/v.1.2.2) and then unzip them into AirSim\Unreal\Environments\Blocks  

[Setting Up Custom Environments](https://microsoft.github.io/AirSim/docs/unreal_custenv/)

#### AirSim Environment
Once AirSim is set up by following above steps  
Select your Unreal project as Start Up project (for example, Blocks project) and make sure Build config is set to "Development Editor" and x64, run the project and wait for it to load then press Play.  
Navigate to C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\AirSim directory and run AirSim.sln and wait for it to load.  
In the Solution Explorer, right click on Python Environment and choose Add/Remove Python Environments and choose your Python Interpreter "Python 3.7.x" and press Ok.  
If you are using Visual Studio 2017 then just open AirSim.sln, set PythonClient as startup project and choose car\hello_car.py as your startup script.  
To change the startup project from the Solution Explorer, right-click PythonClient and choose "Set as StartUp Project".  
To set the starting file/script from the Solution Explorer, right-click car\hello_car.py or the desired file and choose "Set as Starting File" and then Run or press F5.  
You'll the see hello_car.py running and the car is moving.  

[further references](https://microsoft.github.io/AirSim/docs/unreal_proj/)  

