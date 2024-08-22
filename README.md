# Image To Litematica
A python script to convert a png file to either another image made of minecraft blocks or to a Litematica file.
### /!\ Minecraft needs to be downloaded on your computer !

## Download
- Download the zip file from the latest release (https://github.com/TeddouX/ImageToLitematica/releases/latest).
- Extract zip file contents.
- Run the exe inside (no need for more installation).

### Why a zip file
The python script was built using the pyinstaller library with the --onedir option wich makes the executable way faster at launch. This command produces a folder which I compressed into a zip file for faster download speeds.

## Example
### Input: 
![alt text](https://github.com/TeddouX/Picture-To-Litematica/blob/main/example/Screenshot%202024-07-03%20112003.png?raw=true)

### Output:
![alt text](https://github.com/TeddouX/Picture-To-Litematica/blob/main/example/Screenshot%202024-07-03%20112003-Minecraft.png?raw=true)

## Documentation
### /!\ This is useful only if the executable is ran in a command prompt !
Arguments: <br>
  - -h, --help show this help message and exit
  - --version The minecraft version that you want to use
  - --image The file that you want to be converted
  - --scale-factor The scale factor for the output
  - --to-litematica If you want the image to be converted to a litematica file
  - --to-png If you want the image to be converted to a png file
  - --name The name that the image should have. Defaults to the input image's name
  - --dominant-color Use the average color of the block else it will use the average color
  - --out-folder The output folder
  - --verbose Activate verbose
