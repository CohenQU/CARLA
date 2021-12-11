## System
- Windows

## Task settings
- Download a modified version of CARLA_0.8.4 [CarlaSimulator.zip](https://drive.google.com/file/d/13elrKQiOmYiWKi7cAynMP-rRhzuZZecS/view?usp=sharing). Extract the folder and assumer the directory of CARLA is `C:/{Your_Directory}/CarlaSimulator`.
- Download [ControlModule.zip](https://drive.google.com/file/d/186wh_HvAtVDlL0tU-O1QI12jsrvIjECI/view?usp=sharing). Extract the folder to the directory `C:/{Your_Directory}/CarlaSimulator/PythonClient`
- Replace `./ControlModule/controller2d.py` with the new `controller2d.py`

## Running the controller
- open one Command Prompt/Terminal window, run the following command to open the map
<pre><code>cd C:/{Your_Directory}/CarlaSimulator</pre></code>
<pre><code>CarlaUE4.exe /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=30</pre></code>
- open another Command Prompt/Terminal window run the following command to run the test
<pre><code>cd C:/{Your_Directory}/CarlaSimulator/PythonClient/ControlModule</pre></code>
<pre><code>python module_7.py</pre></code>