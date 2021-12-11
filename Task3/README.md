## System
- Windows

## Task settings
- Check out the repository of [CARLA](https://github.com/carla-simulator/carla/releases/tag/0.9.12/)
- Download and unzip [CARLA_0.9.12.zip](https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/CARLA_0.9.12.zip)
- Download additional maps [AdditionalMaps_0.9.12.zip](https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/AdditionalMaps_0.9.12.zip) and unzip the folders to CARLA source folder.
- Move `CarEnv.py` to `C:/{Your_Directory}/CarlaSimulator/PythonAPI/examples`
- Start the CARLA Simulator
<pre><code>cd C:/{Your_Directory}/CarlaSimulator</code></pre>
<pre><code>CarlaUE4.exe</code></pre>
- Switch to Town06
<pre><code>cd C:/{Your_Directory}/CarlaSimulator/PythonAPI/util</code></pre>
<pre><code>python config.py --map Town06</code></pre>
- Run the DQN model
<pre><code>cd C:/{Your_Directory}/CarlaSimulator/PythonAPI/examples</code></pre>
<pre><code>python CarEnv.py</code></pre>