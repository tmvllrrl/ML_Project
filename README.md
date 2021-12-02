# ML Project Submission

## Initial Setup
ALE/Gym will ask if you have permission to run the games under a researcher's license and give you a command for to enter. Enter that command as this will allow you to get the Atari 2600 games and run the code.

## DQN
In order to run DQN, one can simply navigate to the DQN folder and run:
```shell
python3 pipeline.py
```
Or open the folder in an IDE and run it from there. 

If you want to change the game that is being played, then you will have to change the source code directly (it is currently set to play Qbert with frames). To change the game, you would line 21 in pipeline.py to a different Atari 2600 game followed by "-v0" (for example, "Pong-v0").

## DDQN w/ PER
Similar instructions as above with DQN; however, the file to run is the main.py file. So, one can navigate to the DDQN folder and run:
```shell
python3 main.py
```
By default this will run the DDQN agent on Qbert. 

Many CLI arugments can be found in main.py or by running:
```shell
python3 main.py --help
```