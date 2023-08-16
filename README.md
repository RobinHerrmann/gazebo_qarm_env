# gazebo_qarm_env
Provides OpenAI Gym based Envrionment for Reinforcement Learning with gazebo_qarm_sim.

---

Dieses Repository beinhaltet eine Implementation eines [OpenAI Gym Environments](https://gymnasium.farama.org/api/env/) für Nutzung mit einer [Simulation des Quanser QArm Gelenkarmroboters in Gazebo](https://github.com/deltawafer/gazebo_qarm_sim).

---

## Nutzung

Zur Nutzung müssen die Dateien entsprechend abgelegt werden. Anschließend kann die Environment-Klasse importiert und genutzt werden. Dabei muss die Simulation gemäß Installationsanleitung laufen. Es kann ein beliebieger Lernagent implementiert werden. Innerhalb der QArmEnv.py-Datei müssen ggfs. Änderungen an der Observation, Aktion und Rewardfunktion getätigt werden. Die entsprechenden Stellen sind kommentiert.

---

## Zitieren

```
@software{gazebo_qarm_env,
  author = {Herrmann, Robin},
  title = {{gazebo_qarm_env}},
  url = {https://github.com/RobinHerrmann/gazebo_qarm_env},
  version = {1.0},
  year = {2023}
}
```
