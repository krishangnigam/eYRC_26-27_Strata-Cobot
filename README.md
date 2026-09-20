# e-Yantra Robotics Competition 2026-27 — StrataCobot (SC)

Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic

## 1. Install the task dependencies

From this directory:

```bash
./requirements.sh
```

It installs what the task needs on top of an existing ROS 2 Jazzy. It does not install
ROS itself.

## 2. Build the workspace

```bash
cd ..
colcon build
source install/setup.bash
```

Add the `source` line to your `~/.bashrc`.

## 3. Start the simulation

Each task has its own launch file. Leave it running in its own terminal.

```bash
ros2 launch eyantra_kepler_colony task0.launch.py    # Task 0 — the whole arena
ros2 launch eyantra_kepler_colony task1a.launch.py   # Task 1A — ore perception
ros2 launch eyantra_kepler_colony task1b.launch.py   # Task 1B — arm waypoints
ros2 launch eyantra_kepler_colony task1c.launch.py   # Task 1C — eBot navigation
```

None of them starts a solution node. That part is the task.

Add `rviz:=false` to any of them to start without RViz.

## 4. Write and run your node

Your code goes in the **`algorithms`** package. Copy the boilerplate for your subtask into
the matching folder and rename it to whatever you like:

```bash
cd algorithms
cp boilerplate/task1b_boilerplate.py scripts/task1b/arm_waypoints.py
chmod +x scripts/task1b/arm_waypoints.py
```

Add that path to the `SCRIPTS` list in `algorithms/setup.py`. **The entry is a path from
the package root, not a bare file name:**

```python
SCRIPTS = [
    'scripts/task1b/arm_waypoints.py',
]
```

Build, source, and run it. The executable name is the **file name, `.py` included**:

```bash
cd ..
colcon build --packages-select algorithms
source install/setup.bash
ros2 run algorithms arm_waypoints.py
```

Three things that catch people out:

* A file is not installed until it is in `SCRIPTS` **and** you have built again.
* Without `chmod +x` it installs fine and then refuses to start.
* `ros2 run` starts the **installed** copy, so rebuild and re-source after every edit.

`ros2 pkg executables algorithms` lists what is actually installed.

The topics and services your node works with are on the task page.

## 5. Run the evaluator

In a second terminal, with the simulation still running and the workspace sourced:

```bash
./eyrc-sc-evaluator --task 1B --team-id 1455
```

Both arguments can be left out and it will ask for them. It checks your setup, prints
`Ready`, and records your run. **Start your node only after `Ready`.** Press `q` when your
run has finished (`Ctrl-C` if you are not on an interactive terminal).

It writes a **`result.zip`** into a folder named for your team and subtask. Do not unpack
it, rebuild it or rename anything inside it — it is signed, and editing it makes it
unreadable.

## 6. Submit

Copy your node into the folder the evaluator wrote, **renamed as the task page says**,
compress the contents together, and upload that one archive.

```
SC#1455_task1B.zip
  ├── result.zip          from the evaluator, untouched
  └── task1B.py           your node, renamed
```

**Take the archive name, the file names and the full list of what goes in from the task
page** — it differs per subtask, and some subtasks ask for more than these two files.

---

The UR7e description in `ur_description` is modified by e-Yantra from the upstream
Universal Robots ROS 2 description; see `ur_description/LICENSE`.
