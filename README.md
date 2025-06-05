# Final Project in Software Engineering Practical Course

## 🚀 How to run

### 1. Clone the Repository

### 2. Make ROS Nodes Executable

```bash
cd what-the-quack
```

```bash
chmod +x ./packages/main/src/camera_node.py
```

```bash
chmod +x ./packages/main/src/wheel_control_node.py
```

### 3. Build the Project

```bash
dts devel build -f
```

### 4. Run on Duckiebot


```bash
dts devel run -R ROBOTNAME -L default -X
```

> **Note:** `default` refers to the launch script in the `launchers/` directory and should not be renamed.

---
