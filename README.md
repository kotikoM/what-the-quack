# Final Project — *Software Engineering Practical Course*

> _"What the Quack" — a duckiebot tale of late nights, one too many llm chats, and miracles in Python._

This project is the result of an ambitious group of students trying force small robot to follow lanes and recognize traffic signs. 
Miraculously, it sometimes works.

![Project Diagram](assets/what%20the%20quack%20project%20diagram.png)

Currently, it can detect three signs:
- 🛑 Stop  
- 🐢 Slow Down  
- 🅿️ Parking  

This is peak of robotics, engineering and coding. No more no less.

---

### Collaborators

Proudly brought to you by seven brave souls and one duckiebot:

- [@David](https://github.com/David-Mais)  
- [@mariam](https://github.com/Mariam-Katamashvili)  
- [@konstantine](https://github.com/kotikoM)  
- [@mariam](https://github.com/mariamgogo)  
- [@nika](https://github.com/Nika1337)  
- [@taso](https://github.com/Taso007)  
- [@temo](https://github.com/TemoDev)  

---

## 🛠️ How to Run

> _Warning: Running this might lead to either enlightenment or despair. Proceed at your own risk._

<sub>Note: This assumes you have a functioning Duckiebot setup and a high tolerance for debugging.</sub>

### 1. Clone the Repository

```bash
git clone git@github.com:kotikoM/what-the-quack.git

```

### 2. Make ROS Nodes Executable

```bash
cd what-the-quack
chmod +x ./packages/main/src/camera_node.py
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

## 🦆 Final Words

If it moves like a duck, sees signs like a duck, and mostly obeys Python code... it's probably this project.


---
