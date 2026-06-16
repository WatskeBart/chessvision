# chessvision aka Grandmaster Yolo

Tracking chess pieces via camera

## Hardware setup

```mermaid
flowchart LR
    Internet(("🌐 Internet"))
    Laptop["💻 Laptop<br/>shares its Wi-Fi Internet<br/>connection (acts as router)"]

    subgraph netA["Network A — RPi Wi-Fi hotspot"]
        direction LR
        Phone["📱 Android phone<br/>IP camera app<br/>(on tripod above board)"]
        RPi["🖥️ Raspberry Pi 4B<br/>hosts the hotspot<br/>runs the vision pipeline"]
        Phone -- "Wi-Fi" --> RPi
    end

    Internet -- "Wi-Fi" --> Laptop
    Laptop -- "Ethernet (Internet sharing)" --> RPi
```

| Component         | Role                                                                  |
|--------------------|------------------------------------------------------------------------|
| Android phone       | Mounted on a tripod above the board, streams video via an IP camera app |
| Raspberry Pi 4B      | Hosts its own Wi-Fi hotspot for the phone to join; runs the vision pipeline (`detect_chessboard.py` / `main.py`) |
| Laptop                | Connected to the RPi over Ethernet; shares its own Wi-Fi Internet connection with the RPi (acts as a router) |
| Internet                | Reached by the laptop over Wi-Fi, then passed through to the RPi via Ethernet |

## Proof of Concepts

### POC 1

Detect the chessboard

### POC 2

Track chess moves via individual chess pieces