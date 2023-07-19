# GUI for rib fracture detection

In order for pathologists to gain access to the trained model for rib fracture detection on PMCT images, we created a simple and user-friendly graphical user interface (GUI). In the following paragraphs, we provide a step-by-step introduction on how to use the GUI.

**Here is a high level flow chart on the general workflow with the GUI:**

```mermaid
flowchart LR
A((input <br> image)) --> B[run prediction]
B --> C[select class]
C --> D[adjust certainty]
D --> E((output <br> image))
C --> E
E --> C
E --> D
D --> C
```