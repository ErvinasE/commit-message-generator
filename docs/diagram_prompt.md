# Diagram prompt

The prompt below was given to an AI assistant to design the system. It produced the
PlantUML diagrams first and then the Mermaid versions, saved in [`diagrams/`](diagrams/):

| Diagram | PlantUML | Mermaid |
|---|---|---|
| Architecture | [`architecture.puml`](diagrams/architecture.puml) | [`architecture.mmd`](diagrams/architecture.mmd) |
| Sequence | [`sequence.puml`](diagrams/sequence.puml) | [`sequence.mmd`](diagrams/sequence.mmd) |

`architecture.png` is the Mermaid architecture diagram rendered to an image (via
mermaid.ink), because Colab cannot render Mermaid or PlantUML.

## Prompt

```
I am a software engineering student. Design an information system that uses
generative AI, on this topic: "Commit message and changelog generator".

The system takes a git diff and uses the Gemini API to produce a Conventional
Commits message. It validates the output format and, if invalid, retries with the
validation errors as feedback (at most 2 retries). It can also generate a grouped
changelog from a list of commits and checks that every commit is covered.

First, briefly describe the system. Then produce PlantUML diagrams:
1. a component/architecture diagram
2. a sequence diagram covering both the commit message flow and the changelog flow

Then convert the same two diagrams to Mermaid. Keep component names consistent
between the PlantUML and Mermaid versions.
```
